import concurrent.futures
import heapq
import os
import random
from collections import Counter, deque
from contextlib import redirect_stdout
from functools import partial

import ase
import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import (
    complete_cell,
    conditional_find_mic,
    get_distances,
    minkowski_reduce,
)
from ase.io import read, write
from ase.neighborlist import NeighborList, build_neighbor_list, natural_cutoffs
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops

    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


VALENCE_OPTIONS = {
    "H": [(1, 0)],
    "B": [(3, 0), (4, 3)],
    "C": [(4, 0), (3, 6), (2, 8)],
    "N": [(3, 0), (4, 2), (2, 4), (1, 8), (5, 10)],
    "O": [(2, 0), (1, 3), (3, 8)],
    "F": [(1, 0)],
    "Si": [(4, 0)],
    "P": [(3, 0), (5, 2), (4, 6)],
    "S": [(2, 0), (4, 2), (6, 4), (1, 6), (3, 7)],
    "Cl": [(1, 0)],
    "As": [(3, 0), (5, 2), (4, 6)],
    "Se": [(2, 0), (4, 2), (6, 4), (1, 6)],
    "Br": [(1, 0)],
    "Te": [(2, 0), (4, 2), (6, 4)],
    "I": [(1, 0)],
}

NON_METAL_LIKE = {
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Se",
    "Br",
    "I",
    "As",
    "Te",
}

NO_MULTIPLE_BOND = {"H", "F", "Cl", "Br", "I"}
TRIPLE_BOND_ELEMENTS = {"C", "N", "P"}
RD_BOND_TYPE_FROM_ORDER = {
    1: Chem.BondType.SINGLE if HAS_RDKIT else None,
    2: Chem.BondType.DOUBLE if HAS_RDKIT else None,
    3: Chem.BondType.TRIPLE if HAS_RDKIT else None,
}
CCDC_TYPE_FROM_ORDER = {1: "S", 2: "D", 3: "T"}


def get_cycles(assignment):
    n = len(assignment)
    visited = [False] * n
    cycles = []
    for i in range(n):
        if visited[i]:
            continue
        cycle = []
        curr = i
        while not visited[curr]:
            visited[curr] = True
            cycle.append(curr)
            curr = assignment[curr]
        cycles.append(cycle)
    return cycles


def get_invalid_cycles(cycles, node_vecs, edge_vecs):
    invalid_cycles = []
    for cycle in cycles:
        sum_vec = np.zeros(3, dtype=int)
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            sum_vec += node_vecs[u] + edge_vecs[u, v]

        if np.all(sum_vec == 0):
            invalid_cycles.append(cycle)

    return invalid_cycles


def solve_minimal_path(dist_matrix, node_vecs, edge_vecs):
    BIG_M = 1e9
    pq = []
    counter = 0

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    cost = dist_matrix[row_ind, col_ind].sum()
    heapq.heappush(pq, (cost, counter, frozenset(), col_ind))

    visited_states = set()

    while pq:
        cost, _, forbidden_edges, assignment = heapq.heappop(pq)

        cycles = get_cycles(assignment)
        invalid_cycles = get_invalid_cycles(cycles, node_vecs, edge_vecs)
        if not invalid_cycles:
            return cost, assignment

        invalid_cycle = min(invalid_cycles, key=len)

        for i in range(len(invalid_cycle)):
            u = invalid_cycle[i]
            v = invalid_cycle[(i + 1) % len(invalid_cycle)]

            new_forbidden = set(forbidden_edges)
            if (u, v) in new_forbidden:
                continue
            new_forbidden.add((u, v))
            new_forbidden_frozen = frozenset(new_forbidden)

            if new_forbidden_frozen in visited_states:
                continue
            visited_states.add(new_forbidden_frozen)

            dist_matrix_mod = dist_matrix.copy()
            for fu, fv in new_forbidden_frozen:
                dist_matrix_mod[fu, fv] = BIG_M

            r, c = linear_sum_assignment(dist_matrix_mod)
            new_cost = dist_matrix_mod[r, c].sum()

            if new_cost < BIG_M:
                counter += 1
                heapq.heappush(pq, (new_cost, counter, new_forbidden_frozen, c))

    return None, None


def get_total_cell_offset(nl: NeighborList, idx_a: int, idx_b: int):
    queue = deque([(idx_a, np.zeros(3, dtype=int))])
    visited = {idx_a}

    while queue:
        current_idx, current_offset = queue.popleft()
        if current_idx == idx_b:
            return current_offset

        neighbors, offsets = nl.get_neighbors(current_idx)
        for neighbor, offset in zip(neighbors, offsets):
            if neighbor not in visited:
                visited.add(neighbor)
                next_offset = current_offset + offset
                queue.append((neighbor, next_offset))

    raise ValueError(
        f"Atoms {idx_a} and {idx_b} are disconnected and do not belong to the same molecule."
    )


def kth_periodic_distance(
    atoms, i, j, k=2, distinct_distance=False, tol=1e-6, max_shell=5, return_vector=True
):
    if k < 1:
        raise ValueError("k must be a positive integer")

    pbc = np.asarray(atoms.pbc, dtype=bool)
    if pbc.ndim == 0:
        pbc = np.repeat(pbc, 3)

    if not np.any(pbc):
        v = atoms.positions[j] - atoms.positions[i]
        d = np.linalg.norm(v)
        if k != 1:
            raise ValueError("A nonperiodic system has only one image, so k must be 1")
        return (d, v) if return_vector else d

    cell = np.asarray(complete_cell(atoms.cell), dtype=float)
    rcell, _ = minkowski_reduce(cell, pbc=pbc)
    rcell = np.asarray(rcell, dtype=float)

    v_cart = atoms.positions[j] - atoms.positions[i]

    inv_rcell = np.linalg.inv(rcell)

    heights = 1.0 / np.linalg.norm(inv_rcell, axis=0)
    h_min = np.min(heights[pbc])

    f = v_cart @ inv_rcell

    f_wrapped = f.copy()
    f_wrapped[pbc] -= np.floor(f_wrapped[pbc] + 0.5)

    for N in range(max_shell + 1):
        grid_axes = [
            np.arange(-N, N + 1, dtype=float) if periodic else np.array([0.0])
            for periodic in pbc
        ]
        grids = np.meshgrid(*grid_axes, indexing="ij")
        shifts = np.stack(grids, axis=-1).reshape(-1, 3)

        frac_all = f_wrapped[None, :] + shifts
        vecs = frac_all @ rcell
        dists = np.linalg.norm(vecs, axis=1)

        order = np.argsort(dists)
        d_sorted = dists[order]
        v_sorted = vecs[order]

        if distinct_distance:
            picked = []
            for d, vec in zip(d_sorted, v_sorted):
                if not picked or abs(d - picked[-1][0]) > tol:
                    picked.append((d, vec))
                if len(picked) >= k:
                    break

            if len(picked) < k:
                continue

            dk, vk = picked[k - 1]
        else:
            if len(d_sorted) < k:
                continue
            dk = d_sorted[k - 1]
            vk = v_sorted[k - 1]

        outside_lb = (N + 0.5) * h_min

        if dk < outside_lb - tol:
            return (dk, vk) if return_vector else dk

    raise RuntimeError(
        f"Strict convergence was not reached within max_shell={max_shell}; "
        "increase max_shell and retry."
    )


def find_and_link_sites(atoms: ase.Atoms, nl: NeighborList):
    if "label" not in atoms.arrays:
        return None

    labels = atoms.arrays["label"]

    x_sites_idx = [
        i for i, lbl in enumerate(labels) if isinstance(lbl, str) and "x_site" in lbl
    ]
    y_sites_idx = [
        i for i, lbl in enumerate(labels) if isinstance(lbl, str) and "y_site" in lbl
    ]

    if (
        len(x_sites_idx) == 0
        or len(y_sites_idx) == 0
        or len(x_sites_idx) != len(y_sites_idx)
    ):
        return None

    matrix = nl.get_connectivity_matrix(sparse=True).tolil()
    n_components, component_list = sparse.csgraph.connected_components(matrix.tocsr())

    paired_indices = []
    for mol_idx in range(n_components):
        atom_indices = np.where(component_list == mol_idx)[0].tolist()
        tmp_x_sites_idx = [idx for idx in x_sites_idx if idx in atom_indices]
        tmp_y_sites_idx = [idx for idx in y_sites_idx if idx in atom_indices]
        if len(tmp_x_sites_idx) == 0 and len(tmp_y_sites_idx) == 0:
            continue
        if len(tmp_x_sites_idx) != 1 or len(tmp_y_sites_idx) != 1:
            return None
        paired_indices.append((tmp_x_sites_idx[0], tmp_y_sites_idx[0]))

    eff_components = len(paired_indices)
    if eff_components == 0:
        return None

    pbc = atoms.get_pbc()
    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)

    x_indices, y_indices = zip(*paired_indices)
    x_positions = atoms.positions[list(x_indices)]
    y_positions = atoms.positions[list(y_indices)]
    x2y_vecs_mic, x2y_distances = get_distances(
        x_positions, y_positions, cell=cell, pbc=pbc
    )
    x2y_vecs_raw = y_positions[np.newaxis, :, :] - x_positions[:, np.newaxis, :]
    x2y_offsets = np.round((x2y_vecs_mic - x2y_vecs_raw) @ inv_cell).astype(int)

    y2x_offsets = np.zeros((eff_components, 3), dtype=int)
    for i in range(eff_components):
        x_idx, y_idx = paired_indices[i]
        y2x_offsets[i] = get_total_cell_offset(nl, y_idx, x_idx)

        # Break self-loops by selecting the next-nearest periodic image.
        if np.all((y2x_offsets[i] + x2y_offsets[i, i]) == 0):
            x2y_dist_tmp, x2y_vec_tmp = kth_periodic_distance(atoms, x_idx, y_idx, k=2)
            x2y_distances[i, i] = x2y_dist_tmp
            x2y_vecs_mic[i, i] = x2y_vec_tmp
            x2y_offsets[i, i] = np.round(
                (x2y_vec_tmp - x2y_vecs_raw[i, i]) @ inv_cell
            ).astype(int)

    x2y_dist_sqr = x2y_distances**2
    cost, assignment = solve_minimal_path(x2y_dist_sqr, y2x_offsets, x2y_offsets)

    if cost is None or assignment is None:
        return None

    new_links = []
    for i in range(len(assignment)):
        j = assignment[i]
        x_idx = paired_indices[i][0]
        y_idx = paired_indices[j][1]
        offset = x2y_offsets[i, j]
        new_links.append((x_idx, y_idx, offset))

    return matrix.tocsr(), new_links


def is_probable_metal(symbol: str) -> bool:
    return symbol not in NON_METAL_LIKE


def supports_multiple_bond(symbol: str) -> bool:
    if symbol in NO_MULTIPLE_BOND:
        return False
    if is_probable_metal(symbol):
        return False
    return True


def get_candidate_valences(symbol: str, degree: int):
    options = VALENCE_OPTIONS.get(symbol)
    if options is None:
        return [(degree, 0)]

    candidates = []
    seen = set()
    for val, penalty in options:
        if val < degree:
            continue
        if val not in seen:
            candidates.append((val, penalty))
            seen.add(val)

    if degree not in seen:
        candidates.append((degree, 50))

    return candidates


def get_max_extra_valence(symbol: str, degree: int) -> int:
    candidates = get_candidate_valences(symbol, degree)
    if not candidates:
        return 0
    return max(max(0, val - degree) for val, _penalty in candidates)


def estimate_extra_bond_capacity(
    symbol1: str, symbol2: str, degree1: int, degree2: int, is_new: bool
) -> int:
    if is_new:
        return 0
    if not supports_multiple_bond(symbol1) or not supports_multiple_bond(symbol2):
        return 0

    headroom1 = get_max_extra_valence(symbol1, degree1)
    headroom2 = get_max_extra_valence(symbol2, degree2)
    if headroom1 <= 0 or headroom2 <= 0:
        return 0

    max_cap = min(headroom1, headroom2)
    if (
        symbol1 in TRIPLE_BOND_ELEMENTS
        and symbol2 in TRIPLE_BOND_ELEMENTS
        and degree1 <= 2
        and degree2 <= 2
    ):
        return min(2, max_cap)

    return min(1, max_cap)


def get_bond_extra_cost(symbol1: str, symbol2: str, distance: float) -> float:
    num1 = atomic_numbers.get(symbol1)
    num2 = atomic_numbers.get(symbol2)
    if num1 is None or num2 is None:
        return 0.1

    ref = covalent_radii[num1] + covalent_radii[num2]
    if ref <= 0:
        return 0.1

    ratio = float(distance) / float(ref)
    return max(0.02, 0.10 * ratio)


def build_bond_records(atoms, matrix, new_links):
    upper_triangle = sparse.triu(matrix, k=1)
    rows, cols = upper_triangle.nonzero()

    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)
    vecs_raw = atoms.positions[cols] - atoms.positions[rows]
    vecs_mic, dists = conditional_find_mic(vecs_raw, cell, atoms.get_pbc())
    offsets = np.round((vecs_mic - vecs_raw) @ inv_cell).astype(int)

    bond_records = []
    for idx1, idx2, dist, offset in zip(
        rows.tolist(), cols.tolist(), np.asarray(dists).tolist(), offsets.tolist()
    ):
        bond_records.append(
            {
                "idx1": int(idx1),
                "idx2": int(idx2),
                "dist": float(dist),
                "offset": tuple(int(x) for x in offset),
                "is_new": False,
                "bond_type": "S",
            }
        )

    for idx1, idx2, offset in new_links:
        idx1 = int(idx1)
        idx2 = int(idx2)
        offset = np.asarray(offset, dtype=int)
        vec = atoms.positions[idx2] - atoms.positions[idx1] + offset @ cell
        dist = float(np.linalg.norm(vec))
        bond_records.append(
            {
                "idx1": idx1,
                "idx2": idx2,
                "dist": dist,
                "offset": tuple(int(x) for x in offset.tolist()),
                "is_new": True,
                "bond_type": "S",
            }
        )

    return bond_records


def compute_global_degrees(n_atoms: int, bond_records):
    degrees = np.zeros(n_atoms, dtype=int)
    for bond in bond_records:
        degrees[bond["idx1"]] += 1
        degrees[bond["idx2"]] += 1
    return degrees


def get_variable_bond_components(atoms, bond_records, global_degrees):
    adjacency = {}
    variable_atoms = set()
    variable_bond_indices = []

    for bond_idx, bond in enumerate(bond_records):
        idx1 = bond["idx1"]
        idx2 = bond["idx2"]
        cap = estimate_extra_bond_capacity(
            atoms[idx1].symbol,
            atoms[idx2].symbol,
            int(global_degrees[idx1]),
            int(global_degrees[idx2]),
            bond["is_new"],
        )
        if cap <= 0:
            continue

        variable_bond_indices.append(bond_idx)
        variable_atoms.add(idx1)
        variable_atoms.add(idx2)
        adjacency.setdefault(idx1, []).append((idx2, bond_idx))
        adjacency.setdefault(idx2, []).append((idx1, bond_idx))

    component_atom_lists = []
    component_bond_lists = []
    visited = set()

    for start_atom in sorted(variable_atoms):
        if start_atom in visited:
            continue

        queue = deque([start_atom])
        visited.add(start_atom)
        comp_atoms = []
        comp_bonds = set()

        while queue:
            atom_idx = queue.popleft()
            comp_atoms.append(atom_idx)
            for neighbor_idx, bond_idx in adjacency.get(atom_idx, []):
                comp_bonds.add(bond_idx)
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

        component_atom_lists.append(sorted(comp_atoms))
        component_bond_lists.append(sorted(comp_bonds))

    return component_atom_lists, component_bond_lists, set(variable_bond_indices)


def solve_component_bond_orders(
    atoms, component_atom_indices, component_bond_indices, bond_records, global_degrees
):
    atom_to_local = {
        atom_idx: local_idx for local_idx, atom_idx in enumerate(component_atom_indices)
    }
    symbols = [atoms[atom_idx].symbol for atom_idx in component_atom_indices]
    degrees = [int(global_degrees[atom_idx]) for atom_idx in component_atom_indices]

    variable_bonds = []
    fixed_orders = {}
    for bond_idx in component_bond_indices:
        bond = bond_records[bond_idx]
        i_local = atom_to_local[bond["idx1"]]
        j_local = atom_to_local[bond["idx2"]]
        cap = estimate_extra_bond_capacity(
            symbols[i_local],
            symbols[j_local],
            degrees[i_local],
            degrees[j_local],
            bond["is_new"],
        )
        if cap <= 0:
            fixed_orders[bond_idx] = 1
        else:
            variable_bonds.append((bond_idx, i_local, j_local, cap))

    candidate_valences = [
        get_candidate_valences(sym, deg) for sym, deg in zip(symbols, degrees)
    ]

    n_y = len(variable_bonds)
    n_z = sum(len(opts) for opts in candidate_valences)
    n_var = n_y + n_z

    if n_var == 0:
        return {bond_idx: 1 for bond_idx in component_bond_indices}

    c = np.zeros(n_var, dtype=float)
    lower = np.zeros(n_var, dtype=float)
    upper = np.ones(n_var, dtype=float)
    integrality = np.ones(n_var, dtype=int)

    for y_idx, (bond_idx, _i_local, _j_local, cap) in enumerate(variable_bonds):
        bond = bond_records[bond_idx]
        sym1 = atoms[bond["idx1"]].symbol
        sym2 = atoms[bond["idx2"]].symbol
        c[y_idx] = get_bond_extra_cost(sym1, sym2, bond["dist"])
        upper[y_idx] = cap

    z_offset = n_y
    atom_choice_var_indices = []
    for opts in candidate_valences:
        idxs = list(range(z_offset, z_offset + len(opts)))
        atom_choice_var_indices.append(idxs)
        for local_k, (_valence, penalty) in enumerate(opts):
            c[idxs[local_k]] = float(penalty)
        z_offset += len(opts)

    row_idx = []
    col_idx = []
    values = []
    lb = []
    ub = []
    current_row = 0

    for choice_vars in atom_choice_var_indices:
        for var_idx in choice_vars:
            row_idx.append(current_row)
            col_idx.append(var_idx)
            values.append(1.0)
        lb.append(1.0)
        ub.append(1.0)
        current_row += 1

    for atom_local, choice_vars in enumerate(atom_choice_var_indices):
        for y_idx, (_bond_idx, i_local, j_local, _cap) in enumerate(variable_bonds):
            if atom_local == i_local or atom_local == j_local:
                row_idx.append(current_row)
                col_idx.append(y_idx)
                values.append(1.0)

        for var_idx, (valence, _penalty) in zip(
            choice_vars, candidate_valences[atom_local]
        ):
            row_idx.append(current_row)
            col_idx.append(var_idx)
            values.append(-float(valence))

        lb.append(-float(degrees[atom_local]))
        ub.append(-float(degrees[atom_local]))
        current_row += 1

    A = sparse.csc_matrix((values, (row_idx, col_idx)), shape=(current_row, n_var))
    constraints = LinearConstraint(
        A, np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    )
    bounds = Bounds(lower, upper)

    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

    if not result.success or result.x is None:
        return {bond_idx: 1 for bond_idx in component_bond_indices}

    x = np.rint(result.x).astype(int)
    bond_orders = dict(fixed_orders)
    for y_idx, (bond_idx, _i_local, _j_local, _cap) in enumerate(variable_bonds):
        bond_orders[bond_idx] = int(1 + x[y_idx])

    for bond_idx in component_bond_indices:
        bond_orders.setdefault(bond_idx, 1)

    return bond_orders


def aromaticize_component_bonds(
    atoms, component_atom_indices, component_bond_indices, bond_records, bond_orders
):
    if not HAS_RDKIT:
        return {}

    atom_to_local = {
        atom_idx: local_idx for local_idx, atom_idx in enumerate(component_atom_indices)
    }
    rw_mol = Chem.RWMol()
    for atom_idx in component_atom_indices:
        rw_mol.AddAtom(Chem.Atom(atoms[atom_idx].symbol))

    fixed_new_pairs = set()
    for bond_idx in component_bond_indices:
        bond = bond_records[bond_idx]
        pair = tuple(sorted((bond["idx1"], bond["idx2"])))
        if bond["is_new"]:
            fixed_new_pairs.add(pair)

        order = max(1, min(3, int(bond_orders.get(bond_idx, 1))))
        rd_bond_type = RD_BOND_TYPE_FROM_ORDER.get(order)
        if rd_bond_type is None:
            return {}
        rw_mol.AddBond(
            atom_to_local[bond["idx1"]], atom_to_local[bond["idx2"]], rd_bond_type
        )

    mol = rw_mol.GetMol()
    try:
        mol.UpdatePropertyCache(strict=False)
        rdmolops.FastFindRings(mol)
        rdmolops.SetConjugation(mol)
        rdmolops.SetHybridization(mol)
        rdmolops.SetAromaticity(mol)
    except Exception:
        return {}

    aromatic_pairs = {}
    for rd_bond in mol.GetBonds():
        if not rd_bond.GetIsAromatic():
            continue
        global_i = component_atom_indices[rd_bond.GetBeginAtomIdx()]
        global_j = component_atom_indices[rd_bond.GetEndAtomIdx()]
        pair = tuple(sorted((global_i, global_j)))
        if pair in fixed_new_pairs:
            continue
        aromatic_pairs[pair] = "A"

    return aromatic_pairs


def assign_bond_types(atoms, bond_records):
    if not bond_records:
        return bond_records

    global_degrees = compute_global_degrees(len(atoms), bond_records)
    component_atom_lists, component_bond_lists, variable_bond_indices = (
        get_variable_bond_components(
            atoms,
            bond_records,
            global_degrees,
        )
    )

    global_orders = {bond_idx: 1 for bond_idx in range(len(bond_records))}
    aromatic_pairs = {}

    for comp_atoms, comp_bonds in zip(component_atom_lists, component_bond_lists):
        if not comp_bonds:
            continue

        comp_orders = solve_component_bond_orders(
            atoms,
            comp_atoms,
            comp_bonds,
            bond_records,
            global_degrees,
        )
        global_orders.update(comp_orders)

        comp_aromatic = aromaticize_component_bonds(
            atoms, comp_atoms, comp_bonds, bond_records, comp_orders
        )
        aromatic_pairs.update(comp_aromatic)

    for bond_idx, bond in enumerate(bond_records):
        if bond["is_new"]:
            bond["bond_type"] = "S"
            continue

        if bond_idx not in variable_bond_indices:
            bond["bond_type"] = "S"
            continue

        pair = tuple(sorted((bond["idx1"], bond["idx2"])))
        if pair in aromatic_pairs:
            bond["bond_type"] = "A"
            continue

        order = int(global_orders.get(bond_idx, 1))
        bond["bond_type"] = CCDC_TYPE_FROM_ORDER.get(order, "S")

    return bond_records


def generate_bonds(atoms, cutoff_multi=1.05):
    cutoffs = natural_cutoffs(atoms, mult=cutoff_multi)
    nl = build_neighbor_list(
        atoms, cutoffs=cutoffs, self_interaction=False, bothways=True
    )

    tmp = find_and_link_sites(atoms, nl)
    if tmp is None:
        return None

    matrix, new_links = tmp
    bond_records = build_bond_records(atoms, matrix, new_links)
    bond_records = assign_bond_types(atoms, bond_records)
    return bond_records


def expand_cif_tags_to_labels(atoms: ase.Atoms):
    original_tag_key = "_atom_site_label"
    new_tag_key = "label"

    if original_tag_key in atoms.info:
        original_tags = atoms.info[original_tag_key]
    elif original_tag_key in atoms.arrays:
        original_tags = atoms.arrays[original_tag_key]
    else:
        print(
            f"Warning: '{original_tag_key}' is absent from atoms.info and atoms.arrays; "
            "labels cannot be expanded."
        )
        return None

    if not isinstance(original_tags, (list, tuple, np.ndarray)):
        print(
            f"Warning: '{original_tag_key}' is not list-like; skipping label expansion."
        )
        return None

    kinds = atoms.arrays.get("spacegroup_kinds")

    try:
        if kinds is None:
            expanded_tags = list(original_tags)
            print(
                "Note: 'spacegroup_kinds' was not found. The crystal may not have "
                "been generated by symmetry, or the metadata may be absent."
            )
        else:
            total_counts = Counter(kinds)
            current_counts = {}
            expanded_tags = []
            for i in kinds:
                tag_str = str(original_tags[i])
                if total_counts[i] > 1:
                    current_counts[tag_str] = current_counts.get(tag_str, 0) + 1
                    expanded_tags.append(f"{tag_str}_{current_counts[tag_str]}")
                else:
                    expanded_tags.append(tag_str)

        atoms.set_array(new_tag_key, np.array(expanded_tags))

    except (ValueError, IndexError) as e:
        print(f"Error: labels could not be expanded or normalized. Details: {e}")
        return None

    return atoms


def append_bonds_to_cif(name, path, new_path, cutoff_multi=1.05, scale_factor=1.0):
    cif_path = os.path.join(path, name)
    new_cif_path = os.path.join(new_path, name)
    atoms = read(cif_path, store_tags=True)
    atoms = expand_cif_tags_to_labels(atoms)
    if atoms is None:
        print(f"Failed to expand CIF tags for {name}. Skipping bond generation.")
        return

    bonds = generate_bonds(atoms, cutoff_multi)
    if bonds is None:
        print(f"Failed to generate bonds for {name}.")
        return

    if "label" in atoms.arrays:
        labels = atoms.arrays["label"]
    else:
        print(
            f"Warning: labels could not be recovered for {name}; using element-index labels."
        )
        labels = np.array([f"{atom.symbol}{atom.index + 1}" for atom in atoms])

    atoms.set_cell(atoms.cell * scale_factor, scale_atoms=True)
    write(new_cif_path, atoms, format="cif", labels=labels.reshape(1, -1))
    with open(new_cif_path, "a", encoding="utf-8") as f:
        f.write("\n# --- Bond Information (Appended automatically) ---\n")
        f.write("loop_\n")
        f.write("  _geom_bond_atom_site_label_1\n")
        f.write("  _geom_bond_atom_site_label_2\n")
        f.write("  _geom_bond_distance\n")
        f.write("  _geom_bond_site_symmetry_2\n")
        f.write("  _ccdc_geom_bond_type\n")

        for bond in bonds:
            idx1 = bond["idx1"]
            idx2 = bond["idx2"]
            dist = bond["dist"] * scale_factor
            offset = bond["offset"]
            bond_type = bond["bond_type"]

            if offset == (0, 0, 0):
                sym_code = "."
            else:
                sym_x = 5 + offset[0]
                sym_y = 5 + offset[1]
                sym_z = 5 + offset[2]
                sym_code = f"1_{sym_x}{sym_y}{sym_z}"

            f.write(
                f"  {labels[idx1]:<8} {labels[idx2]:<8} {dist:.8f} {sym_code:>8} {bond_type}\n"
            )


def deal_single_cif_file(cif, path, new_path, cutoff_multi=1.05, scale_factor=1.0):
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        try:
            append_bonds_to_cif(
                cif, path, new_path, cutoff_multi, scale_factor=scale_factor
            )
        except Exception:
            pass


def link_structures(
    path, new_path, cutoff_multi=1.05, max_workers=None, scale_factor=1.0
):
    os.makedirs(new_path, exist_ok=True)
    cif_list = [f for f in os.listdir(path) if f.endswith(".cif")]
    done_list = [f for f in os.listdir(new_path) if f.endswith(".cif")]
    cif_list = [f for f in cif_list if f not in done_list]
    if not cif_list:
        print(
            f"No CIF files found in {path}. Please check the directory and ensure it contains CIF files."
        )
        return -1
    random.shuffle(cif_list)

    worker_func = partial(
        deal_single_cif_file,
        path=path,
        new_path=new_path,
        cutoff_multi=cutoff_multi,
        scale_factor=scale_factor,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(worker_func, cif_list),
                total=len(cif_list),
                desc="Linking structures",
                unit="structure",
            )
        )

    print("All files processed successfully.")
    return 0
