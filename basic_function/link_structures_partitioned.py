"""Link periodic fragments using typed atom-label annotations.

Use ``a_x_site_bo2`` / ``a_y_site_bo2`` for a type-a double bond. Orders
1, 2 and 3 are supported; omitting ``_boN`` means a single bond. Legacy
``x_site`` / ``y_site`` labels form an unnamed type. An atom can carry several
markers, e.g. ``a_x_site_bo2__b_y_site_bo1`` (no spaces). ``__`` is reserved
as the marker separator; type names use letters, digits and single underscores.
Automatic symmetry, supercell and numeric copy suffixes apply to the whole
label. A hyphen before x/y is also accepted, matching upstream normalization.

Equivalent members of a type use positive IDs, e.g. ``a_x_site_id1_bo2`` and
``a_x_site_id2_bo2``. IDs distinguish members; they do not prescribe x/y pairs.
Every participating fragment must have the same types and member counts, with
equal numbers of x and y per type. Multi-member groups require explicit IDs,
unique within each direction. Different types may share atoms, as may x and y.
Orders must agree within each type.
All types share one target fragment image. Duplicate physical bonds are errors.
Declared orders are fixed during integer-order inference, but may subsequently
be written as aromatic (A) after RDKit perception in an open three-period graph.
"""

import concurrent.futures
import heapq
import os
import random
import re
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

SITE_MARKER_RE = re.compile(
    r"(?:(?P<kind>[A-Za-z][A-Za-z0-9_]*?)[_-])?"
    r"(?P<role>[xy])_site(?:_id(?P<member>[0-9]+))?(?:_bo(?P<order>[123]))?"
)
# Negative supercell indices become double underscores during normalization.
SITE_COPY_SUFFIX_RE = re.compile(
    r"(?:_s\d+|_sc_[-_]?\d+_[-_]?\d+_[-_]?\d+|_\d+)+$"
)


def parse_site_label(label):
    """Return all (type, role, order) markers, ignoring ordinary atom labels."""
    return [(kind, role, order) for kind, role, order, _ in _parse_site_markers(label)]


def _parse_site_markers(label):
    """Detailed parser; member IDs are separate from automatic copy suffixes."""
    if not isinstance(label, str) or "_site" not in label:
        return []
    base = SITE_COPY_SUFFIX_RE.sub("", label)
    markers = []
    seen = set()
    for token in base.split("__"):
        match = SITE_MARKER_RE.fullmatch(token)
        if match is None:
            raise ValueError(
                f"Invalid connection label {label!r}: expected "
                "[type_]x_site[_idN][_bo1/2/3] or [type_]y_site[_idN][_bo1/2/3], "
                "separated by '__'."
            )
        kind = match.group("kind") or ""
        role = match.group("role")
        if (kind, role) in seen:
            raise ValueError(f"Repeated {kind or 'default'}-{role} marker in {label!r}.")
        seen.add((kind, role))
        member = match.group("member")
        member = int(member) if member is not None else None
        if member == 0:
            raise ValueError(f"Member ID must be a positive integer in {label!r}.")
        markers.append((kind, role, int(match.group("order") or 1), member))
    return markers


def collect_site_groups(labels, context="Fragment"):
    """Validate one fragment's {atom_index: label} and return typed site lists.

    Each group contains ``x`` and ``y`` tuples of atom indices, ordered by
    member ID (then atom index), and a shared integer ``order``. The tuples
    describe interchangeable members, not fixed pairs at corresponding slots.
    """
    groups = {}
    for idx, label in sorted(labels.items()):
        for kind, role, order, member in _parse_site_markers(label):
            group = groups.setdefault(kind, {"x": [], "y": [], "order": order})
            if order != group["order"]:
                raise ValueError(
                    f"{context}: conflicting bond orders for type {kind or 'default'!r} "
                    f"at atom {idx} ({label!r}): {group['order']} and {order}."
                )
            group[role].append((member, idx))

    for kind, group in groups.items():
        nx, ny = len(group["x"]), len(group["y"])
        prefix = f"{context}, type {kind or 'default'!r}"
        if not nx or nx != ny:
            raise ValueError(f"{prefix} needs equal nonzero x/y counts; got {nx} x and {ny} y.")
        for role in ("x", "y"):
            members = group[role]
            if len(members) > 1 and any(member is None for member, _ in members):
                raise ValueError(
                    f"{prefix} has multiple {role} sites; every member needs an explicit "
                    f"_idN: {[(idx, labels[idx]) for _, idx in members]!r}."
                )
            seen = {}
            for member, idx in members:
                if member in seen:
                    previous = seen[member]
                    raise ValueError(
                        f"{prefix} has duplicate {role} member ID {member}: "
                        f"atoms {previous} ({labels[previous]!r}) and {idx} ({labels[idx]!r})."
                    )
                seen[member] = idx
            group[role] = tuple(idx for _, idx in sorted(
                members, key=lambda item: (item[0] or 0, item[1])
            ))
    return {kind: groups[kind] for kind in sorted(groups)}


def match_equivalent_sites(x_positions, y_positions, translation=0):
    """Return target columns and matched squared distances in x-member order.

    Both position arrays have the same (n, 3) shape. A single translation is
    applied to the entire target fragment; do not take per-pair minimum images.
    """
    vectors = y_positions[np.newaxis, :, :] - x_positions[:, np.newaxis, :] + translation
    squared = np.sum(vectors**2, axis=-1)
    if len(x_positions) == 1:
        return np.array([0], dtype=int), squared[0]
    rows, columns = linear_sum_assignment(squared)
    return columns, squared[rows, columns]


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

    matrix = nl.get_connectivity_matrix(sparse=True).tocsr()
    _, component_list = sparse.csgraph.connected_components(matrix)
    fragment_labels = {}
    for idx, label in enumerate(labels):
        fragment_labels.setdefault(int(component_list[idx]), {})[idx] = label
    fragments = {}
    for component, members in fragment_labels.items():
        groups = collect_site_groups(members, context=f"Fragment {component}")
        if groups:
            fragments[component] = groups

    if not fragments:
        return None

    component_ids = sorted(fragments)
    reference = fragments[component_ids[0]]
    kinds = list(reference)
    orders = {kind: reference[kind]["order"] for kind in kinds}
    group_slices = {}
    total_sites = 0
    for kind in kinds:
        count = len(reference[kind]["x"])
        group_slices[kind] = slice(total_sites, total_sites + count)
        total_sites += count
    x_indices = []
    y_indices = []
    for component in component_ids:
        sites = fragments[component]
        if set(sites) != set(kinds):
            raise ValueError(
                f"Fragment {component} has connection types {sorted(sites)!r}; "
                f"expected {kinds!r}."
            )
        for kind in kinds:
            if len(sites[kind]["x"]) != len(reference[kind]["x"]):
                raise ValueError(
                    f"Fragment {component}, type {kind or 'default'!r} "
                    f"has {len(sites[kind]['x'])} members per direction; "
                    f"expected {len(reference[kind]['x'])}."
                )
            if sites[kind]["order"] != orders[kind]:
                raise ValueError(
                    f"Fragment {component}, type {kind or 'default'!r} has conflicting "
                    f"bond order {sites[kind]['order']}; expected {orders[kind]}."
                )
        x_indices.append([idx for kind in kinds for idx in sites[kind]["x"]])
        y_indices.append([idx for kind in kinds for idx in sites[kind]["y"]])

    x_indices = np.asarray(x_indices, dtype=int)
    y_indices = np.asarray(y_indices, dtype=int)
    eff_components = len(component_ids)

    pbc = atoms.get_pbc()
    if not np.any(pbc):
        return None  # Every assignment would form finite molecular cycles.
    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)

    # Unwrap every fragment relative to its first y site. Shared atoms have
    # one offset even when they carry several different connection markers.
    x_offsets = np.zeros((*x_indices.shape, 3), dtype=int)
    y_offsets = np.zeros_like(x_offsets)
    for i in range(eff_components):
        anchor = int(y_indices[i, 0])
        offsets = {
            int(idx): get_total_cell_offset(nl, anchor, int(idx))
            for idx in set(x_indices[i]) | set(y_indices[i])
        }
        x_offsets[i] = [offsets[int(idx)] for idx in x_indices[i]]
        y_offsets[i] = [offsets[int(idx)] for idx in y_indices[i]]
    x_positions = atoms.positions[x_indices] + x_offsets @ cell
    y_positions = atoms.positions[y_indices] + y_offsets @ cell
    # Average all members, not group centers: every proposed bond has equal weight.
    x_centers = x_positions.mean(axis=1)
    y_centers = y_positions.mean(axis=1)

    # Sum_t |d_t + shift|^2 = count * |mean(d_t) + shift|^2 + constant.
    # Thus the centroid MIC minimizes all bond lengths for ONE fragment image.
    # Permuting targets within a group leaves mean(d_t) unchanged, so the
    # optimal fragment translation is independent of the member matching.
    center_vecs, _ = get_distances(x_centers, y_centers, cell=cell, pbc=pbc)
    center_raw = y_centers[np.newaxis, :, :] - x_centers[:, np.newaxis, :]
    fragment_offsets = np.rint((center_vecs - center_raw) @ inv_cell).astype(int)
    centers = ase.Atoms(
        positions=np.concatenate([x_centers, y_centers]), cell=cell, pbc=pbc
    )
    for i in range(eff_components):
        # Avoid linking a fragment back onto itself in the same image. In a
        # distance tie the second image can still be the zero-offset image.
        k = 2
        while np.all(fragment_offsets[i, i] == 0):
            _, vector = kth_periodic_distance(centers, i, eff_components + i, k=k)
            fragment_offsets[i, i] = np.rint(
                (vector - center_raw[i, i]) @ inv_cell
            ).astype(int)
            k += 1

    x2y_dist_sqr = np.zeros((eff_components, eff_components), dtype=float)
    translations = fragment_offsets @ cell
    pairings = np.empty((eff_components, eff_components, total_sites), dtype=int)
    for kind in kinds:
        span = group_slices[kind]
        if span.stop - span.start == 1:
            t = span.start
            vectors = (
                y_positions[np.newaxis, :, t, :]
                - x_positions[:, np.newaxis, t, :]
                + translations
            )
            x2y_dist_sqr += np.sum(vectors**2, axis=-1)
            pairings[:, :, t] = t
            continue
        for i in range(eff_components):
            for j in range(eff_components):
                columns, squared = match_equivalent_sites(
                    x_positions[i, span], y_positions[j, span], translations[i, j]
                )
                x2y_dist_sqr[i, j] += squared.sum()
                pairings[i, j, span] = span.start + columns

    # These are reference offsets for fragment winding, not mandatory site pairs.
    y2x_offsets = x_offsets[:, 0, :]
    x2y_offsets = fragment_offsets - y2x_offsets[:, np.newaxis, :]
    cost, assignment = solve_minimal_path(x2y_dist_sqr, y2x_offsets, x2y_offsets)

    if cost is None or assignment is None:
        return None

    new_links = []
    for i, j in enumerate(assignment):
        for kind in kinds:
            span = group_slices[kind]
            for t in range(span.start, span.stop):
                target = pairings[i, j, t]
                offset = fragment_offsets[i, j] + y_offsets[j, target] - x_offsets[i, t]
                new_links.append(
                    {
                        "idx1": int(x_indices[i, t]),
                        "idx2": int(y_indices[j, target]),
                        "offset": tuple(int(value) for value in offset),
                        "site_type": kind,
                        "fixed_order": orders[kind],
                        "source_component": component_ids[i],
                        "target_component": component_ids[j],
                    }
                )

    validate_unique_bonds(atoms, new_links)
    return matrix, new_links


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
    symbol1: str,
    symbol2: str,
    degree1: int,
    degree2: int,
    is_new: bool,
    base_valence1=None,
    base_valence2=None,
) -> int:
    if is_new:
        return 0
    if not supports_multiple_bond(symbol1) or not supports_multiple_bond(symbol2):
        return 0

    headroom1 = get_max_extra_valence(
        symbol1, degree1 if base_valence1 is None else base_valence1
    )
    headroom2 = get_max_extra_valence(
        symbol2, degree2 if base_valence2 is None else base_valence2
    )
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


def validate_unique_bonds(atoms, bonds):
    """Reject the same physical bond, including reverse periodic duplicates."""
    seen = {}
    labels = atoms.arrays.get("label")
    for bond in bonds:
        idx1, idx2 = bond["idx1"], bond["idx2"]
        offset = tuple(int(value) for value in bond["offset"])
        forward = (idx1, idx2, *offset)
        reverse = (idx2, idx1, *(-value for value in offset))
        key = min(forward, reverse)
        if idx1 == idx2 and not any(offset):
            raise ValueError(f"Connection at atom {idx1} points to the very same atom.")
        if key in seen:
            previous = seen[key]
            names = f"{labels[idx1]!r}, {labels[idx2]!r}" if labels is not None else ""
            raise ValueError(
                f"Duplicate physical bond between atoms {idx1}, {idx2} ({names}), "
                f"offset {offset}: types {previous.get('site_type', 'internal')!r} "
                f"and {bond.get('site_type', 'internal')!r}. "
                "Use one marker with bo2/bo3 to specify a multiple bond."
            )
        seen[key] = bond


def build_bond_records(atoms, matrix, new_links):
    upper_triangle = sparse.triu(matrix, k=1)
    rows, cols = upper_triangle.nonzero()

    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)
    vecs_raw = atoms.positions[cols] - atoms.positions[rows]
    if len(rows):
        vecs_mic, dists = conditional_find_mic(vecs_raw, cell, atoms.get_pbc())
        offsets = np.round((vecs_mic - vecs_raw) @ inv_cell).astype(int)
    else:
        dists = np.empty(0)
        offsets = np.empty((0, 3), dtype=int)

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

    for link in new_links:
        idx1 = link["idx1"]
        idx2 = link["idx2"]
        offset = np.asarray(link["offset"], dtype=int)
        vec = atoms.positions[idx2] - atoms.positions[idx1] + offset @ cell
        dist = float(np.linalg.norm(vec))
        bond_records.append(
            {
                **link,
                "dist": dist,
                "offset": tuple(int(x) for x in offset.tolist()),
                "is_new": True,
                "bond_type": CCDC_TYPE_FROM_ORDER[link["fixed_order"]],
            }
        )

    validate_unique_bonds(atoms, bond_records)
    return bond_records


def compute_global_degrees(n_atoms: int, bond_records):
    degrees = np.zeros(n_atoms, dtype=int)
    for bond in bond_records:
        degrees[bond["idx1"]] += 1
        degrees[bond["idx2"]] += 1
    return degrees


def compute_base_valences(n_atoms: int, bond_records):
    """Count fixed new orders and one unit per internal bond before inference."""
    valences = np.zeros(n_atoms, dtype=int)
    for bond in bond_records:
        order = bond.get("fixed_order", 1) if bond["is_new"] else 1
        # A bond to another image of the same atom contributes at BOTH ends.
        valences[bond["idx1"]] += order
        valences[bond["idx2"]] += order
    return valences


def get_variable_bond_components(
    atoms, bond_records, global_degrees, base_valences=None
):
    if base_valences is None:
        base_valences = compute_base_valences(len(atoms), bond_records)
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
            int(base_valences[idx1]),
            int(base_valences[idx2]),
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
    atoms,
    component_atom_indices,
    component_bond_indices,
    bond_records,
    global_degrees,
    base_valences=None,
):
    if base_valences is None:
        base_valences = compute_base_valences(len(atoms), bond_records)
    atom_to_local = {
        atom_idx: local_idx for local_idx, atom_idx in enumerate(component_atom_indices)
    }
    symbols = [atoms[atom_idx].symbol for atom_idx in component_atom_indices]
    degrees = [int(global_degrees[atom_idx]) for atom_idx in component_atom_indices]
    base = [int(base_valences[atom_idx]) for atom_idx in component_atom_indices]
    default_orders = {
        bond_idx: (
            bond_records[bond_idx].get("fixed_order", 1)
            if bond_records[bond_idx]["is_new"] else 1
        )
        for bond_idx in component_bond_indices
    }

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
            base[i_local],
            base[j_local],
        )
        if cap <= 0:
            fixed_orders[bond_idx] = default_orders[bond_idx]
        else:
            variable_bonds.append((bond_idx, i_local, j_local, cap))

    candidate_valences = [
        get_candidate_valences(sym, valence) for sym, valence in zip(symbols, base)
    ]

    n_y = len(variable_bonds)
    n_z = sum(len(opts) for opts in candidate_valences)
    n_var = n_y + n_z

    if n_var == 0:
        return default_orders

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

        lb.append(-float(base[atom_local]))
        ub.append(-float(base[atom_local]))
        current_row += 1

    A = sparse.csc_matrix((values, (row_idx, col_idx)), shape=(current_row, n_var))
    constraints = LinearConstraint(
        A, np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    )
    bounds = Bounds(lower, upper)

    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

    if not result.success or result.x is None:
        return default_orders

    x = np.rint(result.x).astype(int)
    bond_orders = dict(fixed_orders)
    for y_idx, (bond_idx, _i_local, _j_local, _cap) in enumerate(variable_bonds):
        bond_orders[bond_idx] = int(1 + x[y_idx])

    for bond_idx in component_bond_indices:
        bond_orders.setdefault(bond_idx, default_orders[bond_idx])

    return bond_orders


def aromaticize_periodic_component(
    atoms, atom_offsets, component_bond_indices, bond_records, bond_orders
):
    """Perceive an open three-period lift; return original bond-record indices.

    Tree offsets unwrap one representative of each atom. Nonzero residuals
    around cycles generate the translation of the fragment assignment cycle.
    A finite component needs one copy; a 1D component gets copies -1, 0, 1.
    This is independent of which atoms happen to be wrapped into the CIF cell.
    """
    residuals = []
    for bond_idx in component_bond_indices:
        bond = bond_records[bond_idx]
        residuals.append(
            atom_offsets[bond["idx1"]]
            + np.asarray(bond["offset"], dtype=int)
            - atom_offsets[bond["idx2"]]
        )
    nonzero = [value for value in residuals if np.any(value)]
    period = np.zeros(3, dtype=int)
    layers = (0,)
    if nonzero:
        direction = nonzero[0] // np.gcd.reduce(np.abs(nonzero[0]))
        pivot = int(np.flatnonzero(direction)[0])
        multiples = []
        for value in nonzero:
            multiple, remainder = divmod(int(value[pivot]), int(direction[pivot]))
            if remainder or not np.array_equal(value, multiple * direction):
                # Higher-dimensional networks are outside the 1D linker model.
                return {}
            multiples.append(abs(multiple))
        period = direction * np.gcd.reduce(multiples)
        layers = (-1, 0, 1)

    try:
        rw_mol = Chem.RWMol()
        node_indices = {}
        for layer in layers:
            for atom_idx, offset in atom_offsets.items():
                key = (atom_idx, tuple(offset + layer * period))
                node_indices[key] = rw_mol.AddAtom(Chem.Atom(atoms[atom_idx].symbol))

        bond_mapping = []
        for layer in layers:
            for bond_idx in component_bond_indices:
                bond = bond_records[bond_idx]
                image = atom_offsets[bond["idx1"]] + layer * period
                source = (bond["idx1"], tuple(image))
                target = (bond["idx2"], tuple(image + bond["offset"]))
                if target not in node_indices:
                    continue  # Open boundaries: never reconnect the end copies.
                order = int(bond_orders[bond_idx])
                rw_mol.AddBond(
                    node_indices[source],
                    node_indices[target],
                    RD_BOND_TYPE_FROM_ORDER[order],
                )
                bond_mapping.append((bond_idx, layer == 0))

        mol = rw_mol.GetMol()
        mol.UpdatePropertyCache(strict=False)
        rdmolops.FastFindRings(mol)
        rdmolops.SetConjugation(mol)
        rdmolops.SetHybridization(mol)
        rdmolops.SetAromaticity(mol)
        return {
            bond_idx: "A"
            for rd_bond, (bond_idx, is_middle) in zip(mol.GetBonds(), bond_mapping)
            if is_middle and rd_bond.GetIsAromatic()
        }
    except Exception:
        return {}


def aromaticize_bonds(atoms, bond_records, bond_orders):
    """Include fixed/new bonds and all neighbors in each perception component."""
    if not HAS_RDKIT:
        return {}
    adjacency = {}
    for bond_idx, bond in enumerate(bond_records):
        idx1, idx2 = bond["idx1"], bond["idx2"]
        offset = np.asarray(bond["offset"], dtype=int)
        adjacency.setdefault(idx1, []).append((idx2, offset, bond_idx))
        adjacency.setdefault(idx2, []).append((idx1, -offset, bond_idx))

    visited = set()
    aromatic = {}
    for root in sorted(adjacency):
        if root in visited:
            continue
        offsets = {root: np.zeros(3, dtype=int)}
        queue = deque([root])
        visited.add(root)
        component_bonds = set()
        while queue:
            idx = queue.popleft()
            for neighbor, offset, bond_idx in adjacency[idx]:
                component_bonds.add(bond_idx)
                if neighbor not in visited:
                    visited.add(neighbor)
                    offsets[neighbor] = offsets[idx] + offset
                    queue.append(neighbor)
        aromatic.update(
            aromaticize_periodic_component(
                atoms, offsets, sorted(component_bonds), bond_records, bond_orders
            )
        )
    return aromatic


def assign_bond_types(atoms, bond_records):
    if not bond_records:
        return bond_records

    global_degrees = compute_global_degrees(len(atoms), bond_records)
    base_valences = compute_base_valences(len(atoms), bond_records)
    component_atom_lists, component_bond_lists, _ = (
        get_variable_bond_components(
            atoms,
            bond_records,
            global_degrees,
            base_valences,
        )
    )

    global_orders = {
        bond_idx: bond.get("fixed_order", 1) if bond["is_new"] else 1
        for bond_idx, bond in enumerate(bond_records)
    }

    for comp_atoms, comp_bonds in zip(component_atom_lists, component_bond_lists):
        if not comp_bonds:
            continue

        comp_orders = solve_component_bond_orders(
            atoms,
            comp_atoms,
            comp_bonds,
            bond_records,
            global_degrees,
            base_valences,
        )
        global_orders.update(comp_orders)

    aromatic_bonds = aromaticize_bonds(atoms, bond_records, global_orders)
    for bond_idx, bond in enumerate(bond_records):
        bond["bond_type"] = aromatic_bonds.get(
            bond_idx, CCDC_TYPE_FROM_ORDER[global_orders[bond_idx]]
        )

    return bond_records


def generate_bonds(atoms, cutoff_multi=1.1):
    cutoffs = natural_cutoffs(atoms, mult=cutoff_multi)
    # The neighbor list defines bonds; its reuse buffer would add false bonds.
    nl = build_neighbor_list(
        atoms, cutoffs=cutoffs, skin=0.0, self_interaction=False, bothways=True
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


def append_bonds_to_cif(name, path, new_path, cutoff_multi=1.1, scale_factor=1.0):
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


def deal_single_cif_file(cif, path, new_path, cutoff_multi=1.1, scale_factor=1.0):
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        try:
            append_bonds_to_cif(
                cif, path, new_path, cutoff_multi, scale_factor=scale_factor
            )
        except Exception:
            pass


def link_structures(
    path, new_path, cutoff_multi=1.1, max_workers=None, scale_factor=1.0
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
