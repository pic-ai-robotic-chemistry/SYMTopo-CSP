"""Periodic-image helpers for running bonded LAMMPS jobs on linked structures.

LAMMPS stores a bond as two atom IDs and, at every reneighboring, resolves the
partner to its closest periodic image (domain->closest_image). Image flags in the
data file are not used for this. The same rule is applied from the owner atom to
every other atom of an angle, dihedral and improper. A linked CIF records the
intended image of each bond in _geom_bond_site_symmetry_2, but when that image is
not the closest one in a 1x1x1 cell, LAMMPS connects a different copy and the
relaxation breaks the chain or closes it into a ring.

This module finds the smallest a x b x c supercell in which every bond and every
two-bond walk is the closest image by a safety margin, writes that supercell as a
P1 CIF whose bonds point at explicit copies, and folds a relaxed supercell back
to the primitive cell.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from ase.geometry import cellpar_to_cell, minkowski_reduce
from ase.io.cif import parse_cif

Bond = Tuple[int, int, np.ndarray, float, str]

CELL_KEYS = (
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
)


@dataclass
class BondedCif:
    labels: List[str]
    symbols: List[str]
    frac: np.ndarray
    cellpar: np.ndarray
    bonds: List[Bond]

    @property
    def cell(self) -> np.ndarray:
        return cellpar_to_cell(self.cellpar)


def parse_symflag(code: str) -> np.ndarray:
    """'.' -> (0, 0, 0); '1_655' -> (1, 0, 0)."""
    code = str(code)
    if code in (".", "?"):
        return np.zeros(3, dtype=int)
    return np.array([int(c) - 5 for c in code.split("_")[1]], dtype=int)


def format_symflag(offset: Sequence[int]) -> str:
    if not np.any(offset):
        return "."
    if np.any(np.abs(offset) > 4):
        raise ValueError(f"Offset {tuple(offset)} cannot be written as a 1_xyz code.")
    return "1_" + "".join(str(5 + int(v)) for v in offset)


def read_bonded_cif(path: Path) -> BondedCif:
    """Read a P1 CIF and its _geom_bond loop as (i, j, offset, distance, type)."""
    block = next(iter(parse_cif(str(path))))

    def column(tag, default=None):
        value = block.get(tag)
        return list(value) if value is not None else default

    labels = [str(x) for x in column("_atom_site_label")]
    symbols = [str(x) for x in column("_atom_site_type_symbol")]
    frac = np.column_stack(
        [np.asarray(column(f"_atom_site_fract_{a}"), dtype=float) for a in "xyz"]
    )
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError(f"{path}: atom labels are not unique.")
    l1 = column("_geom_bond_atom_site_label_1", [])
    l2 = column("_geom_bond_atom_site_label_2", [])
    flags = column("_geom_bond_site_symmetry_2", ["."] * len(l1))
    dists = column("_geom_bond_distance", [0.0] * len(l1))
    types = column("_ccdc_geom_bond_type", ["S"] * len(l1))
    bonds = [
        (index[str(a)], index[str(b)], parse_symflag(f), float(d), str(t))
        for a, b, f, d, t in zip(l1, l2, flags, dists, types)
    ]
    return BondedCif(labels, symbols, frac, np.asarray(block.get_cellpar(), float), bonds)


def bond_walk_vectors(cif: BondedCif, depth: int = 2) -> np.ndarray:
    """Cartesian vectors of all walks of 1..depth bonds (no immediate backtracking).

    These are the separations LAMMPS resolves by closest image: bond partners and,
    from the owner atom, the remaining atoms of angles, dihedrals and impropers.
    """
    n = len(cif.labels)
    adj = [[] for _ in range(n)]
    for k, (i, j, t, *_rest) in enumerate(cif.bonds):
        adj[i].append((j, t, k))
        adj[j].append((i, -t, k))
    seen = set()
    rows = []
    for start in range(n):
        frontier = [(start, np.zeros(3, dtype=int), -1)]
        for _ in range(depth):
            nxt = []
            for node, off, last in frontier:
                for nb, t, k in adj[node]:
                    if k == last:
                        continue
                    o = off + t
                    key = (start, nb, *o.tolist())
                    if not (nb == start and not o.any()) and key not in seen:
                        seen.add(key)
                        rows.append((start, nb, *o.tolist()))
                    nxt.append((nb, o, k))
            frontier = nxt
    if not rows:
        return np.zeros((0, 3))
    rows = np.asarray(rows, dtype=int)
    vectors = (cif.frac[rows[:, 1]] - cif.frac[rows[:, 0]] + rows[:, 2:5]) @ cif.cell
    return np.unique(np.round(vectors, 6), axis=0)


def image_margin(vectors: np.ndarray, lattice: np.ndarray, threshold: float) -> float:
    """Smallest (nearest competing image distance - intended length) over all vectors.

    Competitors are v + L for nonzero lattice translations L. Only competitors
    within `threshold` of |v| matter, so |L| < 2|v| + threshold bounds the
    reduced-basis coefficients that have to be enumerated.
    """
    if len(vectors) == 0:
        return float("inf")
    own = np.linalg.norm(vectors, axis=1)
    rcell, _ = minkowski_reduce(lattice)
    vol = abs(np.linalg.det(rcell))
    heights = [vol / np.linalg.norm(np.cross(rcell[(i + 1) % 3], rcell[(i + 2) % 3]))
               for i in range(3)]
    k_max = int(np.ceil((2 * own.max() + threshold + 1.0) / min(heights)))
    ks = [k for k in itertools.product(range(-k_max, k_max + 1), repeat=3) if any(k)]
    shifts = np.asarray(ks) @ rcell
    best = np.empty(len(vectors))
    step = max(1, 4_000_000 // (3 * len(shifts)))
    for start in range(0, len(vectors), step):
        chunk = vectors[start : start + step]
        best[start : start + step] = np.linalg.norm(
            chunk[:, None, :] + shifts[None, :, :], axis=2
        ).min(axis=1)
    return float((best - own).min())


def choose_replication(
    cif: BondedCif, threshold: float = 1.0, max_rep: int = 4
) -> Tuple[Optional[Tuple[int, int, int]], float, float]:
    """Smallest a x b x c (fewest cells first) whose image margin reaches threshold.

    Returns (replication or None, margin at 1x1x1, margin at the chosen replication).
    """
    vectors = bond_walk_vectors(cif, depth=2)
    cell = cif.cell
    candidates = sorted(
        itertools.product(range(1, max_rep + 1), repeat=3),
        key=lambda m: (int(np.prod(m)), max(m), m),
    )
    margin_1x1x1 = None
    for rep in candidates:
        margin = image_margin(vectors, cell * np.asarray(rep)[:, None], threshold)
        if margin_1x1x1 is None:
            margin_1x1x1 = margin
        if margin >= threshold:
            return tuple(int(v) for v in rep), margin_1x1x1, margin
    return None, margin_1x1x1, float("nan")


def write_supercell_cif(
    cif: BondedCif, rep: Sequence[int], path: Path
) -> Tuple[List[List[int]], List[str]]:
    """Write cif replicated rep times.

    Returns [primitive index, na, nb, nc] per supercell atom and the supercell
    atom labels, both in supercell CIF order. Atom i of copy n is labelled
    '<label>_S<k>', k being the copy index. Bond (i, j, t) of copy n points to j
    in copy (n + t) mod rep with supercell offset floor((n + t) / rep).
    """
    M = np.asarray(rep, dtype=int)
    copies = [np.array(n) for n in np.ndindex(*M)]
    copy_id = {tuple(n): k for k, n in enumerate(copies)}
    n_prim = len(cif.labels)

    labels, symbols, frac, mapping = [], [], [], []
    for k, n in enumerate(copies):
        for i in range(n_prim):
            labels.append(f"{cif.labels[i]}_S{k}")
            symbols.append(cif.symbols[i])
            frac.append((cif.frac[i] + n) / M)
            mapping.append([i, *n.tolist()])
    bonds = []
    for k, n in enumerate(copies):
        for i, j, t, dist, btype in cif.bonds:
            target = n + t
            k2 = copy_id[tuple(np.mod(target, M))]
            bonds.append((k * n_prim + i, k2 * n_prim + j, np.floor_divide(target, M), dist, btype))
    frac = np.asarray(frac)
    cellpar = np.concatenate([cif.cellpar[:3] * M, cif.cellpar[3:]])

    lines = [f"data_{path.stem}", ""]
    lines += [f"{key:<22}{value:.10f}" for key, value in zip(CELL_KEYS, cellpar)]
    lines += [
        "",
        '_space_group_name_H-M_alt    "P 1"',
        "_space_group_IT_number       1",
        "",
        "loop_",
        "  _space_group_symop_operation_xyz",
        "  'x, y, z'",
        "",
        "loop_",
        "  _atom_site_type_symbol",
        "  _atom_site_label",
        "  _atom_site_symmetry_multiplicity",
        "  _atom_site_fract_x",
        "  _atom_site_fract_y",
        "  _atom_site_fract_z",
        "  _atom_site_occupancy",
    ]
    lines += [
        f"  {s:<3} {l:<24} 1.0  {f[0]:.12f}  {f[1]:.12f}  {f[2]:.12f}  1.0000"
        for s, l, f in zip(symbols, labels, frac)
    ]
    lines += [
        "",
        "loop_",
        "  _geom_bond_atom_site_label_1",
        "  _geom_bond_atom_site_label_2",
        "  _geom_bond_distance",
        "  _geom_bond_site_symmetry_2",
        "  _ccdc_geom_bond_type",
    ]
    lines += [
        f"  {labels[i]:<24} {labels[j]:<24} {d:.8f} {format_symflag(T):>8} {bt}"
        for i, j, T, d, bt in bonds
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    check = read_bonded_cif(path)
    if len(check.bonds) != len(bonds) or not np.allclose(check.frac, frac, atol=1e-9):
        raise RuntimeError(f"Supercell CIF {path} did not read back as written.")
    return mapping, labels


def fold_to_primitive(
    frac_super: np.ndarray, mapping: np.ndarray, rep: Sequence[int], h_prim: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Take copy 0 of a relaxed supercell as the primitive cell.

    The relaxation keeps the supercell symmetric under the unit-cell
    translations, so all copies coincide after removing their translation.
    frac_super holds supercell fractional coordinates, continuous with the
    supercell CIF; mapping is the [primitive index, na, nb, nc] list from
    write_supercell_cif. Returns primitive fractional coordinates of copy 0,
    continuous with the primitive CIF, and the largest distance (A) of any copy
    from copy 0, which checks that the symmetry held.
    """
    M = np.asarray(rep, dtype=int)
    prim_index, copy = mapping[:, 0], mapping[:, 1:4]
    per_copy = frac_super * M - copy
    n_prim = int(prim_index.max()) + 1
    first = np.all(copy == 0, axis=1)
    folded = np.full((n_prim, 3), np.nan)
    folded[prim_index[first]] = per_copy[first]
    if np.isnan(folded).any():
        raise ValueError("A unit-cell atom has no copy 0")
    spread = np.linalg.norm((per_copy - folded[prim_index]) @ h_prim, axis=1)
    return folded, float(spread.max())
