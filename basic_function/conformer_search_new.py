"""Extension-stratified conformer search and connection-site label templates.

Pure energy ranking favours folded conformers whose chain ends touch, while in
the crystal the extended forms are often competitive.
``stratified_conformer_search``:

1. embeds a large ETKDGv3 pool and optimizes it with MMFF94s (default),
   MMFF94 or UFF;
2. removes duplicates by symmetry-aware heavy-atom RMSD (mirror images merged
   for achiral molecules);
3. splits the unique conformers into layers by an extension descriptor, with
   equal-width or equal-count layers, or with user-given boundaries as a
   percentage of the extension range or as percentiles of the pool;
4. keeps, inside each layer, conformers within an energy window of that
   layer's own minimum;
5. picks first the ``n_global`` lowest-energy conformers of the whole pool,
   then each layer's quota in energy order; every pick must be at least
   ``cluster_rmsd`` heavy-atom RMSD from all earlier picks, in any layer;
6. writes labeled XYZ files, a summary CSV and an energy-extension plot.

The default extension descriptor is the mean intramolecular x_site-y_site
distance, which is the quantity that sets the chain repeat. Site labels come
from a labeled XYZ input or from a labeled template; without labels the
distance between the topologically most distant heavy atoms is used.

Default selection: the 3 lowest-energy conformers of the pool, plus 2
conformers from the most extended 20 % of the extension range
(layer edges 80, layer quota 0 2, 3 global picks).

``write_label_template`` writes one quickly embedded structure whose atoms the
user labels with ``x_site``/``y_site`` before running the search.
"""

import csv
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdMolAlign, rdmolfiles
from rdkit.Chem import rdMolDescriptors
from rdkit.Geometry import Point3D

from basic_function import format_parser
from basic_function.link_structures_partitioned import parse_site_label


def _load_mol_from_smiles_or_pdb(smiles_or_path: str) -> Any:
    """
    Load a molecule from a PDB/XYZ file or parse it as a SMILES string.

    Bonds are inferred for coordinate files. The returned molecule is
    hydrogenated and has no conformers, ready for fresh conformer generation.
    """
    is_file = os.path.isfile(smiles_or_path)
    if is_file and smiles_or_path.lower().endswith(".pdb"):
        # Defer sanitization until bonds have been inferred from the geometry.
        mol = Chem.MolFromPDBFile(smiles_or_path, sanitize=False, removeHs=False)
        if mol is None:
            raise ValueError(f"Could not read a molecule from PDB: {smiles_or_path}")
        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol)

    elif is_file and smiles_or_path.lower().endswith(".xyz"):
        # XYZ stores elements and coordinates but no explicit topology.
        mol = rdmolfiles.MolFromXYZFile(smiles_or_path)
        if mol is None:
            raise ValueError(f"Could not read a molecule from XYZ: {smiles_or_path}")
        if mol.GetNumConformers() == 0:
            raise ValueError(f"XYZ input contains no coordinates: {smiles_or_path}")
        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol)

    else:
        # Treat any other input as a SMILES string.
        mol = Chem.MolFromSmiles(smiles_or_path)
        if mol is None:
            raise ValueError(
                "Could not generate a molecule from the supplied SMILES string."
            )

    mol = Chem.AddHs(mol, addCoords=True)

    # Remove source coordinates so all inputs follow the same embedding workflow.
    for conf in list(mol.GetConformers()):
        mol.RemoveConformer(conf.GetId())
    print(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    return mol


# Heavy-atom pairs at least this many bonds apart count as "long range".
LONG_RANGE_MIN_BONDS = 5
# Long-range heavy-atom pairs closer than this (Angstrom) count as a contact.
LONG_RANGE_CONTACT = 4.0
# Largest atomic gradient (kcal/mol/A) an optimized conformer may keep. True
# minima of the test molecule stay below 2e-3; stalled geometries reach 1e3.
MAX_GRADIENT = 0.1

# Default selection when no layering or quota option is given on the CLI.
DEFAULT_LAYER_EDGES = [80.0]
DEFAULT_LAYER_QUOTA = [0, 2]
DEFAULT_N_GLOBAL = 3

# --ff choices: RDKit MMFF variant names and labels used in outputs.
MMFF_VARIANTS = {"mmff": "MMFF94", "mmff94s": "MMFF94s"}
FF_LABELS = {"uff": "UFF", **MMFF_VARIANTS}
DEFAULT_FF = "mmff94s"

# File written by write_label_template and picked up by the search CLI.
LABEL_TEMPLATE_NAME = "label_template.xyz"


# ---------------------------------------------------------------------------
# Input and site labels
# ---------------------------------------------------------------------------


def read_labeled_xyz(path, charge=0, connectivity_only=False):
    """Read an XYZ file that may carry ``label=...`` notes.

    RDKit cannot parse the extra column, so coordinates are rebuilt into a
    clean XYZ block before bond perception. ``connectivity_only`` skips bond
    orders and charges, which is enough for mapping a template's labels.
    Returns the molecule and a {atom_index: normalized_label} dict for atoms
    that had a note.
    """
    molecule = format_parser.read_xyz_file(path)
    lines = [str(len(molecule.atoms)), ""]
    labels = {}
    for idx, atom in enumerate(molecule.atoms):
        x, y, z = atom.cart_xyz
        lines.append(f"{atom.element} {x:.8f} {y:.8f} {z:.8f}")
        if atom.comment.get("raw_xyz_note", "").strip():
            labels[idx] = atom.site_label
    mol = rdmolfiles.MolFromXYZBlock("\n".join(lines) + "\n")
    if mol is None:
        raise ValueError(f"Could not read a molecule from XYZ: {path}")
    if connectivity_only:
        rdDetermineBonds.DetermineConnectivity(mol)
        return mol, labels
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    Chem.SanitizeMol(mol)
    return mol, labels


def _connectivity_only(mol):
    """Copy of ``mol`` with only elements and single bonds, for graph matching."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


def map_template(mol, template):
    """Return ``mapping[template_idx] = mol_idx``, matching on the bond graph.

    The template's bonds come from interatomic distances, so short
    non-bonded contacts (e.g. the diagonals of a four-membered ring) can
    appear as extra bonds. The template therefore only has to contain every
    bond of ``mol``. If the atom order differs, the longest perceived
    contacts are dropped until the bond counts agree before graph matching.
    """
    if mol.GetNumAtoms() != template.GetNumAtoms():
        raise ValueError(
            f"Template has {template.GetNumAtoms()} atoms, molecule has "
            f"{mol.GetNumAtoms()}."
        )

    def bonds(m):
        return {
            frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in m.GetBonds()
        }

    mol_bonds, template_bonds = bonds(mol), bonds(template)
    same_elements = all(
        a.GetAtomicNum() == b.GetAtomicNum()
        for a, b in zip(mol.GetAtoms(), template.GetAtoms())
    )
    if same_elements and mol_bonds <= template_bonds:
        return list(range(mol.GetNumAtoms()))

    # Keep the shortest contacts relative to the covalent-radius sum.
    xyz = template.GetConformer().GetPositions()
    table = Chem.GetPeriodicTable()

    def stretch(bond):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        radii = table.GetRcovalent(bond.GetBeginAtom().GetAtomicNum()) + table.GetRcovalent(
            bond.GetEndAtom().GetAtomicNum()
        )
        return np.linalg.norm(xyz[i] - xyz[j]) / radii

    pruned = Chem.RWMol(template)
    for bond in sorted(template.GetBonds(), key=stretch, reverse=True):
        if pruned.GetNumBonds() <= len(mol_bonds):
            break
        pruned.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    match = _connectivity_only(mol).GetSubstructMatch(_connectivity_only(pruned.GetMol()))
    if len(match) != template.GetNumAtoms():
        raise ValueError(
            "The template does not match the molecule's bond graph: the molecule has "
            f"{len(mol_bonds)} bonds, the template geometry gives {len(template_bonds)}. "
            "Check that the template is the same molecule with all hydrogens."
        )
    print("Template atom order differs from the molecule; labels mapped by graph.")
    return list(match)


def load_molecule(source, template=None, charge=0):
    """Load SMILES/PDB/XYZ input; return (mol without conformers, labels)."""
    if os.path.isfile(source) and source.lower().endswith(".xyz"):
        mol, labels = read_labeled_xyz(source, charge=charge)
        mol = Chem.AddHs(mol, addCoords=True)
        mol.RemoveAllConformers()
        print(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    else:
        mol = _load_mol_from_smiles_or_pdb(source)
        labels = {}

    if template is not None:
        template_mol, template_labels = read_labeled_xyz(template, connectivity_only=True)
        mapping = map_template(mol, template_mol)
        labels = {mapping[t_idx]: label for t_idx, label in template_labels.items()}
        print(f"Site labels taken from template {template}.")
    return mol, labels


def write_label_template(source, out_path, charge=0, seed=-1, overwrite=False):
    """Write one quickly embedded structure for marking connection sites.

    A single ETKDGv3 embedding without force-field optimization; the geometry
    only has to be good enough to identify atoms. The comment line holds a
    SMILES whose atom-map number N is the N-th atom line, so the sites can
    also be found without a viewer. Returns the template path.
    """
    out_dir = Path(out_path) / "conformers"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / LABEL_TEMPLATE_NAME
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} already exists and may hold your labels; pass --overwrite to replace it."
        )

    mol, labels = load_molecule(source, charge=charge)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("Could not embed a 3D structure for the label template.")

    mapped = Chem.Mol(mol)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    mapped_smiles = Chem.MolToSmiles(Chem.RemoveHs(mapped))
    write_xyz(target, mol, mol.GetConformers()[0].GetId(), labels,
              f"label_template map-number=atom-line SMILES {mapped_smiles}")
    print(f"Wrote {target}.\n"
          "Add label=x_site and label=y_site (or typed a_x_site/a_y_site, ...) to the\n"
          "connection atoms, then run the search again without --make_template.")
    return target


def connection_pairs(labels):
    """Return {site_type: (x_index, y_index)} from ``*_x_site``/``*_y_site``."""
    sites = {}
    for idx, label in labels.items():
        for kind, role, _ in parse_site_label(label):
            roles = sites.setdefault(kind, {})
            if role in roles:
                raise ValueError(f"Site type {kind or 'default'!r} has two {role} sites.")
            roles[role] = idx
    for kind, roles in sites.items():
        if set(roles) != {"x", "y"}:
            raise ValueError(
                f"Site type {kind or 'default'!r} needs exactly one x and one y site."
            )
    return {kind: (roles["x"], roles["y"]) for kind, roles in sorted(sites.items())}


# ---------------------------------------------------------------------------
# Shape descriptors
# ---------------------------------------------------------------------------


class ShapeDescriptors:
    """Per-conformer shape descriptors that do not depend on the force field."""

    def __init__(self, mol, site_pairs):
        self.site_pairs = site_pairs
        self.heavy = np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1])
        topo = Chem.GetDistanceMatrix(mol)[np.ix_(self.heavy, self.heavy)]
        iu = np.triu_indices(len(self.heavy), k=1)
        # Topologically most distant heavy atoms: the chain "ends".
        far = topo[iu] == topo[iu].max() if len(iu[0]) else np.zeros(0, bool)
        self.end_pairs = (self.heavy[iu[0][far]], self.heavy[iu[1][far]])
        long_range = topo[iu] >= LONG_RANGE_MIN_BONDS
        self.long_pairs = (self.heavy[iu[0][long_range]], self.heavy[iu[1][long_range]])

    def compute(self, mol, cid):
        conf = mol.GetConformer(cid)
        xyz = conf.GetPositions()
        out = {}
        for kind, (x_idx, y_idx) in self.site_pairs.items():
            out[f"d_{kind or 'site'}"] = float(np.linalg.norm(xyz[x_idx] - xyz[y_idx]))
        if self.site_pairs:
            out["d_xy"] = float(np.mean([out[f"d_{k or 'site'}"] for k in self.site_pairs]))
        out["d_ends"] = _mean_pair_distance(xyz, self.end_pairs)
        heavy_xyz = xyz[self.heavy]
        out["rg"] = float(np.sqrt(((heavy_xyz - heavy_xyz.mean(0)) ** 2).sum(1).mean()))
        out["npr1"] = rdMolDescriptors.CalcNPR1(mol, confId=cid)
        out["npr2"] = rdMolDescriptors.CalcNPR2(mol, confId=cid)
        i, j = self.long_pairs
        out["long_contacts"] = int(
            np.sum(np.linalg.norm(xyz[i] - xyz[j], axis=1) < LONG_RANGE_CONTACT)
        )
        return out


def _mean_pair_distance(xyz, pairs):
    i, j = pairs
    if len(i) == 0:
        return 0.0
    return float(np.linalg.norm(xyz[i] - xyz[j], axis=1).mean())


EXTENSION_KEYS = {"sites": "d_xy", "ends": "d_ends", "rg": "rg"}


# ---------------------------------------------------------------------------
# Heavy-atom RMSD: duplicate removal and clustering
# ---------------------------------------------------------------------------


def is_achiral(mol):
    """True when RDKit finds no tetrahedral stereocenter (mirror images equivalent)."""
    return not Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )


def heavy_atom_rmsd_matrix(mol, positions, mirror=False):
    """Symmetry-aware heavy-atom RMSD between the conformers at ``positions``.

    Works on a hydrogen-free copy, so the output coordinates are never moved.
    With ``mirror``, every conformer is also compared with the x-inverted
    copies of the others and the smaller value is kept. Returns an
    (n_conformers, n_conformers) matrix; rows/columns outside ``positions``
    stay NaN. Memory and time grow as the square of len(positions).
    """
    conformers = list(mol.GetConformers())
    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    heavy = Chem.RemoveHs(mol)
    heavy.RemoveAllConformers()
    copies = [False, True] if mirror else [False]
    for flip in copies:
        for pos in positions:
            xyz = conformers[pos].GetPositions()[heavy_idx]
            if flip:
                xyz[:, 0] *= -1.0
            conf = Chem.Conformer(len(heavy_idx))
            for i, (x, y, z) in enumerate(xyz):
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            heavy.AddConformer(conf, assignId=True)

    n = len(positions)
    packed = rdMolAlign.GetAllConformerBestRMS(heavy, numThreads=0)
    full = np.zeros((n * len(copies), n * len(copies)))
    full[np.tril_indices(len(full), -1)] = packed
    full += full.T
    rmsd = full[:n, :n] if not mirror else np.minimum(full[:n, :n], full[:n, n:])

    out = np.full((len(conformers), len(conformers)), np.nan)
    out[np.ix_(positions, positions)] = rmsd
    return out


def remove_duplicates(order, rmsd, threshold):
    """Energy-ordered removal; returns (unique positions, {dup: kept position}).

    A conformer is a duplicate when its RMSD to any kept conformer is below
    ``threshold``; it is assigned to the closest kept one.
    """
    kept, duplicate_of = [], {}
    for pos in order:
        if kept:
            dist = rmsd[pos, kept]
            hit = int(np.argmin(dist))
            if dist[hit] < threshold:
                duplicate_of[pos] = kept[hit]
                continue
        kept.append(pos)
    return kept, duplicate_of


# ---------------------------------------------------------------------------
# Stratified selection
# ---------------------------------------------------------------------------


def compute_layer_edges(values, n_layers=4, binning="uniform", edges=None, percentiles=None):
    """Return n_layers + 1 layer boundaries for the extension ``values``.

    ``edges`` gives the inner boundaries as percentages of the pool's
    extension range (0 = shortest, 100 = longest conformer), so 20 80 cuts at
    min + 0.2 * (max - min) and min + 0.8 * (max - min). ``percentiles`` gives
    them as percentiles of the pool, so 20 80 puts 20 % of the unique
    conformers in the first layer. Either one fixes the number of layers and
    lets the layers differ. Otherwise ``binning`` splits the range into
    ``n_layers`` equal-width ("uniform") or equal-count ("quantile") layers.
    """
    if edges is not None and percentiles is not None:
        raise ValueError("Give either layer edges or layer percentiles, not both.")
    low, high = float(values.min()), float(values.max())
    if edges is not None or percentiles is not None:
        given = np.asarray(edges if edges is not None else percentiles, dtype=float)
        if np.any(given <= 0) or np.any(given >= 100):
            raise ValueError("Layer boundaries must lie strictly between 0 and 100.")
        if edges is not None:
            inner = low + given / 100.0 * (high - low)
        else:
            inner = np.percentile(values, given)
    elif binning == "quantile":
        inner = np.quantile(values, np.linspace(0.0, 1.0, n_layers + 1))[1:-1]
    else:
        inner = np.linspace(low, high, n_layers + 1)[1:-1]
    if np.any(np.diff(inner) <= 0):
        raise ValueError(f"Layer boundaries must increase strictly, got {inner.tolist()}.")
    return np.concatenate([[low], inner, [high]])


def assign_layers(values, edges):
    # Inner edges only, so the extremes always fall in the first/last layer.
    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)


def leader_clusters(members, rmsd, threshold):
    """Energy-ordered leader clustering; ``members`` must be sorted by energy.

    A conformer becomes a new leader when its heavy-atom RMSD to every
    existing leader is at least ``threshold``; otherwise it joins the
    closest leader.
    """
    leaders, leader_of = [], {}
    for pos in members:
        if leaders:
            dist = rmsd[pos, leaders]
            hit = int(np.argmin(dist))
            if dist[hit] < threshold:
                leader_of[pos] = leaders[hit]
                continue
        leaders.append(pos)
        leader_of[pos] = pos
    return leaders, leader_of


def layer_quotas(total, n_layers, explicit=None):
    """Split ``total`` over layers; the remainder goes to the extended layers."""
    if explicit is not None:
        return list(explicit)
    base, remainder = divmod(total, n_layers)
    return [base + (1 if layer >= n_layers - remainder else 0) for layer in range(n_layers)]


def resolve_layering(n_layers, binning, layer_edges, layer_percentiles, layer_quota,
                     need_structure):
    """Fill in default layering and quotas; raise ValueError on inconsistent input.

    With no layering option at all, the layers are DEFAULT_LAYER_EDGES (as % of
    the extension range). ``n_layers`` or ``binning`` alone selects equal
    layers (4 by default). With no quota option, DEFAULT_LAYER_QUOTA is used,
    which only fits the default two layers.
    Returns (n_layers, binning, layer_edges, layer_percentiles, layer_quota,
    need_structure).
    """
    if layer_edges is not None and layer_percentiles is not None:
        raise ValueError("Give either --layer_edges or --layer_percentiles, not both.")
    inner = layer_edges if layer_edges is not None else layer_percentiles
    if inner is not None:
        if binning is not None:
            raise ValueError("--binning only applies without --layer_edges/--layer_percentiles.")
        if n_layers is not None and n_layers != len(inner) + 1:
            raise ValueError(f"{len(inner)} inner boundaries give {len(inner) + 1} layers, "
                             f"not --n_layers {n_layers}.")
        n_layers = len(inner) + 1
    elif n_layers is None and binning is None:
        layer_edges = list(DEFAULT_LAYER_EDGES)
        n_layers = len(layer_edges) + 1
    else:
        n_layers = 4 if n_layers is None else n_layers
        binning = binning or "uniform"
    if n_layers < 1:
        raise ValueError("--n_layers must be at least 1.")

    if layer_quota is not None and need_structure is not None:
        raise ValueError("Give either --layer_quota or --need_structure, not both.")
    if layer_quota is None and need_structure is None:
        if n_layers != len(DEFAULT_LAYER_QUOTA):
            raise ValueError(f"The default --layer_quota {DEFAULT_LAYER_QUOTA} fits "
                             f"{len(DEFAULT_LAYER_QUOTA)} layers; with {n_layers} layers "
                             "give --layer_quota or --need_structure.")
        layer_quota = list(DEFAULT_LAYER_QUOTA)
    if layer_quota is not None:
        if len(layer_quota) != n_layers:
            raise ValueError(f"--layer_quota needs {n_layers} values, one per layer.")
        if min(layer_quota) < 0:
            raise ValueError("--layer_quota values must not be negative.")
    elif need_structure < 0:
        raise ValueError("--need_structure must not be negative.")
    return n_layers, binning, layer_edges, layer_percentiles, layer_quota, need_structure


def select_conformers(pool, layer_of, in_window, rmsd, n_global, quotas, min_rmsd):
    """Pick global lowest-energy conformers first, then per-layer conformers.

    ``pool`` must be sorted by energy. Every pick must be at least ``min_rmsd``
    (heavy-atom RMSD) from all earlier picks, whatever their layer, so a layer
    never re-picks a global conformer or a near copy of a neighbour layer's
    pick; the next conformer of that layer is tried instead. Global picks
    ignore layers and energy windows; layer picks come from ``in_window``.
    Slots a layer cannot fill go to other layers, extended layers first.

    Returns (global picks, picks per layer, {rejected: blocking pick}).
    """
    picked, blocked_by = [], {}

    def accept(pos):
        if picked:
            dist = rmsd[pos, picked]
            hit = int(np.argmin(dist))
            if dist[hit] < min_rmsd:
                blocked_by.setdefault(pos, picked[hit])
                return False
        picked.append(pos)
        return True

    global_picks = []
    for pos in pool:
        if len(global_picks) >= n_global:
            break
        if accept(pos):
            global_picks.append(pos)

    n_layers = len(quotas)
    by_layer = [[] for _ in range(n_layers)]
    for pos in pool:  # global energy order; each layer stops at its quota
        layer = layer_of[pos]
        if pos in in_window and pos not in picked and len(by_layer[layer]) < quotas[layer]:
            if accept(pos):
                by_layer[layer].append(pos)

    spare = sum(quotas) - sum(len(chosen) for chosen in by_layer)
    while spare > 0:
        progressed = False
        for layer in reversed(range(n_layers)):
            if spare == 0:
                break
            for pos in pool:
                if layer_of[pos] == layer and pos in in_window and pos not in picked:
                    if accept(pos):
                        by_layer[layer].append(pos)
                        spare -= 1
                        progressed = True
                        break
        if not progressed:
            break
    for pos in picked:
        blocked_by.pop(pos, None)
    return global_picks, by_layer, blocked_by


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def optimize(mol, ff, max_iters):
    """Optimize every conformer in place; return (energies, converged, max_gradient).

    RDKit can report convergence for a geometry stuck far from a minimum, so a
    conformer only counts as converged when its largest atomic gradient is
    also below MAX_GRADIENT.
    """
    if ff in MMFF_VARIANTS:
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise ValueError("MMFF94 parameters are missing for this molecule; use --ff uff.")
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=0, maxIters=max_iters, mmffVariant=MMFF_VARIANTS[ff]
        )
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=MMFF_VARIANTS[ff])
        fields = [AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
                  for conf in mol.GetConformers()]
    else:
        if not AllChem.UFFHasAllMoleculeParams(mol):
            raise ValueError("UFF parameters are missing for this molecule.")
        results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=max_iters)
        fields = [AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
                  for conf in mol.GetConformers()]
    max_gradient = np.array(
        [np.linalg.norm(np.reshape(field.CalcGrad(), (-1, 3)), axis=1).max() for field in fields]
    )
    converged = np.array([flag == 0 for flag, _ in results]) & (max_gradient < MAX_GRADIENT)
    energies = np.array([energy for _, energy in results], dtype=float)
    return energies, converged, max_gradient


def stratified_conformer_search(
    source,
    out_path,
    template=None,
    charge=0,
    num_conformers=2000,
    rms_thresh=0.3,
    with_random=False,
    seed=-1,
    ff=DEFAULT_FF,
    max_iters=2000,
    extension="auto",
    n_layers=None,
    binning=None,
    layer_edges=None,
    layer_percentiles=None,
    energy_window=10.0,
    max_rel_energy=None,
    dedup_rmsd=0.3,
    cluster_rmsd=0.3,
    keep_mirror=False,
    need_structure=None,
    layer_quota=None,
    n_global=DEFAULT_N_GLOBAL,
    plot=True,
    overwrite=False,
):
    n_layers, binning, layer_edges, layer_percentiles, layer_quota, need_structure = (
        resolve_layering(n_layers, binning, layer_edges, layer_percentiles,
                         layer_quota, need_structure)
    )
    out_dir = Path(out_path) / "conformers"
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("conformer_[GL]*.xyz")) and not overwrite:
        raise FileExistsError(
            f"{out_dir} already holds stratified conformers; pass --overwrite or a new --path."
        )

    mol, labels = load_molecule(source, template=template, charge=charge)
    site_pairs = connection_pairs(labels)
    if template is not None and not site_pairs:
        raise ValueError(f"Template {template} has no x_site/y_site labels yet; label it first.")
    if not site_pairs:
        print("Warning: no connection-site labels; the written conformers must be labeled "
              "before structure generation (see --make_template).")
    if extension == "auto":
        extension = "sites" if site_pairs else "ends"
    if extension == "sites" and not site_pairs:
        raise ValueError("--extension sites needs x/y site labels (labeled XYZ or --template).")
    ext_key = EXTENSION_KEYS[extension]
    print(f"Connection sites: {site_pairs or 'none'}; extension descriptor: {ext_key}")

    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.pruneRmsThresh = rms_thresh
    params.useRandomCoords = with_random
    params.randomSeed = seed
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params))
    if not conf_ids:
        raise RuntimeError("No conformer could be embedded.")
    print(f"Embedded {len(conf_ids)} conformers.")

    shapes = ShapeDescriptors(mol, site_pairs)
    ext_before = np.array([shapes.compute(mol, cid)[ext_key] for cid in conf_ids])
    energies, converged, max_gradient = optimize(mol, ff, max_iters)
    descriptors = [shapes.compute(mol, cid) for cid in conf_ids]
    ext = np.array([d[ext_key] for d in descriptors])

    mirror = is_achiral(mol) and not keep_mirror
    order = [int(p) for p in np.argsort(energies) if converged[p]]
    start = time.perf_counter()
    rmsd = heavy_atom_rmsd_matrix(mol, order, mirror=mirror)
    print(
        f"Heavy-atom RMSD matrix for {len(order)} conformers in "
        f"{time.perf_counter() - start:.1f} s; mirror images "
        f"{'merged' if mirror else 'kept apart'}."
    )

    e_min = energies[order[0]]
    rel = energies - e_min
    unique, duplicate_of = remove_duplicates(order, rmsd, dedup_rmsd)
    pool = [p for p in unique if max_rel_energy is None or rel[p] <= max_rel_energy]
    print(
        f"{int(converged.sum())} converged, {len(unique)} unique, {len(pool)} in pool "
        f"(E_min = {e_min:.4f} kcal/mol)."
    )
    stalled = int(np.sum(max_gradient >= MAX_GRADIENT))
    if stalled:
        print(f"{stalled} conformer(s) kept a gradient >= {MAX_GRADIENT} kcal/mol/A after "
              "optimization and were treated as unconverged.")

    edges = compute_layer_edges(ext[pool], n_layers, binning, layer_edges, layer_percentiles)
    n_layers = len(edges) - 1
    layer_of = dict(zip(pool, assign_layers(ext[pool], edges).tolist()))
    quotas = layer_quotas(need_structure, n_layers, layer_quota)
    if len(quotas) != n_layers:
        raise ValueError(f"{len(quotas)} layer quotas given for {n_layers} layers.")

    # Leader clusters are a diagnostic only; selection below applies the same
    # RMSD rule across layers and to the global picks.
    layer_min, in_window, leader_of, leaders_by_layer = {}, set(), {}, []
    for layer in range(n_layers):
        members = [p for p in pool if layer_of[p] == layer]  # pool is energy-ordered
        if not members:
            print(f"Warning: layer {layer} is empty.")
            leaders_by_layer.append([])
            continue
        layer_min[layer] = energies[members[0]]
        window = [p for p in members if energies[p] - layer_min[layer] <= energy_window]
        in_window.update(window)
        leaders, assignment = leader_clusters(window, rmsd, cluster_rmsd)
        leader_of.update(assignment)
        leaders_by_layer.append(leaders)
    global_picks, selected_by_layer, blocked_by = select_conformers(
        pool, layer_of, in_window, rmsd, n_global, quotas, cluster_rmsd
    )

    names = {pos: f"conformer_G-{rank}" for rank, pos in enumerate(global_picks)}
    for layer, chosen in enumerate(selected_by_layer):
        names.update({pos: f"conformer_L{layer}-{rank}" for rank, pos in enumerate(chosen)})
    for pos, name in names.items():
        layer = layer_of[pos]
        write_xyz(
            out_dir / f"{name}.xyz",
            mol,
            conf_ids[pos],
            labels,
            f"{name} Energy = {energies[pos]:.4f} kcal/mol ({FF_LABELS[ff]} opt) "
            f"pick={'global' if pos in global_picks else 'layer'} layer={layer} "
            f"dE_layer={energies[pos] - layer_min[layer]:.4f} "
            f"dE_global={rel[pos]:.4f} {ext_key}={ext[pos]:.3f}",
        )
    chosen_all = list(names)
    nearest_pick = {}
    for pos in pool:
        others_picked = [q for q in chosen_all if q != pos]
        if others_picked:
            dist = rmsd[pos, others_picked]
            hit = int(np.argmin(dist))
            nearest_pick[pos] = (others_picked[hit], float(dist[hit]))

    rows = []
    for pos in sorted(range(len(conf_ids)), key=lambda p: energies[p]):
        if not converged[pos]:
            status = "unconverged"
        elif pos in duplicate_of:
            status = "duplicate"
        elif pos not in layer_of:
            status = "above_max_rel_energy"
        else:
            status = "unique"
        layer = layer_of.get(pos)
        dup = duplicate_of.get(pos)
        row = {
            "conf_id": conf_ids[pos],
            "energy": round(float(energies[pos]), 4),
            "rel_energy": round(float(rel[pos]), 4),
            "status": status,
            "duplicate_of": "" if dup is None else conf_ids[dup],
            "rmsd_to_kept": "" if dup is None else round(float(rmsd[pos, dup]), 3),
            "extension_before_opt": round(float(ext_before[pos]), 3),
            "max_gradient": float(f"{max_gradient[pos]:.3g}"),
        }
        row.update({k: round(v, 3) if isinstance(v, float) else v for k, v in descriptors[pos].items()})
        row.update(
            {
                "layer": "" if layer is None else layer,
                "rel_energy_in_layer": ""
                if layer is None
                else round(float(energies[pos] - layer_min[layer]), 4),
                "in_window": pos in in_window,
                "cluster_leader": conf_ids[leader_of[pos]] if pos in leader_of else "",
                "rmsd_to_leader": round(float(rmsd[pos, leader_of[pos]]), 3)
                if pos in leader_of and leader_of[pos] != pos
                else "",
                "selected_as": names.get(pos, ""),
                "blocked_by": names.get(blocked_by[pos], "") if pos in blocked_by else "",
                "nearest_pick": names[nearest_pick[pos][0]] if pos in nearest_pick else "",
                "rmsd_to_nearest_pick": round(nearest_pick[pos][1], 3)
                if pos in nearest_pick
                else "",
            }
        )
        rows.append(row)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'layer':>5} {ext_key + ' range':>16} {'unique':>7} {'window':>7} "
          f"{'leaders':>8} {'global':>7} {'quota':>6} {'picked':>7} {'blocked':>8} "
          f"{'E_min(rel)':>11}")
    for layer in range(n_layers):
        n_unique = sum(1 for p in pool if layer_of[p] == layer)
        n_window = sum(1 for p in in_window if layer_of[p] == layer)
        n_global_here = sum(1 for p in global_picks if layer_of[p] == layer)
        n_blocked = sum(
            1
            for p, b in blocked_by.items()
            if layer_of[p] == layer and (b in global_picks or layer_of[b] != layer)
        )
        e_text = f"{layer_min[layer] - e_min:.3f}" if layer in layer_min else "-"
        print(f"{layer:>5} {edges[layer]:>7.2f}-{edges[layer + 1]:<8.2f} {n_unique:>7} "
              f"{n_window:>7} {len(leaders_by_layer[layer]):>8} {n_global_here:>7} "
              f"{quotas[layer]:>6} {len(selected_by_layer[layer]):>7} {n_blocked:>8} "
              f"{e_text:>11}")
    print("('global' counts global picks that fall in the layer; 'blocked' counts conformers "
          "rejected for being within --cluster_rmsd of a global pick or of another layer's pick.)")
    if len(chosen_all) > 1:
        sub = rmsd[np.ix_(chosen_all, chosen_all)] + np.diag(np.full(len(chosen_all), np.inf))
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        print(f"Closest pair of picks: {names[chosen_all[i]]} / {names[chosen_all[j]]}, "
              f"RMSD {sub[i, j]:.3f} A.")
    shrunk = int(np.sum((ext_before - ext)[converged] > 1.0))
    print(f"\n{shrunk} of {int(converged.sum())} converged conformers lost more than "
          f"1 A of {ext_key} during optimization.")

    if plot:
        plot_energy_extension(out_dir / "energy_vs_extension.png", ext, rel, pool,
                              layer_of, names, global_picks, edges, ext_key, ff)
    print(f"Done. Wrote {len(names)} conformers and summary.csv to {out_dir}")
    return [conf_ids[p] for p in global_picks] + [
        conf_ids[p] for chosen in selected_by_layer for p in chosen
    ]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_xyz(path, mol, cid, labels, comment):
    """Write one conformer; labeled atoms get a ``label=...`` note."""
    conf = mol.GetConformer(cid)
    lines = [f"{mol.GetNumAtoms()}\n", comment + "\n"]
    for idx, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(idx)
        line = f"{atom.GetSymbol():6} {pos.x:16.8f} {pos.y:16.8f} {pos.z:16.8f}"
        if idx in labels:
            line += f"    label={labels[idx]}"
        lines.append(line + "\n")
    with open(path, "w") as f:
        f.writelines(lines)


def plot_energy_extension(path, ext, rel, pool, layer_of, names, global_picks, edges,
                          ext_key, ff):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping the plot.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    pool = np.asarray(pool)
    layers = np.array([layer_of[p] for p in pool])
    scatter = ax.scatter(ext[pool], rel[pool], c=layers, cmap="viridis", s=12, alpha=0.6)
    layer_picks = np.array([p for p in names if p not in global_picks], dtype=int)
    global_picks = np.asarray(global_picks, dtype=int)
    if len(layer_picks):
        ax.scatter(ext[layer_picks], rel[layer_picks], marker="*", s=180, facecolors="none",
                   edgecolors="red", linewidths=1.2, label="layer pick")
    if len(global_picks):
        ax.scatter(ext[global_picks], rel[global_picks], marker="D", s=80, facecolors="none",
                   edgecolors="black", linewidths=1.2, label="global pick")
    if len(names):
        ax.legend()
    for edge in edges[1:-1]:
        ax.axvline(edge, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"{ext_key} (A)")
    ax.set_ylabel(f"relative {FF_LABELS[ff]} energy (kcal/mol)")
    fig.colorbar(scatter, ax=ax, label="layer")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
