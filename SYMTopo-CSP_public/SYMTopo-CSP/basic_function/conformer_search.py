import os
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdmolfiles


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


def conformer_search(
    smiles_or_pdb: str,
    out_path: str,
    num_conformers: int = 100,
    max_attempts: int = 5000,
    rms_thresh: float = 0.3,
    with_opt: bool | str = False,
    need_structure: int = 10,
    with_random: bool = False,
) -> list[int]:
    """
    Search conformers with ETKDGv3 from SMILES, PDB, or XYZ input.

    ``with_opt=True`` performs UFF optimization, post-optimization RMSD
    deduplication, and energy ranking. ``with_opt="Energy"`` ranks unoptimized
    embeddings by UFF single-point energy. ``with_opt=False`` preserves the
    native ETKDG order without calculating energies. UFF energies are reported
    in kcal/mol.
    """

    out_dir = os.path.join(out_path, "conformers")
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        print(
            "Warning, there is already a 'conformers' folder in this path, skipping mkdir."
        )

    mol = _load_mol_from_smiles_or_pdb(smiles_or_pdb)

    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.pruneRmsThresh = rms_thresh  # Deduplicate during embedding.
    params.useRandomCoords = with_random

    conformer_ids = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_conformers, params=params
    )
    if not conformer_ids:
        raise RuntimeError(
            "No conformer could be generated; relax the settings or increase max_attempts."
        )

    # Normalize the three supported modes: True, "Energy", and False.
    _is_energy_mode = isinstance(with_opt, str) and with_opt.lower() == "energy"
    _is_true_mode = with_opt is True
    _is_false_mode = with_opt is False

    def _uff_single_point_energy(mol, cid):
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        return ff.CalcEnergy()  # kcal/mol

    if _is_true_mode:
        # UFF optimization, energy evaluation, RMSD deduplication, and ranking.
        energies = []
        for cid in conformer_ids:
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
            energies.append(_uff_single_point_energy(mol, cid))

        # Reapply the same RMSD threshold after geometry optimization.
        rmslist = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
        n = len(conformer_ids)
        unique = []
        removed = set()
        for i in range(n):
            if i in removed:
                continue
            unique.append(i)
            for j in range(i + 1, n):
                if j in removed:
                    continue
                idx = j * (j - 1) // 2 + i  # Packed upper-triangular index.
                if rmslist[idx] < rms_thresh:
                    removed.add(j)

        kept = [(conformer_ids[i], energies[i]) for i in unique]
        sorted_confs = sorted(kept, key=lambda x: x[1])

        # Export only the requested number of lowest-energy conformers.
        for i, (cid, energy) in enumerate(sorted_confs[0:need_structure]):
            lines = [f"{mol.GetNumAtoms()}\n"]
            lines.append(
                f"conformer_{i} Energy = {energy:.4f} kcal/mol (UFF opt + RMSD dedup)\n"
            )

            conf = mol.GetConformer(cid)
            for aidx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(aidx)
                pos = conf.GetAtomPosition(aidx)
                lines.append(
                    f"{atom.GetSymbol():6} {pos.x:16.8f} {pos.y:16.8f} {pos.z:16.8f}\n"
                )

            with open(os.path.join(out_dir, f"conformer_{i}.xyz"), "w") as f:
                f.writelines(lines)

        print(f"Done. Wrote {len(sorted_confs)} conformers to {out_dir}")
        return [cid for cid, _ in sorted_confs]

    elif _is_energy_mode:
        # Rank embeddings by UFF single-point energy without optimization.
        energies = []
        for cid in conformer_ids:
            energies.append(_uff_single_point_energy(mol, cid))
        sorted_confs = sorted(zip(conformer_ids, energies), key=lambda x: x[1])

        # Export only the requested number of lowest-energy conformers.
        for i, (cid, energy) in enumerate(sorted_confs[0:need_structure]):
            lines = [f"{mol.GetNumAtoms()}\n"]
            lines.append(
                f"conformer_{i} Energy = {energy:.4f} kcal/mol (UFF single-point)\n"
            )

            conf = mol.GetConformer(cid)
            for aidx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(aidx)
                pos = conf.GetAtomPosition(aidx)
                lines.append(
                    f"{atom.GetSymbol():6} {pos.x:16.8f} {pos.y:16.8f} {pos.z:16.8f}\n"
                )

            with open(os.path.join(out_dir, f"conformer_{i}.xyz"), "w") as f:
                f.writelines(lines)

        print(f"Done. Wrote {len(sorted_confs)} conformers to {out_dir}")
        return [cid for cid, _ in sorted_confs]

    else:
        # Preserve the native ETKDG order without energy evaluation.
        sorted_ids = list(conformer_ids)

        # Export only the requested number, without energy annotations.
        for i, cid in enumerate(sorted_ids[0:need_structure]):
            lines = [f"{mol.GetNumAtoms()}\n"]
            lines.append(f"conformer_{i} (ETKDG order, no energy)\n")

            conf = mol.GetConformer(cid)
            for aidx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(aidx)
                pos = conf.GetAtomPosition(aidx)
                lines.append(
                    f"{atom.GetSymbol():6} {pos.x:16.8f} {pos.y:16.8f} {pos.z:16.8f}\n"
                )

            with open(os.path.join(out_dir, f"conformer_{i}.xyz"), "w") as f:
                f.writelines(lines)

        print(f"Done. Wrote {len(sorted_ids)} conformers (ETKDG order) to {out_dir}")
        return sorted_ids
