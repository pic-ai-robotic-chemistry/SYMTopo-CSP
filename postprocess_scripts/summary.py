import argparse
import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import ase
import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import build_neighbor_list
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm


def read_molecule_atom_count(path):
    molecule_file = os.path.join(path, "conformers", "conformer_tar.xyz")
    try:
        with open(molecule_file, "r") as file:
            atom_count = int(file.readline().strip())
        if atom_count < 1:
            raise ValueError
    except (OSError, ValueError):
        print(
            f"Warning: Could not read a positive molecule atom count from "
            f"{molecule_file}; using 1.",
            flush=True,
        )
        return 1
    return atom_count


def periodic_rank(nl, n_atoms):
    visited = np.zeros(n_atoms, dtype=bool)
    cell_pos = np.zeros((n_atoms, 3), dtype=int)
    periods = []
    visited[0] = True
    q = deque([0])
    while q:
        i = q.popleft()
        js, offsets = nl.get_neighbors(i)
        for j, offset in zip(js, offsets):
            offset = np.asarray(offset, dtype=int)
            new_pos = cell_pos[i] + offset
            if not visited[j]:
                visited[j] = True
                cell_pos[j] = new_pos
                q.append(j)
            else:
                w = new_pos - cell_pos[j]
                if np.any(w):
                    periods.append(w)
    if len(periods) == 0:
        return 0
    periods = np.unique(np.asarray(periods, dtype=int), axis=0)
    return np.linalg.matrix_rank(periods.astype(float))


def check_single_cif_file(cif_file, cutoff_multi):
    if not os.path.exists(cif_file) or os.path.getsize(cif_file) == 0:
        return "cif_missing"

    try:
        atoms = read(cif_file)
        cutoffs = ase.data.vdw_radii[atoms.numbers] * cutoff_multi
        nl = build_neighbor_list(
            atoms, cutoffs=cutoffs, self_interaction=False, bothways=True
        )
        connectivity = nl.get_connectivity_matrix()
        n_components = connected_components(
            connectivity, directed=False, return_labels=False
        )
        if n_components > 1:
            return "cif_disconnected"
        if periodic_rank(nl, len(atoms)) != 3:
            return "cif_disconnected"

    except Exception as e:
        return "cif_error"

    return "success"


def _check_one_cif_worker(args):
    data, cif_folder, cutoff_multi = args

    cif_file = os.path.join(cif_folder, f"{data['name']}_opt.cif")
    status = check_single_cif_file(cif_file, cutoff_multi)

    if status == "success":
        return status, data
    else:
        data["status"] = status
        return status, data


def check_cif_files(data_list, cif_folder, cutoff_multi, max_workers):
    success_list = []
    error_list = []

    tasks = [(data, cif_folder, cutoff_multi) for data in data_list]
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iter = executor.map(_check_one_cif_worker, tasks)
        for row in tqdm(results_iter, total=total, desc="Processing CIF", unit="file"):
            status, data = row
            if status == "success":
                success_list.append(data)
            else:
                error_list.append(data)
    return success_list, error_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--workers", type=int, default=120)
    parser.add_argument("--cutoff_multi", type=float, default=1.1)
    parser.add_argument(
        "--molecule_single",
        "--molecule-size",
        dest="molecule_single",
        type=int,
        default=None,
        help=(
            "Number of atoms in one molecule. If omitted, read it from "
            "conformers/conformer_tar.xyz."
        ),
    )
    args = parser.parse_args()

    path = args.path
    if args.molecule_single is None:
        molecule_atom_count = read_molecule_atom_count(path)
    elif args.molecule_single < 1:
        parser.error("--molecule_single must be a positive integer")
    else:
        molecule_atom_count = args.molecule_single

    json_folder = os.path.join(path, "json_result")
    cif_folder = os.path.join(path, "cif_result_final")
    csv_path = os.path.join(path, "result.csv")
    error_path = os.path.join(path, "error.csv")

    workers = args.workers
    cutoff_multi = args.cutoff_multi

    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    data_list = []
    error_list = []

    for json_file in json_files:
        with open(os.path.join(json_folder, json_file), "r") as f:
            data = json.load(f)

        if data.get("status", "success") != "success":
            error_list.append(data)
        else:
            data["energy_kj"] = float(data["energy_kj"]) * molecule_atom_count
            data_list.append(data)

    if data_list:
        data_list, error_list_extend = check_cif_files(
            data_list, cif_folder, cutoff_multi, max_workers=workers
        )
        error_list.extend(error_list_extend)

    if data_list:
        df_data = pd.DataFrame(data_list)
        if "status" in df_data.columns:
            df_data = df_data.drop(columns=["status"])
        df_data = df_data.sort_values(by="energy_kj").reset_index(drop=True)
        df_data["relative_energy"] = df_data["energy_kj"] - df_data.loc[0, "energy_kj"]
        df_data.to_csv(csv_path, index=False)

    if error_list:
        df_error = pd.DataFrame(error_list)
        df_error.to_csv(error_path, index=False)


if __name__ == "__main__":
    main()
