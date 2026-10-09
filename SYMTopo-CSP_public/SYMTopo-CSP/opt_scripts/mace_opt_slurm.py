import argparse
import io
import json
import os
import random
import sys
import time
import warnings
from contextlib import redirect_stdout

import torch
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS
from joblib import Parallel, delayed
from mace.calculators import mace_mp, mace_off

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)
#####################################################################

EV_TO_KJ_PER_MOL = 96.485


def resolve_target_folder(root_path, target_folder):
    """Resolve a relative or absolute CIF input folder."""
    root_path = os.path.abspath(os.path.expanduser(root_path))
    target_folder = os.path.expanduser(target_folder)
    if not os.path.isabs(target_folder):
        target_folder = os.path.join(root_path, target_folder)
    return root_path, os.path.abspath(target_folder)


def calculate_density(crystal):
    total_mass = sum(crystal.get_masses())
    volume = crystal.get_volume()
    density = total_mass / (volume * 10**-24) / (6.022140857 * 10**23)
    return density


def run_calculation_one(file, calc):
    target_path = os.path.join(target_folder, file)
    press_path = os.path.join(press_folder, file[:-4] + press_suffix)
    final_path = os.path.join(final_folder, file[:-4] + final_suffix)
    json_path = os.path.join(json_folder, file[:-4] + json_suffix)
    lock_path = os.path.join(lock_folder, file[:-4])

    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        return 0
    if os.path.exists(lock_path):
        return 0
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return 0
    except OSError as e:
        print(f"Lock error on {file}: {e}")
        return 0
    try:
        with io.StringIO() as buf, redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            crystal = read(target_path)
            crystal.calc = calc
            sf = FrechetCellFilter(crystal, scalar_pressure=0.0006)
            optimizer = BFGS(sf)
            optimizer.run(fmax=0.01, steps=3000)
            crystal.write(press_path)

            crystal = read(press_path)
            crystal.calc = calc
            sf = FrechetCellFilter(crystal)
            optimizer = BFGS(sf)
            optimizer.run(fmax=0.01, steps=3000)
            density = calculate_density(crystal)
            crystal.write(final_path)

            energy = crystal.get_potential_energy()
            step_used = optimizer.get_number_of_steps()
            energy_per_atom_kj = (
                energy / len(crystal.get_atomic_numbers()) * EV_TO_KJ_PER_MOL
            )
            if step_used < 3000:
                status = "success"
            else:
                status = "not_converged"
            new_row = {
                "name": file[:-4],
                "density": density,
                "energy_kj": energy_per_atom_kj,
                "step_used": step_used,
                "status": status,
            }
    except Exception as e:
        status = str(e)
        new_row = {
            "name": file[:-4],
            "density": 100000.0,
            "energy_kj": 100000.0,
            "step_used": 100000,
            "status": status,
        }
    with open(json_path, "w") as json_file:
        json.dump(new_row, json_file, indent=4)
    os.remove(lock_path)
    return 1


def run_single_worker(target_folder, idx):
    pid = os.getpid()
    fd = os.open(
        pid_file,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.write(fd, f"{pid}\n".encode())
    finally:
        os.close(fd)

    time.sleep(idx)
    old_files = set()
    if not use_cpu_only and torch.cuda.is_available():
        device = "cuda"
    else:
        print(
            "CUDA is not available or CPU-only mode is enabled. Using CPU for calculations."
        )
        device = "cpu"
    calc = mace_off(model="small", dispersion=True, device=device)
    while True:
        files = [f for f in os.listdir(target_folder) if f.endswith(".cif")]
        new_files = list(set(files) - old_files)

        if not new_files:
            break

        random.shuffle(new_files)
        for file in new_files:
            run_calculation_one(file, calc)

        old_files.update(new_files)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MACE Optimization with SLURM")
    parser.add_argument(
        "--path", type=str, default=".", help="Path to the working directory"
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument(
        "--target_folder",
        type=str,
        default="structures",
        help="CIF input folder; may be absolute or relative to --path",
    )
    parser.add_argument(
        "--use_cpu_only",
        action="store_true",
        help="Flag to use CPU only for calculations",
    )
    args = parser.parse_args()
    path, target_folder = resolve_target_folder(args.path, args.target_folder)
    n_jobs = args.n_jobs
    use_cpu_only = args.use_cpu_only

    if not os.path.isdir(target_folder):
        parser.error(f"target CIF folder does not exist: {target_folder}")

    json_suffix = ".json"
    press_suffix = "_press.cif"
    final_suffix = "_opt.cif"

    press_folder = os.path.join(path, "cif_result_press")
    final_folder = os.path.join(path, "cif_result_final")
    json_folder = os.path.join(path, "json_result")
    lock_folder = os.path.join(path, "lock")
    pid_file = os.path.join(lock_folder, "pid.txt")

    os.makedirs(press_folder, exist_ok=True)
    os.makedirs(final_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(lock_folder, exist_ok=True)

    Parallel(n_jobs=n_jobs)(
        delayed(run_single_worker)(target_folder, idx) for idx in range(n_jobs)
    )

    files = [f for f in os.listdir(target_folder) if f.endswith(".cif")]
    files_set = {f.removesuffix(".cif") for f in files}
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    json_set = {f.removesuffix(".json") for f in json_files}
    if files_set != json_set:
        sys.exit(0)

    lock_files = [f for f in os.listdir(lock_folder) if f != "pid.txt"]
    if not lock_files:
        os.remove(pid_file)
    os.rmdir(lock_folder)
