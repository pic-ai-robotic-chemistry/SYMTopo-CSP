"""Run the staged SYMTopo-CSP command-line workflow."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    default_workers = os.cpu_count() or 1
    default_gpu_list = [
        i for i in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    ]
    default_jobs_per_gpu = min(10, max(1, default_workers // len(default_gpu_list)))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="./example", help="Path to the target directory"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--gpu_list", nargs="+", default=default_gpu_list, help="List of GPU IDs to use"
    )
    parser.add_argument(
        "--jobs_per_gpu",
        type=int,
        default=default_jobs_per_gpu,
        help="Number of jobs per GPU",
    )
    parser.add_argument(
        "--smiles", type=str, default="", help="SMILES string of the molecule"
    )

    args = parser.parse_args()
    os.makedirs(args.path, exist_ok=True)
    project_path = Path(__file__).resolve().parent

    conformer_path = project_path / "conformer_search.py"
    structure_path = project_path / "structure_generation.py"
    opt_path = project_path / "optimization_mace.py"
    post_path = project_path / "post_process.py"

    if args.smiles != "":
        subprocess.run(
            [sys.executable, conformer_path, args.smiles, "--out_path", args.path]
        )
    else:
        subprocess.run(
            [
                sys.executable,
                structure_path,
                "--path",
                args.path,
                "--workers",
                str(args.workers),
            ]
        )
        subprocess.run(
            [sys.executable, opt_path, "--path", args.path, "--gpu_list"]
            + args.gpu_list
            + ["--jobs_per_gpu", str(args.jobs_per_gpu)]
        )
        subprocess.run(
            [
                sys.executable,
                post_path,
                "--path",
                args.path,
                "--workers",
                str(args.workers),
            ]
        )
