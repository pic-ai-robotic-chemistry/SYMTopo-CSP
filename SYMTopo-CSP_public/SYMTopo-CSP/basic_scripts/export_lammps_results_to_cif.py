#!/usr/bin/env python3
"""Batch-convert LAMMPS final.data files to CIF using ASE, with multiprocessing.

Usage:
    python export_lammps_results_to_cif_ase_simple_parallel.py JOBS_DIR OUTPUT_DIR

Behavior:
- Recursively finds files named ``final.data`` under JOBS_DIR.
- Reads each file with ASE using format='lammps-data'.
- Writes one CIF per job directory into OUTPUT_DIR.
- CIF filename is the job directory name plus '.cif'.
- Converts files in parallel with multiple processes.

No metadata is used or preserved. Only the output filename follows the job directory name.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ase.io import read, write
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all final.data files under a jobs directory to CIF using ASE."
    )
    parser.add_argument(
        "jobs_dir", type=Path, help="Root directory containing job folders"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write CIF files into"
    )
    parser.add_argument(
        "--pattern",
        default="final.data",
        help="Filename to search for recursively (default: final.data)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CIF files in output_dir",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    return parser.parse_args()


def convert_one(
    data_file_str: str, output_dir_str: str, overwrite: bool
) -> tuple[bool, str]:
    data_file = Path(data_file_str)
    output_dir = Path(output_dir_str)
    job_name = data_file.parent.name
    cif_path = output_dir / f"{job_name}.cif"

    if cif_path.exists() and not overwrite:
        return True, f"skip  {data_file} -> {cif_path}"

    atoms = read(str(data_file), format="lammps-data")
    write(str(cif_path), atoms, format="cif")
    return True, f"ok    {data_file} -> {cif_path}"


def main() -> int:
    args = parse_args()
    jobs_dir = args.jobs_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not jobs_dir.exists():
        print(f"ERROR: jobs_dir does not exist: {jobs_dir}", file=sys.stderr)
        return 2

    if args.processes < 1:
        print("ERROR: --processes must be >= 1", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = sorted(jobs_dir.rglob(args.pattern))
    if not data_files:
        print(
            f"No files named '{args.pattern}' found under: {jobs_dir}", file=sys.stderr
        )
        return 1

    failed = 0
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = {
            executor.submit(
                convert_one, str(data_file), str(output_dir), args.overwrite
            ): data_file
            for data_file in data_files
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Exporting CIF files",
            unit="structure",
        ):
            try:
                future.result()
            except Exception:  # noqa: BLE001
                failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
