"""Run validation, periodicity analysis, deduplication, and reference matching."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    default_workers = os.cpu_count() or 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./example", help="Target directory")
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument(
        "--molecule_single",
        "--molecule-size",
        dest="molecule_single",
        type=int,
        default=None,
        help="Optional molecule atom count forwarded to summary.py",
    )
    parser.add_argument(
        "--folders", nargs="+", help="Optional subdirectories of --path to process"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.molecule_single is not None and args.molecule_single < 1:
        parser.error("--molecule_single must be a positive integer")

    base_dir = Path(args.path).resolve()
    folders = (
        [base_dir / folder for folder in args.folders] if args.folders else [base_dir]
    )
    scripts_dir = Path(__file__).resolve().parent / "postprocess_scripts"
    basic_scripts = [
        "summary.py",
        "periodicity_check.py",
        "duplicate_remove.py",
    ]
    refs_scripts = [
        "fix_cifs.py",
        "compare_cif.py",
    ]
    for folder in folders:
        scripts = basic_scripts.copy()
        if (folder / "refs").exists():
            scripts.extend(refs_scripts)
        print(f"Running workflow: {folder}", flush=True)
        for script in scripts:
            command = [
                sys.executable,
                str(scripts_dir / script),
                "--path",
                str(folder),
                "--workers",
                str(args.workers),
            ]
            if script == "summary.py" and args.molecule_single is not None:
                command.extend(["--molecule_single", str(args.molecule_single)])
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
