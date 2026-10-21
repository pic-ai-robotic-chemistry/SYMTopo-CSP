"""Generate symmetry-packed candidates and reconstruct periodic chain topology."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from basic_function import format_parser, packaged_function
from basic_function.link_structures_partitioned import link_structures

if __name__ == "__main__":
    default_workers = os.cpu_count() or 1
    parser = argparse.ArgumentParser(
        description="Generate linked structures for a given labeled fragment."
    )
    parser.add_argument("--name", type=str, default="TEST", help="Name of the molecule")
    parser.add_argument(
        "--path", type=str, default="./example", help="Path to the molecule's directory"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--generate_only", 
        action="store_true", 
        help="Only generate structures without linking and LAMMPS processing"
    )
    args = parser.parse_args()

    name = args.name
    path = Path(args.path)
    workers = args.workers

    script_path = Path(__file__).resolve().parent / "basic_scripts"
    conformer_file = path / "conformers" / "conformer_tar.xyz"
    ori_path = path / "structures_ori"
    linked_path = path / "linked_structures"
    lammps_path = path / "lammps_data"
    qeq_path = script_path / "qeq_params_openbabel_original.json"
    output_path = path / "structures"
    bad_path = path / "bad_structures"

    molecule = format_parser.read_xyz_file(conformer_file)

    if args.generate_only:
        ori_path = output_path
    packaged_function.CSP_generater_parallel(
        [molecule],
        ori_path,
        need_structure=25,
        space_group_list=[14, 2, 19, 4, 33, 9, 1, 5, 29, 7],
        add_name=name,
        max_workers=workers,
        start_seed="auto",
    )
    if args.generate_only:
        print("Structure generation completed. Exiting as --generate_only flag is set.")
        sys.exit(0)

    link_structures(ori_path, linked_path, max_workers=workers, scale_factor=1.0)

    to_lammps_path = script_path / "to_lammps_data.py"
    run_lammps_path = script_path / "run_lammps_qeq_relaxations.py"
    l2c_path = script_path / "export_lammps_results_to_cif.py"
    filter_path = script_path / "filter_indv.py"

    subprocess.run(
        [
            sys.executable,
            to_lammps_path,
            linked_path,
            lammps_path,
            "--processes",
            str(workers),
        ]
    )
    subprocess.run(
        [
            sys.executable,
            run_lammps_path,
            lammps_path,
            qeq_path,
            "--use-qeq",
            "off",
            "--processes",
            str(workers),
        ]
    )
    subprocess.run(
        [
            sys.executable,
            l2c_path,
            lammps_path,
            output_path,
            "--processes",
            str(workers),
            "--overwrite",
        ]
    )
    subprocess.run(
        [
            sys.executable,
            filter_path,
            output_path,
            bad_path,
            "--processes",
            str(workers),
        ]
    )
