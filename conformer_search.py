"""Command-line entry point for ETKDG conformer generation."""

import argparse

from basic_function.conformer_search import conformer_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate conformers for a given molecule."
    )
    parser.add_argument("smiles", type=str, help="SMILES string of the molecule")
    parser.add_argument(
        "--path", type=str, default="./example", help="Output path for conformers"
    )
    parser.add_argument(
        "--num_conformers",
        type=int,
        default=2000,
        help="Number of conformers to generate",
    )
    parser.add_argument(
        "--rms_thresh",
        type=float,
        default=0.3,
        help="RMS threshold for conformer selection",
    )
    parser.add_argument(
        "--with_opt", type=bool, default=True, help="Whether to perform optimization"
    )
    parser.add_argument(
        "--need_structure", type=int, default=10, help="Number of structures needed"
    )
    parser.add_argument(
        "--with_random", action="store_true", help="Whether to use random sampling"
    )

    args = parser.parse_args()
    conformer_search(
        args.smiles,
        out_path=args.path,
        num_conformers=args.num_conformers,
        rms_thresh=args.rms_thresh,
        with_opt=args.with_opt,
        need_structure=args.need_structure,
        with_random=args.with_random,
    )
