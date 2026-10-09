import argparse
import os
from multiprocessing import Pool

from ase.io import read, write


def fix_one_cif(cif_path):
    atoms = read(cif_path)
    atoms.write(cif_path, format="cif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    path = os.path.join(os.path.abspath(args.path), "refs")
    cif_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".cif")]
    if args.workers == 1:
        for cif_path in cif_list:
            fix_one_cif(cif_path)
    else:
        with Pool(processes=args.workers) as pool:
            pool.map(fix_one_cif, cif_list)
