import argparse
import inspect
import os
from multiprocessing import Pool

import pandas as pd
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.io.cif import CifParser
from tqdm import tqdm

# ========== paths ==========
cif_path = None

name_suffix = "_opt.cif"
worker_cnt = 10
sort_by_energy = False


# ========== StructureMatcher: ignore H only ==========
ltol = 0.2
stol = 0.3
angle_tol = 5

TARGETS = []
MATCHER = None


def load_cif(file):
    kwargs = {}
    if "frac_tolerance" in inspect.signature(CifParser.__init__).parameters:
        kwargs["frac_tolerance"] = 0

    parser = CifParser(str(file), **kwargs)
    if hasattr(parser, "parse_structures"):
        structures = parser.parse_structures(primitive=False)
    else:
        structures = parser.get_structures(primitive=False)

    if not structures:
        raise RuntimeError(f"No structure could be read from CIF: {file}")
    return structures[0]


def make_matcher():
    return StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=False,
        scale=True,
        attempt_supercell=False,
        comparator=ElementComparator(),
        ignored_species=["H"],
    )


def build_cif_name(name):
    name = str(name).strip()
    return name if name.endswith(".cif") else name + name_suffix


def init_worker(ref_files, candidate_path):
    global TARGETS, MATCHER, cif_path
    cif_path = candidate_path
    TARGETS = [(os.path.basename(f), load_cif(f)) for f in ref_files]
    MATCHER = make_matcher()


def match_one(task):
    idx, row = task
    name = row.get("name")

    try:
        if pd.isna(name):
            raise ValueError("name is empty")

        cif_file = os.path.join(cif_path, build_cif_name(name))
        if not os.path.exists(cif_file):
            raise FileNotFoundError(cif_file)

        candidate = load_cif(cif_file)

        matched = []
        matched_rmsd = []

        for ref_name, target in TARGETS:
            # get_rms_dist returns (rmsd, max_dist), or None for no match.
            # Store the RMSD in the same order as matched_cif.
            rms_dist = MATCHER.get_rms_dist(target, candidate)
            if rms_dist is not None:
                matched.append(ref_name)
                matched_rmsd.append(float(rms_dist[0]))

        if not matched:
            return None, None

        row["matched_cif"] = matched
        row["matched_rmsd"] = matched_rmsd
        return row, None

    except Exception as e:
        return None, {
            "row_index": idx,
            "name": name,
            "error": str(e),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".")
    parser.add_argument("--workers", type=int, default=worker_cnt)
    args = parser.parse_args()
    path = os.path.abspath(args.path)
    cif_path = os.path.join(path, "cif_result_final")
    refs_path = os.path.join(path, "refs")
    result_table = os.path.join(path, "result_validated.csv")
    output_table = os.path.join(path, "result_matched_to_refs.csv")
    error_table = os.path.join(path, "result_match_to_refs_errors.csv")

    if not os.path.exists(result_table):
        raise FileNotFoundError(f"result_validated.csv does not exist: {result_table}")
    if not os.path.isdir(refs_path):
        raise FileNotFoundError(f"Reference directory does not exist: {refs_path}")

    ref_files = sorted(
        os.path.join(refs_path, f)
        for f in os.listdir(refs_path)
        if f.lower().endswith(".cif")
    )
    if not ref_files:
        raise FileNotFoundError(
            f"No CIF files were found in the reference directory: {refs_path}"
        )

    df = pd.read_csv(result_table)
    if "name" not in df.columns:
        raise ValueError("result_validated.csv must contain a 'name' column")

    if sort_by_energy:
        df = df.sort_values("energy_kj", ascending=True).reset_index(drop=True)

    tasks = [(i, row.to_dict()) for i, row in df.iterrows()]
    matched, errors = [], []

    with Pool(
        processes=args.workers, initializer=init_worker, initargs=(ref_files, cif_path)
    ) as pool:
        for row, err in tqdm(
            pool.imap(match_one, tasks), total=len(tasks), desc="Matching"
        ):
            if row is not None:
                matched.append(row)
            if err is not None:
                errors.append(err)

    pd.DataFrame(
        matched,
        columns=list(df.columns) + ["matched_cif", "matched_rmsd"],
    ).to_csv(output_table, index=False)
    print(f"Matched results saved to: {output_table}")
    print(f"Number of matches: {len(matched)}")
    print(f"Number of reference CIF files: {len(ref_files)}")

    if errors:
        pd.DataFrame(errors).to_csv(error_table, index=False)
        print(f"Error details saved to: {error_table}")
