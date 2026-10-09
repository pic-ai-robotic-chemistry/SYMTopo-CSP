import argparse
import inspect
import json
import os
from multiprocessing import Pool

import pandas as pd
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.io.cif import CifParser
from tqdm import tqdm

# ========== paths ==========
name_suffix = "_opt.cif"
worker_cnt = 120

# Stop after this many unique groups; -1 processes the complete dataset.
max_unique_groups = -1
max_re = 10.0

energy_col = "energy_kj"
re_col = "relative_energy"
density_col = "density"

energy_tol = 5.0
density_tol = 0.2


# ========== StructureMatcher: ignore H only ==========
ltol = 0.2
stol = 0.3
angle_tol = 5

MATCHER = None


def load_cif(file):
    kwargs = {}
    if "frac_tolerance" in inspect.signature(CifParser.__init__).parameters:
        kwargs["frac_tolerance"] = 0

    parser = CifParser(str(file), **kwargs)
    if hasattr(parser, "parse_structures"):
        return parser.parse_structures(primitive=False)[0]
    return parser.get_structures(primitive=False)[0]


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


def init_worker():
    global MATCHER
    MATCHER = make_matcher()


def build_cif_name(name):
    name = str(name).strip()
    return name if name.endswith(".cif") else name + name_suffix


def fit_one(task):
    rep_idx, rep_structure, candidate_structure = task
    return rep_idx, MATCHER.fit(rep_structure, candidate_structure)


def find_match(candidate_structure, candidate_energy, candidate_density, groups, pool):
    tasks = [
        (i, g["structure"], candidate_structure)
        for i, g in enumerate(groups)
        if abs(candidate_energy - g["energy"]) <= energy_tol
        and abs(candidate_density - g["density"]) <= density_tol
    ]

    if not tasks:
        return None

    if pool is None:
        matcher = make_matcher()
        for i, rep_structure, _ in tasks:
            if matcher.fit(rep_structure, candidate_structure):
                return i
        return None

    for i, matched in pool.map(fit_one, tasks):
        if matched:
            return i

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".")
    parser.add_argument("--workers", type=int, default=worker_cnt)
    args = parser.parse_args()
    path = os.path.abspath(args.path)
    cif_path = os.path.join(path, "cif_result_final")
    result_table = os.path.join(path, "result_validated.csv")
    output_table = os.path.join(path, "result_deduplicated.csv")

    df = pd.read_csv(result_table)
    if max_re > 0:
        df = df[df[re_col] <= max_re]

    groups = []

    pool = (
        Pool(processes=args.workers, initializer=init_worker)
        if args.workers > 1
        else None
    )

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Deduplicating"):
            row_dict = row.to_dict()

            name = str(row_dict["name"]).strip()
            energy = float(row_dict[energy_col])
            density = float(row_dict[density_col])

            cif_file = os.path.join(cif_path, build_cif_name(name))
            structure = load_cif(cif_file)

            matched_idx = find_match(
                candidate_structure=structure,
                candidate_energy=energy,
                candidate_density=density,
                groups=groups,
                pool=pool,
            )

            if matched_idx is None:
                groups.append(
                    {
                        "row": row_dict,
                        "structure": structure,
                        "energy": energy,
                        "density": density,
                        "same": [],
                    }
                )

                if max_unique_groups != -1 and len(groups) >= max_unique_groups:
                    break
            else:
                groups[matched_idx]["same"].append(name)

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    out = []
    for g in groups:
        row = dict(g["row"])
        row["same_structures"] = json.dumps(g["same"], ensure_ascii=False)
        out.append(row)

    pd.DataFrame(out, columns=list(df.columns) + ["same_structures"]).to_csv(
        output_table,
        index=False,
    )

    print(f"Deduplicated results saved to: {output_table}")
    print(f"Number of unique structures: {len(groups)}")
