# -*- coding: utf-8 -*-

import argparse
import csv
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from math import gcd
from pathlib import Path

import pandas as pd
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from tqdm import tqdm

# =========================
# Default analysis settings.
# =========================
INPUT_PATH = r"./cif_result_final"  # CIF file or directory
OUTPUT_CSV = r"./periodicity_summary.csv"
PROCESSES = 120
MULT = 1.2
RECURSIVE = True


def build_graph(atoms, mult):
    cutoffs = natural_cutoffs(atoms, mult=mult)
    i_list, j_list, s_list = neighbor_list("ijS", atoms, cutoffs)

    g = [[] for _ in range(len(atoms))]
    for i, j, s in zip(i_list, j_list, s_list):
        i = int(i)
        j = int(j)
        s = tuple(int(x) for x in s)
        g[i].append((j, s))
        g[j].append((i, (-s[0], -s[1], -s[2])))
    return g


def connected_components(g):
    seen = [False] * len(g)
    comps = []

    for start in range(len(g)):
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        comp = []

        while q:
            i = q.popleft()
            comp.append(i)
            for j, _ in g[i]:
                if not seen[j]:
                    seen[j] = True
                    q.append(j)

        comps.append(comp)

    return comps


def normalize_period_vector(v):
    """Reduce a periodic direction to a canonical integer triplet."""
    a, b, c = (int(v[0]), int(v[1]), int(v[2]))
    if (a, b, c) == (0, 0, 0):
        return (0, 0, 0)

    d = gcd(gcd(abs(a), abs(b)), abs(c))
    a, b, c = a // d, b // d, c // d

    # Opposite vectors represent the same periodic direction.
    for x in (a, b, c):
        if x != 0:
            if x < 0:
                a, b, c = -a, -b, -c
            break

    return (a, b, c)


def periodic_vectors_component(g, comp):
    """
    Return the periodic direction vectors for one connected component.

    Cell offsets from the ASE neighbor list are accumulated along graph edges.
    A loop with a nonzero accumulated offset establishes periodicity.
    """
    comp = set(comp)
    a = next(iter(comp))

    q = deque([a])
    offsets = {a: (0, 0, 0)}
    vectors = set()

    while q:
        i = q.popleft()
        off = offsets[i]

        for j, s in g[i]:
            if j not in comp:
                continue

            new_off = (off[0] + s[0], off[1] + s[1], off[2] + s[2])

            if j not in offsets:
                offsets[j] = new_off
                q.append(j)
            else:
                old_off = offsets[j]
                loop_vec = (
                    new_off[0] - old_off[0],
                    new_off[1] - old_off[1],
                    new_off[2] - old_off[2],
                )
                if loop_vec != (0, 0, 0):
                    vectors.add(normalize_period_vector(loop_vec))

    return vectors


def cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def vector_dimension(vectors):
    """
    Determine the dimension spanned by the periodic direction vectors.

    One direction is one-dimensional; multiple coplanar directions are
    two-dimensional; otherwise the periodicity is three-dimensional.
    """
    vectors = list(vectors)
    n = len(vectors)

    if n == 0:
        return ""
    if n == 1:
        return 1
    if n == 2:
        return 2

    # Find two nonparallel vectors that define a plane.
    normal = None
    for i in range(n):
        for j in range(i + 1, n):
            cp = cross(vectors[i], vectors[j])
            if cp != (0, 0, 0):
                normal = cp
                break
        if normal is not None:
            break

    # Retain a guard for degenerate collinear inputs.
    if normal is None:
        return 1

    for v in vectors:
        if dot(normal, v) != 0:
            return 3

    return 2


def format_vectors(vectors):
    vectors = sorted(vectors)
    return ";".join(f"({a},{b},{c})" for a, b, c in vectors)


def check_one_file(path, mult):
    try:
        atoms = read(path)
        g = build_graph(atoms, mult)
        comps = connected_components(g)

        all_vectors = set()
        for comp in comps:
            comp_vectors = periodic_vectors_component(g, comp)
            if not comp_vectors:
                return {
                    "file": Path(path).stem,
                    "component_count": len(comps),
                    "all_periodic": False,
                    "periodic_vector_count": "",
                    "periodic_vector_dimension": "",
                    "periodic_vectors": "",
                    "error": "",
                }
            all_vectors.update(comp_vectors)

        return {
            "file": Path(path).stem,
            "component_count": len(comps),
            "all_periodic": True,
            "periodic_vector_count": len(all_vectors),
            "periodic_vector_dimension": vector_dimension(all_vectors),
            "periodic_vectors": format_vectors(all_vectors),
            "error": "",
        }

    except Exception as e:
        return {
            "file": Path(path).stem,
            "component_count": len(comps),
            "all_periodic": False,
            "periodic_vector_count": "",
            "periodic_vector_dimension": "",
            "periodic_vectors": "",
            "error": f"{type(e).__name__}: {e}",
        }


def gather_cifs(inp, recursive):
    p = Path(inp)
    if p.is_file():
        return [p.resolve()]

    files = []
    if recursive:
        files.extend(p.rglob("*.cif"))
        files.extend(p.rglob("*.CIF"))
    else:
        files.extend(p.glob("*.cif"))
        files.extend(p.glob("*.CIF"))

    return sorted(set(f.resolve() for f in files if f.is_file()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".")
    parser.add_argument("--workers", type=int, default=PROCESSES)
    args = parser.parse_args()
    path = Path(args.path).resolve()

    files = gather_cifs(path / INPUT_PATH, RECURSIVE)
    output_csv = path / OUTPUT_CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "component_count",
        "all_periodic",
        "periodic_vector_count",
        "periodic_vector_dimension",
        "periodic_vectors",
        "error",
    ]

    total = len(files)
    periodic_count = 0

    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results_iter = pool.map(
                check_one_file,
                map(str, files),
                repeat(MULT),
            )

            for row in tqdm(
                results_iter,
                total=total,
                desc="Processing CIF",
                unit="file",
            ):
                writer.writerow(row)
                if row["all_periodic"]:
                    periodic_count += 1

    print(
        f"done: total={total}, all_periodic={periodic_count}, "
        f"not_all_periodic={total - periodic_count}, csv={output_csv}"
    )

    result = pd.read_csv(path / "result.csv")
    period = pd.read_csv(output_csv)
    period["name"] = period["file"].str.replace(r"_opt$", "", regex=True)
    out = result.merge(
        period.drop(columns="file"), on="name", how="left", sort=False
    ).query("all_periodic == True")
    out.to_csv(path / "result_validated.csv", index=False)


if __name__ == "__main__":
    main()
