#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from tqdm import tqdm


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


def is_periodic_component(g, comp):
    comp = set(comp)
    a = next(iter(comp))

    q = deque([(a, (0, 0, 0))])
    seen = {(a, (0, 0, 0))}

    while q:
        i, off = q.popleft()

        for j, s in g[i]:
            if j not in comp:
                continue

            new_off = (off[0] + s[0], off[1] + s[1], off[2] + s[2])

            if j == a and new_off != (0, 0, 0):
                return True

            state = (j, new_off)
            if state not in seen:
                seen.add(state)
                q.append(state)

    return False


def check_one_file(path, mult):
    try:
        atoms = read(path)
        g = build_graph(atoms, mult)
        comps = connected_components(g)

        for comp in comps:
            if not is_periodic_component(g, comp):
                return path, False, ""

        return path, True, ""

    except Exception as e:
        return path, False, f"{type(e).__name__}: {e}"


def unique_path(dst):
    if not dst.exists():
        return dst

    stem = dst.stem
    suffix = dst.suffix
    k = 1
    while True:
        new_dst = dst.with_name(f"{stem}_{k}{suffix}")
        if not new_dst.exists():
            return new_dst
        k += 1


def gather_cifs(inp, recursive):
    p = Path(inp)
    if p.is_file():
        return [p]

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
    parser.add_argument("input", help="CIF file or directory to validate")
    parser.add_argument("bad_dir", help="Directory to receive rejected CIF files")
    parser.add_argument(
        "--processes", type=int, default=max(1, (os.cpu_count() or 1) - 1)
    )
    parser.add_argument("--mult", type=float, default=1.2)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    files = gather_cifs(args.input, args.recursive)
    bad_dir = Path(args.bad_dir).resolve()
    bad_dir.mkdir(parents=True, exist_ok=True)

    total = len(files)
    moved = 0

    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        for path, ok, _ in tqdm(
            pool.map(check_one_file, map(str, files), repeat(args.mult)),
            total=total,
            desc="Filtering structures",
            unit="structure",
        ):
            path = Path(path)

            if not ok:
                dst = unique_path(bad_dir / path.name)
                shutil.move(str(path), str(dst))
                moved += 1

    print(f"done: total={total}, moved={moved}, kept={total - moved}")


if __name__ == "__main__":
    main()
