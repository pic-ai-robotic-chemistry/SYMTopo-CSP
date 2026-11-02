#!/usr/bin/env python3
"""Batch-convert LAMMPS final.data files to CIF using ASE, with multiprocessing.

Usage:
    python export_lammps_results_to_cif.py JOBS_DIR OUTPUT_DIR [--bad-dir BAD_DIR]

Behavior:
- Recursively finds files named ``final.data`` under JOBS_DIR.
- Jobs that to_lammps_data.py ran as a supercell were relaxed under the
  unit-cell translations; copy 0 is written as the relaxed unit cell. If any
  copy deviates from copy 0 by more than --max-fold-spread, the symmetry did not
  hold and the structure is reported as "asymmetric". Other jobs are read with
  ASE using format='lammps-data' as before.
- When the job has metadata.json and a source.cif with a bond loop, the relaxed
  structure is checked against the intended bonds (including their periodic
  images). With --bad-dir, structures whose bonds differ, and asymmetric
  supercells, are written there instead of OUTPUT_DIR.
- Writes one CIF per job directory; the filename is the job directory name plus
  '.cif'. A per-job summary is written to JOBS_DIR/export_manifest.csv.
- Converts files in parallel with multiple processes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, neighbor_list
from tqdm import tqdm

from filter_indv import build_graph, connected_components, is_periodic_component
from supercell_utils import fold_to_primitive, read_bonded_cif

MANIFEST_FIELDS = [
    "job",
    "status",
    "destination",
    "replication",
    "topology",
    "broken",
    "extra",
    "missing",
    "periodic",
    "fold_spread_max",
    "message",
]


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
    parser.add_argument(
        "--bad-dir",
        type=Path,
        default=None,
        help="Write structures whose relaxed bonds differ from the intended ones here",
    )
    parser.add_argument(
        "--mult",
        type=float,
        default=1.2,
        help="Covalent-radius multiplier defining a bond, as in filter_indv.py (default: 1.2)",
    )
    parser.add_argument(
        "--max-fold-spread",
        type=float,
        default=1.0e-3,
        help=(
            "Largest allowed distance (A) of a supercell copy from copy 0; beyond "
            "it the structure is reported as asymmetric (default: 1e-3)"
        ),
    )
    return parser.parse_args()


def read_lammps_frame(path: Path) -> Dict[str, Any]:
    """Box matrix, atom IDs, types, masses and unwrapped fractional coordinates."""
    lines = path.read_text().splitlines()
    box = {"xy": 0.0, "xz": 0.0, "yz": 0.0}
    for line in lines[:60]:
        words = line.split()
        tail = line.rstrip()
        if tail.endswith("xlo xhi"):
            box["xlo"], box["xhi"] = float(words[0]), float(words[1])
        elif tail.endswith("ylo yhi"):
            box["ylo"], box["yhi"] = float(words[0]), float(words[1])
        elif tail.endswith("zlo zhi"):
            box["zlo"], box["zhi"] = float(words[0]), float(words[1])
        elif tail.endswith("xy xz yz"):
            box["xy"], box["xz"], box["yz"] = map(float, words[:3])
    h = np.array(
        [
            [box["xhi"] - box["xlo"], 0.0, 0.0],
            [box["xy"], box["yhi"] - box["ylo"], 0.0],
            [box["xz"], box["yz"], box["zhi"] - box["zlo"]],
        ]
    )
    origin = np.array([box["xlo"], box["ylo"], box["zlo"]])

    def section(name):
        start = next(k for k, l in enumerate(lines) if l.split("#")[0].strip() == name)
        rows, k = [], start + 2
        while k < len(lines) and lines[k].strip():
            rows.append(lines[k].split("#")[0].split())
            k += 1
        return rows

    masses = {int(r[0]): float(r[1]) for r in section("Masses")}
    rows = section("Atoms")  # atom_style full: id mol type q x y z [ix iy iz]
    ids = np.array([int(r[0]) for r in rows])
    types = np.array([int(r[2]) for r in rows])
    xyz = np.array([[float(v) for v in r[4:7]] for r in rows])
    img = np.array([[int(v) for v in r[7:10]] if len(r) >= 10 else [0, 0, 0] for r in rows])
    frac = (xyz - origin) @ np.linalg.inv(h) + img
    order = np.argsort(ids)
    return {"h": h, "ids": ids[order], "types": types[order], "frac": frac[order],
            "masses": masses}


def bond_key(i: int, j: int, s) -> Tuple[int, ...]:
    a = (int(i), int(j), *(int(v) for v in s))
    b = (int(j), int(i), *(-int(v) for v in s))
    return min(a, b)


def compare_topology(symbols, frac, bonds, h, mult) -> Tuple[Dict[str, Any], Atoms]:
    """Compare intended bonds (i, j, t) with the bonds found at mult x covalent radii.

    frac are fractional coordinates continuous with the CIF that defines t, so
    the intended vector of a bond is (frac[j] + t - frac[i]) @ h.
    """
    radii = covalent_radii[[atomic_numbers[s] for s in symbols]]
    wrap = np.floor(frac).astype(int)
    intended, broken = set(), 0
    for i, j, t in bonds:
        length = np.linalg.norm((frac[j] + t - frac[i]) @ h)
        broken += int(length >= mult * (radii[i] + radii[j]))
        intended.add(bond_key(i, j, t + wrap[j] - wrap[i]))
    atoms = Atoms(symbols, scaled_positions=frac - wrap, cell=h, pbc=True)
    ii, jj, ss = neighbor_list("ijS", atoms, natural_cutoffs(atoms, mult=mult))
    found = {bond_key(i, j, s) for i, j, s in zip(ii, jj, ss)}
    graph = build_graph(atoms, mult)
    periodic = all(is_periodic_component(graph, c) for c in connected_components(graph))
    report = {
        "broken": broken,
        "extra": len(found - intended),
        "missing": len(intended - found),
        "periodic": int(periodic),
    }
    report["topology"] = "ok" if found == intended and periodic else "mismatch"
    return report, atoms


def job_replication(job_dir: Path) -> Tuple[int, int, int]:
    meta_path = job_dir / "metadata.json"
    if not meta_path.exists():
        return (1, 1, 1)
    info = json.loads(meta_path.read_text()).get("bond_images", {})
    return tuple(int(v) for v in info.get("replication", [1, 1, 1]))


def check_job(
    job_dir: Path, final_data: Path, mult: float, max_fold_spread: float = 1.0e-3
) -> Tuple[Dict[str, Any], Optional[Atoms]]:
    """Check the relaxed topology; take copy 0 of supercell jobs as the unit cell.

    Returns the report and, for supercell jobs, the folded primitive structure.
    Atoms are matched through metadata.json (LAMMPS atom ID -> CIF label), so the
    check does not depend on which images LAMMPS used.
    """
    meta = json.loads((job_dir / "metadata.json").read_text())
    prim = read_bonded_cif(job_dir / "source.cif")
    if not prim.bonds:
        return {"topology": "unchecked", "message": "source.cif has no bond loop"}, None
    info = meta.get("bond_images", {})
    rep = tuple(int(v) for v in info.get("replication", [1, 1, 1]))
    supercell = rep != (1, 1, 1)
    lmp_cif = read_bonded_cif(job_dir / info["supercell_cif"]) if supercell else prim

    index = {label: k for k, label in enumerate(lmp_cif.labels)}
    cif_of_id = {int(a["atom_id"]): index[a["ciflabel"]] for a in meta["atoms"]}
    data_file = job_dir / Path(meta["output_files"]["data_file"]).name
    frames = {}
    for tag, path in (("initial", data_file), ("final", final_data)):
        frame = read_lammps_frame(path)
        if len(frame["ids"]) != len(lmp_cif.labels):
            raise ValueError(f"{path.name}: {len(frame['ids'])} atoms, CIF has {len(lmp_cif.labels)}")
        frac = np.zeros((len(lmp_cif.labels), 3))
        for atom_id, atom_type, f in zip(frame["ids"], frame["types"], frame["frac"]):
            k = cif_of_id[int(atom_id)]
            expected = atomic_masses[atomic_numbers[lmp_cif.symbols[k]]]
            if abs(frame["masses"][atom_type] - expected) > 0.05:
                raise ValueError(f"{path.name}: atom {atom_id} mass does not match {lmp_cif.symbols[k]}")
            frac[k] = f
        frames[tag] = (frac, frame["h"])

    # lammps-interface may wrap atoms; undo that so coordinates follow the CIF.
    f0, _ = frames["initial"]
    shift = np.rint(f0 - lmp_cif.frac)
    if np.abs(f0 - lmp_cif.frac - shift).max() > 1e-4:
        raise ValueError("initial data file does not match the CIF coordinates")
    f1, h1 = frames["final"]
    f1 = f1 - shift

    report: Dict[str, Any] = {}
    if supercell:
        mapping = np.asarray(info["supercell_atoms"], dtype=int)
        h_prim = h1 / np.asarray(rep)[:, None]
        f1, report["fold_spread_max"] = fold_to_primitive(f1, mapping, rep, h_prim)
        h1 = h_prim
    bonds = [(i, j, t) for i, j, t, *_ in prim.bonds]
    topo, atoms = compare_topology(prim.symbols, f1, bonds, h1, mult)
    report.update(topo)
    if supercell and report["fold_spread_max"] > max_fold_spread:
        report["topology"] = "asymmetric"
    return report, (atoms if supercell else None)


def convert_one(
    data_file_str: str,
    output_dir_str: str,
    bad_dir_str: Optional[str],
    overwrite: bool,
    mult: float,
    max_fold_spread: float,
) -> Dict[str, Any]:
    data_file = Path(data_file_str)
    job_dir = data_file.parent
    job_name = job_dir.name
    good_path = Path(output_dir_str) / f"{job_name}.cif"
    bad_path = Path(bad_dir_str) / f"{job_name}.cif" if bad_dir_str else None
    row: Dict[str, Any] = {"job": job_name}

    if not overwrite and (good_path.exists() or (bad_path is not None and bad_path.exists())):
        row.update(status="skipped", message="CIF already exists")
        return row

    rep = job_replication(job_dir)
    row["replication"] = "x".join(map(str, rep))
    folded = None
    if (job_dir / "metadata.json").exists() and (job_dir / "source.cif").exists():
        try:
            report, folded = check_job(job_dir, data_file, mult, max_fold_spread)
        except Exception as exc:  # noqa: BLE001
            if rep != (1, 1, 1):
                raise  # a supercell cannot be exported without folding it back
            report = {"topology": "unchecked", "message": f"{type(exc).__name__}: {exc}"}
        row.update(report)
    else:
        row.update(topology="unchecked", message="no metadata.json/source.cif")

    atoms = folded if folded is not None else read(str(data_file), format="lammps-data")
    rejected = bad_path is not None and row.get("topology") in ("mismatch", "asymmetric")
    target, other = (bad_path, good_path) if rejected else (good_path, bad_path)
    write(str(target), atoms, format="cif")
    if overwrite and other is not None and other.exists():
        other.unlink()  # drop a stale copy from an earlier export
    row.update(status="ok", destination="bad" if rejected else "output")
    return row


def main() -> int:
    args = parse_args()
    jobs_dir = args.jobs_dir.resolve()
    output_dir = args.output_dir.resolve()
    bad_dir = args.bad_dir.resolve() if args.bad_dir else None

    if not jobs_dir.exists():
        print(f"ERROR: jobs_dir does not exist: {jobs_dir}", file=sys.stderr)
        return 2

    if args.processes < 1:
        print("ERROR: --processes must be >= 1", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    if bad_dir is not None:
        bad_dir.mkdir(parents=True, exist_ok=True)

    data_files = sorted(jobs_dir.rglob(args.pattern))
    if not data_files:
        print(
            f"No files named '{args.pattern}' found under: {jobs_dir}", file=sys.stderr
        )
        return 1

    rows = []
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = {
            executor.submit(
                convert_one,
                str(data_file),
                str(output_dir),
                str(bad_dir) if bad_dir else None,
                args.overwrite,
                args.mult,
                args.max_fold_spread,
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
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"job": futures[future].parent.name, "status": "failed",
                             "message": f"{type(exc).__name__}: {exc}"})

    manifest = jobs_dir / "export_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["job"]):
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})

    def count(key, value):
        return sum(r.get(key) == value for r in rows)

    failed = count("status", "failed")
    print(
        f"\nExported: output={count('destination', 'output')}, "
        f"bad={count('destination', 'bad')}, skipped={count('status', 'skipped')}, "
        f"failed={failed}; topology mismatch={count('topology', 'mismatch')}, "
        f"asymmetric={count('topology', 'asymmetric')}, "
        f"unchecked={count('topology', 'unchecked')}"
    )
    print(f"manifest: {manifest}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
