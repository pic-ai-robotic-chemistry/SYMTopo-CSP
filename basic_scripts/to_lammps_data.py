#!/usr/bin/env python3
"""
Batch-convert CIF files into LAMMPS input/data files with lammps-interface,
while forcing framework-only handling:
- disable molecule/guest splitting
- force replication = 1x1x1

The script creates one job directory per CIF and writes:
- source.cif
- data.<name>
- in.<name>
- metadata.json
- status.json
- manifest.csv (batch summary in the output root)

Designed for periodic framework structures such as 1D/2D/3D COFs where
lammps-interface may incorrectly classify parts of the framework as discrete
molecules/guests.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

SCRIPT_VERSION = "1.0.0"


@dataclass
class JobResult:
    source_cif: str
    job_name: str
    job_dir: str
    status: str
    message: str
    duration_sec: float
    atom_count: int = 0
    bond_count: int = 0
    angle_count: int = 0
    dihedral_count: int = 0
    improper_count: int = 0
    input_file: str = ""
    data_file: str = ""
    metadata_file: str = ""
    status_file: str = ""


@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy/scientific/python objects into JSON-safe values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]

    # numpy-like scalars / arrays
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    return str(obj)


def safe_node_data(graph: Any, node: int) -> Dict[str, Any]:
    """Support both old and newer networkx-style node access used by lammps-interface."""
    try:
        return graph.nodes[node]
    except Exception:
        return graph.node[node]


def build_job_name(cif_path: Path, input_root: Path) -> str:
    rel = cif_path.resolve().relative_to(input_root.resolve())
    no_suffix = rel.with_suffix("")
    name = "__".join(no_suffix.parts)
    safe = []
    for ch in name:
        safe.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(safe)


def default_workers(user_value: Optional[int]) -> int:
    if user_value is not None and user_value > 0:
        return user_value
    cpu = os.cpu_count() or 1
    return max(1, cpu - 1)


def collect_cif_files(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    files = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def build_li_options(args: argparse.Namespace, cif_file: Path) -> SimpleNamespace:
    """Mirror the attributes expected by lammps-interface's Options class."""
    return SimpleNamespace(
        cif_file=str(cif_file),
        output_cif=False,
        output_pdb=False,
        output_raspa=False,
        force_field=args.force_field,
        mol_ff=args.molecule_ff,
        h_bonding=args.h_bonding,
        dreid_bond_type=args.dreid_bond_type,
        fix_metal=args.fix_metal,
        minimize=False,
        thermal_anneal=False,
        relax=False,
        bulk_moduli=False,
        thermal_scaling=False,
        npt=False,
        nvt=False,
        cutoff=args.cutoff,
        replication="1x1x1",
        orthogonalize=False,
        random_vel=False,
        dump_dcd=0,
        dump_xyz=0,
        dump_lammpstrj=0,
        restart=False,
        tol=args.tol,
        neighbour_size=args.neighbour_size,
        iter_count=10,
        max_dev=0.01,
        temp=298.0,
        pressure=1.0,
        nprodstp=200000,
        neqstp=200000,
    )


def import_lammps_modules():
    try:
        import lammps_interface  # type: ignore
        from lammps_interface.lammps_main import LammpsSimulation  # type: ignore
        from lammps_interface.structure_data import from_CIF  # type: ignore

        return lammps_interface, LammpsSimulation, from_CIF
    except Exception as exc:
        raise RuntimeError(
            "Could not import lammps_interface. Install it in the active Python "
            "environment and verify that `import lammps_interface` succeeds."
        ) from exc


def make_framework_only_simulation_class(LammpsSimulation):
    class FrameworkOnlyLammpsSimulation(LammpsSimulation):
        def split_graph(self):  # type: ignore[override]
            """Hard-disable molecule/guest splitting for periodic framework systems."""
            self.molecules = []
            self.subgraphs = []
            self.molecule_types = {}
            self.framework = True
            self.separate_molecule_types = True
            return None

        def compute_simulation_size(self):  # type: ignore[override]
            """Force replication to 1x1x1 and disable orthogonalization."""
            self.options.replication = "1x1x1"
            self.options.orthogonalize = False
            super().compute_simulation_size()
            self.supercell = (1, 1, 1)

    return FrameworkOnlyLammpsSimulation


def collect_cell_metadata(cell: Any) -> Dict[str, Any]:
    keys = [
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "lx",
        "ly",
        "lz",
        "xy",
        "xz",
        "yz",
    ]
    data: Dict[str, Any] = {}
    for key in keys:
        if hasattr(cell, key):
            data[key] = sanitize_for_json(getattr(cell, key))
    return data


def collect_atom_type_metadata(sim: Any) -> List[Dict[str, Any]]:
    atom_types: List[Dict[str, Any]] = []
    for type_idx in sorted(sim.unique_atom_types.keys()):
        rep_node, rep_data = sim.unique_atom_types[type_idx]
        atom_types.append(
            {
                "lammps_type": int(type_idx),
                "representative_atom_id": int(rep_node),
                "element": rep_data.get("element"),
                "force_field_type": rep_data.get("force_field_type"),
                "mass": sanitize_for_json(rep_data.get("mass")),
                "type_molecule_group": sanitize_for_json(
                    sim.type_molecules.get(type_idx, 0)
                ),
            }
        )
    return atom_types


def collect_atoms_metadata(graph: Any) -> List[Dict[str, Any]]:
    atoms: List[Dict[str, Any]] = []
    for node in sorted(graph.nodes()):
        data = safe_node_data(graph, node)
        atoms.append(
            {
                "atom_id": int(node),
                "ciflabel": data.get("ciflabel"),
                "element": data.get("element"),
                "force_field_type": data.get("force_field_type"),
                "lammps_type": sanitize_for_json(data.get("ff_type_index")),
                "charge": sanitize_for_json(data.get("charge")),
                "mass": sanitize_for_json(data.get("mass")),
                "molid": sanitize_for_json(data.get("molid", 0)),
                "cartesian_coordinates": sanitize_for_json(
                    data.get("cartesian_coordinates")
                ),
                "fractional_coordinates": sanitize_for_json(
                    data.get("fractional_coordinates")
                ),
            }
        )
    return atoms


def collect_bond_summary(graph: Any) -> Dict[str, Any]:
    total_bonds = 0
    cross_boundary_bonds = 0
    symflags_sample: List[Dict[str, Any]] = []
    for n1, n2, data in graph.edges_iter2(data=True):
        total_bonds += 1
        symflag = data.get("symflag")
        if symflag not in (None, "."):
            cross_boundary_bonds += 1
            if len(symflags_sample) < 20:
                symflags_sample.append(
                    {
                        "n1": int(n1),
                        "n2": int(n2),
                        "symflag": sanitize_for_json(symflag),
                    }
                )
    return {
        "total_bonds": total_bonds,
        "cross_boundary_bonds": cross_boundary_bonds,
        "cross_boundary_bonds_sample": symflags_sample,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(sanitize_for_json(payload), fh, ensure_ascii=False, indent=2)


def run_single_job(
    cif_path_str: str,
    input_root_str: str,
    output_root_str: str,
    args_dict: Dict[str, Any],
) -> Dict[str, Any]:
    start = time.time()
    cif_path = Path(cif_path_str).resolve()
    input_root = Path(input_root_str).resolve()
    output_root = Path(output_root_str).resolve()
    args = argparse.Namespace(**args_dict)
    job_name = build_job_name(cif_path, input_root)
    job_dir = output_root / job_name
    status_file = job_dir / "status.json"
    metadata_file = job_dir / "metadata.json"

    def finalize(result: JobResult) -> Dict[str, Any]:
        return asdict(result)

    try:
        if job_dir.exists():
            if args.skip_existing and metadata_file.exists() and status_file.exists():
                result = JobResult(
                    source_cif=str(cif_path),
                    job_name=job_name,
                    job_dir=str(job_dir),
                    status="skipped",
                    message="metadata.json and status.json already exist; skipped by --skip-existing.",
                    duration_sec=time.time() - start,
                    metadata_file=str(metadata_file),
                    status_file=str(status_file),
                )
                return finalize(result)
            if args.overwrite:
                shutil.rmtree(job_dir)
            else:
                raise FileExistsError(
                    f"Output directory already exists: {job_dir}. "
                    "Use --overwrite or --skip-existing."
                )

        job_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(cif_path, job_dir / "source.cif")

        with suppress_output():
            lammps_interface, LammpsSimulation, from_CIF = import_lammps_modules()
        FrameworkOnlyLammpsSimulation = make_framework_only_simulation_class(
            LammpsSimulation
        )
        options = build_li_options(args, cif_path)

        with suppress_output():
            with pushd(job_dir):
                cell, graph = from_CIF(str(cif_path))
                sim = FrameworkOnlyLammpsSimulation(options)
                sim.set_cell(cell)
                sim.set_graph(graph)
                sim.split_graph()  # hard-disabled; keeps all atoms in framework
                sim.assign_force_fields()
                sim.compute_simulation_size()  # hard-forced to replication = 1x1x1
                sim.merge_graphs()
                sim.write_lammps_files(wd=str(job_dir))

        input_file = job_dir / f"in.{sim.name}"
        data_file = job_dir / f"data.{sim.name}"
        table_file = job_dir / f"table.{sim.name}"

        metadata = {
            "script_version": SCRIPT_VERSION,
            "source_cif": str(cif_path),
            "copied_source_cif": str(job_dir / "source.cif"),
            "job_name": job_name,
            "job_dir": str(job_dir),
            "lammps_interface_version": getattr(
                lammps_interface, "__version__", "unknown"
            ),
            "generation_policy": {
                "force_field": args.force_field,
                "molecule_guest_splitting_disabled": True,
                "forced_replication": "1x1x1",
                "orthogonalize": False,
                "h_bonding": bool(args.h_bonding),
                "dreid_bond_type": args.dreid_bond_type,
                "fix_metal": bool(args.fix_metal),
                "cutoff": float(args.cutoff),
                "tolerance": float(args.tol),
                "neighbour_size": int(args.neighbour_size),
            },
            "output_files": {
                "input_file": str(input_file),
                "data_file": str(data_file),
                "table_file": str(table_file) if table_file.exists() else None,
            },
            "cell": collect_cell_metadata(cell),
            "counts": {
                "atoms": int(sim.graph.number_of_nodes()),
                "bonds": int(sim.graph.number_of_edges()),
                "angles": int(sim.graph.count_angles()),
                "dihedrals": int(sim.graph.count_dihedrals()),
                "impropers": int(sim.graph.count_impropers()),
                "atom_types": int(len(sim.unique_atom_types)),
                "bond_types": int(len(sim.unique_bond_types)),
                "angle_types": int(len(sim.unique_angle_types)),
                "dihedral_types": int(len(sim.unique_dihedral_types)),
                "improper_types": int(len(sim.unique_improper_types)),
            },
            "bond_summary": collect_bond_summary(sim.graph),
            "atom_types": collect_atom_type_metadata(sim),
            "atoms": collect_atoms_metadata(sim.graph),
        }
        write_json(metadata_file, metadata)

        result = JobResult(
            source_cif=str(cif_path),
            job_name=job_name,
            job_dir=str(job_dir),
            status="success",
            message="Conversion completed",
            duration_sec=time.time() - start,
            atom_count=metadata["counts"]["atoms"],
            bond_count=metadata["counts"]["bonds"],
            angle_count=metadata["counts"]["angles"],
            dihedral_count=metadata["counts"]["dihedrals"],
            improper_count=metadata["counts"]["impropers"],
            input_file=str(input_file),
            data_file=str(data_file),
            metadata_file=str(metadata_file),
            status_file=str(status_file),
        )

        write_json(
            status_file,
            {
                "status": result.status,
                "message": result.message,
                "source_cif": result.source_cif,
                "job_name": result.job_name,
                "job_dir": result.job_dir,
                "duration_sec": result.duration_sec,
                "counts": metadata["counts"],
                "output_files": metadata["output_files"],
            },
        )
        return finalize(result)

    except Exception as exc:
        job_dir.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        failure_payload = {
            "status": "failed",
            "message": str(exc),
            "source_cif": str(cif_path),
            "job_name": job_name,
            "job_dir": str(job_dir),
            "duration_sec": time.time() - start,
            "traceback": tb,
        }
        write_json(status_file, failure_payload)
        result = JobResult(
            source_cif=str(cif_path),
            job_name=job_name,
            job_dir=str(job_dir),
            status="failed",
            message=str(exc),
            duration_sec=time.time() - start,
            metadata_file=str(metadata_file),
            status_file=str(status_file),
        )
        return finalize(result)


def write_manifest(output_root: Path, results: Iterable[Dict[str, Any]]) -> Path:
    manifest_path = output_root / "manifest.csv"
    rows = list(results)
    fieldnames = [
        "source_cif",
        "job_name",
        "job_dir",
        "status",
        "message",
        "duration_sec",
        "atom_count",
        "bond_count",
        "angle_count",
        "dihedral_count",
        "improper_count",
        "input_file",
        "data_file",
        "metadata_file",
        "status_file",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return manifest_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LAMMPS input files in batches with lammps-interface while "
            "disabling molecule/guest splitting and fixing replication at 1x1x1."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing CIF files")
    parser.add_argument(
        "output_dir", type=Path, help="Root directory for generated jobs"
    )
    parser.add_argument(
        "--pattern", default="*.cif", help="CIF glob pattern (default: *.cif)"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search input_dir recursively"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Worker count (default: CPU count minus one)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs with successful existing output",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing job directories"
    )

    parser.add_argument(
        "--force-field",
        default="Dreiding",
        help="lammps-interface force field (default: Dreiding)",
    )
    parser.add_argument(
        "--molecule-ff",
        default=None,
        help="Compatibility option; molecule splitting is disabled here",
    )
    parser.add_argument(
        "--h-bonding",
        action="store_true",
        help="Pass --h-bonding to the Dreiding force field",
    )
    parser.add_argument(
        "--dreid-bond-type",
        default="harmonic",
        choices=["harmonic", "morse"],
        help="Dreiding bond potential",
    )
    parser.add_argument(
        "--fix-metal", action="store_true", help="Pass --fix-metal to lammps-interface"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=12.5,
        help="Nonbonded cutoff used only during parameterization",
    )
    parser.add_argument(
        "--tol", type=float, default=0.4, help="Topology-detection tolerance"
    )
    parser.add_argument(
        "--neighbour-size", type=int, default=5, help="Neighbor-search depth"
    )

    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(
            f"input_dir does not exist or is not a directory: {args.input_dir}"
        )
    if args.skip_existing and args.overwrite:
        parser.error("--skip-existing and --overwrite are mutually exclusive")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_files = collect_cif_files(input_dir, args.pattern, args.recursive)
    if not cif_files:
        print(
            f"No CIF files found: {input_dir} / pattern={args.pattern}", file=sys.stderr
        )
        return 2

    workers = default_workers(args.processes)
    args_dict = vars(args).copy()
    args_dict["input_dir"] = str(args.input_dir)
    args_dict["output_dir"] = str(args.output_dir)

    results: List[Dict[str, Any]] = []

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                run_single_job,
                str(cif_path),
                str(input_dir),
                str(output_dir),
                args_dict,
            ): cif_path
            for cif_path in cif_files
        }
        for fut in tqdm(
            cf.as_completed(futures),
            total=len(futures),
            desc="Creating LAMMPS data",
            unit="structure",
        ):
            cif_path = futures[fut]
            try:
                result = fut.result()
            except (
                Exception
            ) as exc:  # should be rare; worker already traps most failures
                result = asdict(
                    JobResult(
                        source_cif=str(cif_path),
                        job_name=build_job_name(cif_path, input_dir),
                        job_dir=str(output_dir / build_job_name(cif_path, input_dir)),
                        status="failed",
                        message=f"Worker failed: {exc}",
                        duration_sec=0.0,
                    )
                )
            results.append(result)

    manifest = write_manifest(output_dir, results)
    n_success = sum(r["status"] == "success" for r in results)
    n_failed = sum(r["status"] == "failed" for r in results)
    n_skipped = sum(r["status"] == "skipped" for r in results)

    print(f"\nCompleted: success={n_success}, failed={n_failed}, skipped={n_skipped}")
    print(f"manifest: {manifest}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
