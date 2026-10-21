# SYMTopo-CSP

SYMTopo-CSP generates candidate crystal structures for rectilinear one-dimensional covalent polymers. It implements the packing-first, topology-later strategy described in the accompanying manuscript: minimal molecular fragments are packed under crystallographic symmetry, labeled connection sites are linked across periodic boundaries, and the resulting structures are regularized and ranked.

The repository covers conformer generation, symmetry-guided packing, periodic topology reconstruction, Dreiding/LAMMPS regularization, MACE optimization, periodicity validation, structural deduplication, and optional comparison with reference CIF files. The high-accuracy DFT calculations described in the manuscript are not automated in this repository.

This repository, and especially `all_processes.py`, should be treated as a reference implementation of the overall workflow rather than a universal turnkey launcher. Cluster schedulers, GPU allocation, executable paths, available memory, and suitable levels of parallelism vary substantially between systems. For production calculations, users are encouraged to adapt the commands to their own environment, run the stages as separate modules, inspect intermediate results, and resume from validated outputs instead of running the entire workflow as one monolithic job.

## Workflow

1. Generate fragment conformers with RDKit ETKDGv3.
2. Select one conformer and label the two polymerization sites as `x_site` and `y_site`.
3. Pack the fragment in selected space groups using Sobol quasi-random sampling.
4. Reconstruct periodic covalent links while rejecting finite cyclic assignments.
5. Regularize the linked structures with Dreiding in LAMMPS.
6. Optimize and rank the structures with MACE.
7. Validate periodicity, remove duplicates, and optionally match reference structures.

## Requirements

Python 3.10 or newer is required (3.11 is recommended). The Python dependencies used across the workflow are:

- NumPy, SciPy (1.9 or newer, for `scipy.optimize.milp`), pandas, tqdm, NetworkX, and joblib
- RDKit for conformer generation and optional aromatic-bond assignment
- ASE (3.23 or newer, for `ase.filters.FrechetCellFilter`) for structure I/O, neighbor lists, and optimization
- pymatgen for structure matching and deduplication
- PyTorch and MACE (`mace-torch`) for MLIP optimization
- `lammps_interface`, importable as `lammps_interface`, for Dreiding input generation

LAMMPS must also be installed and its executable must be available as `lmp`, unless another command is supplied to `basic_scripts/run_lammps_qeq_relaxations.py` with `--lmp-cmd`.

One possible environment setup is:

```bash
conda create -n symtopo-csp python=3.11
conda activate symtopo-csp
pip install numpy scipy pandas tqdm networkx joblib rdkit ase pymatgen torch mace-torch
```

Install `lammps_interface` from its upstream distribution in the same environment and verify the external programs before running the pipeline:

```bash
python -c "import lammps_interface"
lmp -help
```

## Input preparation

### Generate conformers

The conformer command accepts a SMILES string or a path to a PDB or XYZ file:

```bash
python conformer_search.py "O=C(OCC)C1=CC=CC=C1" \
  --path ./example \
  --num_conformers 2000 \
  --rms_thresh 0.3 \
  --need_structure 10
```

Generated conformers are written to `example/conformers/conformer_<index>.xyz`.

### Label connection sites

Choose a conformer, add `label=x_site` and `label=y_site` to the two atoms that define polymer propagation, and save the result as:

```text
example/conformers/conformer_tar.xyz
```

The XYZ atom lines may use this form:

```text
C    0.00000000    0.00000000    0.00000000    label=x_site
C    1.40000000    0.00000000    0.00000000    label=y_site
```

Each molecular component must contain exactly one `x_site` and one `y_site`. The topology reconstruction code uses these labels to find a minimum-cost periodic assignment and excludes assignments that form finite molecular cycles.

## Run the staged workflow

Running each stage explicitly is recommended because it exposes the parameters that affect energies and resource use.

### 1. Generate and regularize structures

```bash
python structure_generation.py \
  --name SYSTEM_NAME \
  --path ./example \
  --workers 16
```

This entry point searches space groups 14, 2, 19, 4, 33, 9, 1, 5, 29, and 7. It then reconstructs bonds, generates Dreiding inputs, runs LAMMPS with QEq disabled, converts the relaxed structures back to CIF, and moves nonperiodic structures to `bad_structures`.

The checked-in entry point requests 25 accepted structures per space group as a lightweight default. The manuscript used 2,500 candidates per space group, or 25,000 per benchmark system. Production-scale studies can call the existing API with a larger sampling budget:

```python
from basic_function import format_parser, packaged_function

molecule = format_parser.read_xyz_file("example/conformers/conformer_tar.xyz")
packaged_function.CSP_generater_parallel(
    [molecule],
    "example/structures_ori",
    need_structure=2500,
    space_group_list=[14, 2, 19, 4, 33, 9, 1, 5, 29, 7],
    max_workers=16,
    add_name="SYSTEM_NAME",
    start_seed="auto",
)
```

This call only performs the packing step. It writes to the same `structures_ori/` directory that `structure_generation.py` uses; the linking, LAMMPS, and filtering steps listed above must still be run on these candidates.

### 2. Optimize with MACE

The optimization stage does not require the number of atoms in one repeat fragment. Each worker stores the mean energy per atom in the existing `energy_kj` JSON field. During postprocessing, `summary.py` reads the atom count from the first line of `conformers/conformer_tar.xyz` and converts the stored value to energy per molecular unit before ranking the structures.

Choose exactly one of the following launch methods for a given working directory. The local launcher and the SLURM launcher both invoke `opt_scripts/mace_opt_slurm.py` and write to the same result and lock directories; do not run them concurrently on the same dataset.

#### Option A: local or bare-metal launch

Use `optimization_mace.py` on a workstation or bare-metal node where the GPUs are directly available:

```bash
python optimization_mace.py \
  --path ./example \
  --gpu_list 0 1 \
  --jobs_per_gpu 4
```

This launcher always reads the input CIF files from `<path>/structures`. 

#### Option B: SLURM submission

On a SLURM cluster, first adjust the `#SBATCH` partition and GPU directives in `opt_scripts/slurm.sh` for the target system. Submit the optimization from the repository root with:

```bash
bash opt_scripts/submit.sh \
  --target-path ./example \
  --target-folder structures \
  --gpu-count 8 \
  --tasks-per-node 10
```

The options have the following meanings:

- `--target-path` is the working directory where MACE result, JSON, lock, and log directories are created.
- `--target-folder` contains the input CIF files. It may be an absolute path or a path relative to `--target-path`.
- `--gpu-count` is the number of SLURM jobs to submit. Each job requests one GPU according to `slurm.sh`.
- `--tasks-per-node` is the number of parallel MACE workers started by each submitted job.

The options may be supplied in any order and may be omitted to use their defaults. Run `bash opt_scripts/submit.sh --help` for the current defaults and short-option aliases. Logs are written to `<target-path>/logs` unless `LOG_DIR` is set. The submission scripts use file locks so that the jobs can share one input directory, but they should not be mixed with a simultaneous local launch.

The current worker loads the small MACE-OFF model with dispersion and performs pressure-biased followed by unconstrained cell optimization.

### 3. Validate and postprocess

```bash
python post_process.py --path ./example --workers 16
```

By default, `summary.py` reads the molecule atom count from `conformers/conformer_tar.xyz`. It prints a warning and uses `1` if that value cannot be read. To supply the value explicitly, use:

```bash
python post_process.py \
  --path ./example \
  --workers 16 \
  --molecule_single 12
```

This command runs the following analyses in order:

- `summary.py`: reads the fragment atom count, converts per-atom MACE energies to per-molecule energies, and collects densities. Failed or unconverged MACE runs, and structures whose van der Waals contact network is not a single component periodic in all three directions, are written to `error.csv`; the remaining structures are ranked by energy in `result.csv`.
- `periodicity_check.py`: records periodic directions and retains structures whose components are all periodic.
- `duplicate_remove.py`: considers only structures within 10 kJ/mol of the lowest energy and groups equivalent structures with pymatgen `StructureMatcher` while ignoring hydrogen atoms. Only pairs within 5 kJ/mol and 0.2 g/cm³ of each other are compared. These thresholds are module-level constants at the top of the script.
- `fix_cifs.py` and `compare_cif.py`: run only when `example/refs` exists, then compare the candidates in `result_validated.csv` with reference CIF files.

`fix_cifs.py` rewrites CIF files inside `refs` in place through ASE, including the bundled `example/refs/WIMZEX15.cif`. Keep a copy of reference files if their original formatting or metadata must be preserved.

## Reference convenience driver

`all_processes.py` demonstrates how the major stages can be connected through a two-phase convenience interface. It is primarily a workflow reference and may require changes to resource settings, paths, and stage boundaries for a particular workstation or cluster. With `--smiles`, it only creates conformers:

```bash
python all_processes.py --path ./example --smiles "YOUR_SMILES"
```

After preparing `conformer_tar.xyz`, omit `--smiles` to run structure generation, MACE optimization, and postprocessing:

```bash
python all_processes.py \
  --path ./example \
  --name TEST \
  --workers 16 \
  --gpu_list 0 1 \
  --jobs_per_gpu 4
```

For production studies, prefer the staged commands so that each module can be validated and resumed independently.

## Output layout

The main working directory contains:

```text
example/
├── conformers/                 ETKDG conformers and conformer_tar.xyz
├── structures_ori/             symmetry-packed candidates
├── linked_structures/          candidates with reconstructed periodic bonds
├── lammps_data/                per-structure LAMMPS jobs and metadata
├── structures/                 Dreiding-regularized CIF files
├── bad_structures/             rejected nonperiodic CIF files
├── cif_result_press/           pressure-biased MACE intermediates
├── cif_result_final/           final MACE-optimized CIF files
├── json_result/                per-structure MACE summaries
├── result.csv                  collected successful structures
├── error.csv                   failed, unconverged, or disconnected structures
├── periodicity_summary.csv     periodicity diagnostics
├── result_validated.csv        periodic structures
└── result_deduplicated.csv     unique structures within 10 kJ/mol of the minimum
```

SLURM submissions also write job logs to `logs/` by default. The `lock/` directory is used during MACE optimization and is removed once every input structure has a result.

If `refs/` is present, postprocessing also writes `result_matched_to_refs.csv` and, when needed, `result_match_to_refs_errors.csv`.

## Reproducibility notes

- Structure generation uses seeded Sobol sampling. Filenames record the space group and seed.
- `start_seed="auto"` resumes after the largest existing seed for each space group.
- The DFT refinement and final lattice-energy ranking reported in the manuscript require an external electronic-structure workflow.
- No automated test suite or pinned environment file is currently included. Before a production campaign, validate the installed scientific packages and external executables on a small input set.

When using SYMTopo-CSP in published work, cite the accompanying manuscript and the external methods used for conformer generation, force-field regularization, MACE optimization, structure matching, and electronic-structure refinement.
