"""Command-line entry point for the extension-stratified conformer search.

Two steps:

1. ``python conformer_search.py SMILES --path DIR --make_template`` writes
   DIR/conformers/label_template.xyz, one quickly embedded structure. Add
   ``label=x_site`` and ``label=y_site`` (or typed markers such as
   ``a_x_site``) to the connection atoms in that file.
2. ``python conformer_search.py SMILES --path DIR`` runs the search. The
   labels of DIR/conformers/label_template.xyz are copied to every conformer,
   so the written files can be used as conformer_tar.xyz directly.
"""

import argparse
import os
from pathlib import Path

from basic_function.conformer_search_new import (
    DEFAULT_FF,
    DEFAULT_N_GLOBAL,
    FF_LABELS,
    LABEL_TEMPLATE_NAME,
    resolve_layering,
    stratified_conformer_search,
    write_label_template,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="SMILES string or PDB/XYZ path (XYZ may carry labels)")
    parser.add_argument("--path", default="./example",
                        help="Working directory; files go to <path>/conformers")
    parser.add_argument("--make_template", action="store_true",
                        help="Only write <path>/conformers/label_template.xyz (one fast, "
                             "unoptimized 3D structure) for marking the connection sites")
    parser.add_argument("--template", default=None,
                        help="Labeled XYZ whose x/y site labels are copied to every conformer; "
                             "default <path>/conformers/label_template.xyz when it exists")
    parser.add_argument("--charge", type=int, default=0, help="Net charge for XYZ bond perception")
    parser.add_argument("--num_conformers", type=int, default=2000)
    parser.add_argument("--rms_thresh", type=float, default=0.3,
                        help="ETKDG pruning RMSD before optimization")
    parser.add_argument("--with_random", action="store_true", help="ETKDG random coordinates")
    parser.add_argument("--seed", type=int, default=1, help="ETKDG random seed (-1: random)")
    parser.add_argument("--ff", choices=list(FF_LABELS), default=DEFAULT_FF,
                        help=f"Force field: mmff94s, mmff (= MMFF94) or uff (default {DEFAULT_FF})")
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--extension", choices=["auto", "sites", "ends", "rg"], default="auto",
                        help="auto: x/y site distance when labels exist, otherwise ends")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Use this many equal layers (see --binning) instead of the default "
                             "--layer_edges 80; implied by --layer_edges/--layer_percentiles")
    parser.add_argument("--binning", choices=["uniform", "quantile"], default=None,
                        help="Equal layers: uniform = equal width, quantile = equal counts "
                             "(4 layers unless --n_layers is given)")
    boundaries = parser.add_mutually_exclusive_group()
    boundaries.add_argument("--layer_edges", type=float, nargs="+", default=None,
                            help="Inner layer boundaries as %% of the pool's extension range "
                                 "(0 = shortest, 100 = longest conformer); default 80 when no "
                                 "layering option is given")
    boundaries.add_argument("--layer_percentiles", type=float, nargs="+", default=None,
                            help="Inner layer boundaries as percentiles of the pool "
                                 "(share of conformers below each cut), e.g. 20 80")
    parser.add_argument("--energy_window", type=float, default=10.0,
                        help="kcal/mol above each layer's own minimum")
    parser.add_argument("--max_rel_energy", type=float, default=None,
                        help="Optional global cutoff (kcal/mol) applied before layering")
    parser.add_argument("--dedup_rmsd", type=float, default=0.25,
                        help="Heavy-atom RMSD (A) below which an optimized conformer is a duplicate")
    parser.add_argument("--cluster_rmsd", type=float, default=0.3,
                        help="Minimum heavy-atom RMSD (A) between any two picks, across layers")
    parser.add_argument("--keep_mirror", action="store_true",
                        help="Treat mirror-image conformers as different")
    parser.add_argument("--n_global", type=int, default=DEFAULT_N_GLOBAL,
                        help="Lowest-energy conformers of the whole pool to pick before the "
                             f"layers (default {DEFAULT_N_GLOBAL})")
    quota = parser.add_mutually_exclusive_group()
    quota.add_argument("--need_structure", type=int, default=None,
                       help="Layer picks split evenly over the layers")
    quota.add_argument("--layer_quota", type=int, nargs="+", default=None,
                       help="Per-layer counts, from folded to extended; default 0 2, which "
                            "fits the default two layers")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing conformers or a label template from an earlier run")
    args = parser.parse_args()

    if args.make_template:
        write_label_template(args.source, args.path, charge=args.charge, seed=args.seed,
                             overwrite=args.overwrite)
        return
    default_template = Path(args.path) / "conformers" / LABEL_TEMPLATE_NAME
    is_xyz_source = os.path.isfile(args.source) and args.source.lower().endswith(".xyz")
    if args.template is None and not is_xyz_source and default_template.exists():
        args.template = str(default_template)

    inner = args.layer_edges or args.layer_percentiles
    if inner is not None:
        if any(b <= a for a, b in zip(inner, inner[1:])):
            parser.error("layer boundaries must increase strictly")
        if min(inner) <= 0 or max(inner) >= 100:
            parser.error("layer boundaries must lie strictly between 0 and 100")
    try:
        (args.n_layers, args.binning, args.layer_edges, args.layer_percentiles,
         args.layer_quota, args.need_structure) = resolve_layering(
            args.n_layers, args.binning, args.layer_edges, args.layer_percentiles,
            args.layer_quota, args.need_structure)
    except ValueError as exc:
        parser.error(str(exc))
    if args.n_global < 0:
        parser.error("--n_global must not be negative")
    if args.cluster_rmsd <= args.dedup_rmsd:
        parser.error("--cluster_rmsd must be larger than --dedup_rmsd")

    stratified_conformer_search(
        args.source,
        args.path,
        template=args.template,
        charge=args.charge,
        num_conformers=args.num_conformers,
        rms_thresh=args.rms_thresh,
        with_random=args.with_random,
        seed=args.seed,
        ff=args.ff,
        max_iters=args.max_iters,
        extension=args.extension,
        n_layers=args.n_layers,
        binning=args.binning,
        layer_edges=args.layer_edges,
        layer_percentiles=args.layer_percentiles,
        energy_window=args.energy_window,
        max_rel_energy=args.max_rel_energy,
        dedup_rmsd=args.dedup_rmsd,
        cluster_rmsd=args.cluster_rmsd,
        keep_mirror=args.keep_mirror,
        need_structure=args.need_structure,
        layer_quota=args.layer_quota,
        n_global=args.n_global,
        plot=not args.no_plot,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
