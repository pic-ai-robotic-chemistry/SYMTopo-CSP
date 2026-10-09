"""Convenience wrappers for serial and parallel crystal generation."""

import concurrent.futures
import os
import re
from contextlib import redirect_stdout

from tqdm import tqdm

from basic_function import CSP_generator_normal, format_parser


def process_crystal(
    seed: int,
    sg: int,
    molecules: list,
    output_path: str | os.PathLike,
    add_name: str,
) -> bool:
    """Generate and write one trial crystal for a space group and Sobol seed."""
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        generator = CSP_generator_normal.CrystalGenerator(molecules, space_group=sg)
        new_crystal = generator.generate(seed=seed)
    if new_crystal is not None:
        cif_out = format_parser.write_cif_file(new_crystal)
        with open(
            f"{output_path}/structures_ori/{add_name}_{sg}_{seed}_z{len(molecules)}.cif",
            "w",
        ) as target:
            target.writelines(cif_out)
        return True
    return False


def CSP_generater_parallel(
    molecules: list,
    output_path: str | os.PathLike,
    need_structure: int | list[int] = 100,
    space_group_list: list[int] = [1],
    max_workers: int = 8,
    add_name: str = "",
    start_seed: int | str = 1,
) -> None:
    space_groups = space_group_list
    if isinstance(need_structure, int):
        accept_count_list = [need_structure] * len(space_groups)
    elif isinstance(need_structure, list):
        accept_count_list = need_structure
    else:
        raise TypeError("need_structure must be either an integer or a list")

    if len(space_groups) != len(accept_count_list):
        raise ValueError(
            "Length of space_group_list and need_structure must be the same"
        )
    tar_folder = "{}/structures_ori".format(output_path)
    os.makedirs(tar_folder, exist_ok=True)

    existing_files = [
        f for f in os.listdir(tar_folder) if os.path.isfile(os.path.join(tar_folder, f))
    ]
    progress = tqdm(
        total=sum(accept_count_list), desc="Generating structures", unit="structure"
    )
    for sg, accept_count in zip(space_groups, accept_count_list):
        accept = 0
        if start_seed == "auto" and len(existing_files) > 0:
            pattern = (
                rf"{re.escape(add_name)}_{sg}_(?P<seed>-?\d+)_z{len(molecules)}\.cif"
            )
            seed = (
                max(
                    (
                        int(m["seed"])
                        for f in existing_files
                        if (m := re.fullmatch(pattern, f))
                    ),
                    default=0,
                )
                + 1
            )
        elif isinstance(start_seed, int):
            seed = start_seed
        else:
            seed = 1

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = {}
            while accept < accept_count:
                # Keep at most one task per worker in flight.
                while (
                    len(futures) < max_workers and accept + len(futures) < accept_count
                ):
                    future = executor.submit(
                        process_crystal, seed, sg, molecules, output_path, add_name
                    )
                    futures[future] = seed
                    seed += 1

                # Process tasks as soon as any worker completes.
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    if future.result():
                        accept += 1
                        progress.update(1)
                    # Remove every completed task, regardless of its result.
                    del futures[future]

                # Cancel outstanding tasks once the requested count is reached.
                if accept >= accept_count:
                    for future in futures:
                        future.cancel()
                    break
    progress.close()
