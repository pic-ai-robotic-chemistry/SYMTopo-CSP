import copy

from basic_function import data_classes, others


def super_cell(crystal, cell_range=[[-1, 1], [-1, 1], [-1, 1]]):

    neighbor_cell = []
    all_atoms_new = []
    atoms = crystal.atoms

    new_lattice = [
        [a * (cell_range[0][1] - cell_range[0][0] + 1) for a in crystal.cell_vect[0]],
        [b * (cell_range[1][1] - cell_range[1][0] + 1) for b in crystal.cell_vect[1]],
        [c * (cell_range[2][1] - cell_range[2][0] + 1) for c in crystal.cell_vect[2]],
    ]

    for h in range(cell_range[0][0], cell_range[0][1] + 1):
        for k in range(cell_range[1][0], cell_range[1][1] + 1):
            for l in range(cell_range[2][0], cell_range[2][1] + 1):
                neighbor_cell.append([h, k, l])

    for i in range(0, len(atoms)):
        for cell in neighbor_cell:
            atoms_new_element = atoms[i].element

            atoms_new_frac_xyz = [
                (atoms[i].frac_xyz[0] + cell[0])
                / (cell_range[0][1] - cell_range[0][0] + 1),
                (atoms[i].frac_xyz[1] + cell[1])
                / (cell_range[1][1] - cell_range[1][0] + 1),
                (atoms[i].frac_xyz[2] + cell[2])
                / (cell_range[2][1] - cell_range[2][0] + 1),
            ]

            src_atom = atoms[i]
            cell_suffix = f"_sc_{cell[0]}_{cell[1]}_{cell[2]}"

            all_atoms_new.append(
                data_classes.Atom(
                    element=atoms_new_element,
                    frac_xyz=[
                        atoms_new_frac_xyz[0],
                        atoms_new_frac_xyz[1],
                        atoms_new_frac_xyz[2],
                    ],
                    atom_id=len(all_atoms_new),
                    source_atom_id=src_atom.source_atom_id,
                    symm_op_id=src_atom.symm_op_id,
                    site_label=others.normalize_site_label(
                        (
                            src_atom.site_label + cell_suffix
                            if src_atom.site_label != "unknown"
                            else None
                        ),
                        src_atom.element,
                        src_atom.atom_id if src_atom.atom_id != "unknown" else i,
                    ),
                    comment=copy.deepcopy(src_atom.comment),
                )
            )

    if crystal.energy != "unknown":
        new_energy = (
            crystal.energy
            * (cell_range[0][1] - cell_range[0][0] + 1)
            * (cell_range[1][1] - cell_range[1][0] + 1)
            * (cell_range[2][1] - cell_range[2][0] + 1)
        )
    else:
        new_energy = "unknown"

    new_crystal = data_classes.Crystal(
        cell_vect=new_lattice, energy=new_energy, atoms=all_atoms_new
    )

    return new_crystal
