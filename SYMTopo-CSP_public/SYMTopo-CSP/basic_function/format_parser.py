import copy
import re

from basic_function import chemical_knowledge, data_classes, others


def get_cif_atom_label(atom, fallback_idx):
    return others.normalize_site_label(
        atom.site_label,
        atom.element,
        atom.atom_id if atom.atom_id != "unknown" else fallback_idx,
    )


def parse_xyz_atom_label(raw_note, element, atom_id):
    raw_note = str(raw_note).strip()
    if raw_note == "":
        return others.normalize_site_label(None, element, atom_id)

    # Preferred form: an explicit ``label=...`` token.
    m = re.search(r"label\s*=\s*([^\s]+)", raw_note)
    if m:
        return others.normalize_site_label(m.group(1), element, atom_id)

    # Otherwise, treat the complete XYZ annotation as the site label.
    return others.normalize_site_label(raw_note, element, atom_id)


def read_xyz_file(file_path):
    input_file = open(file_path, "r")
    lines = input_file.readlines()
    number_of_atoms = int(lines[0])
    name = str(lines[1][:-1])
    atoms = []
    for index, line in enumerate(lines):
        split_line = list(filter(lambda x: x != "", re.split("\\s+", line)))
        if (
            len(split_line) >= 4
            and others.is_number(split_line[1])
            and others.is_number(split_line[2])
            and others.is_number(split_line[3])
        ):
            extra_tokens = split_line[4:]
            raw_note = " ".join(extra_tokens)
            label = parse_xyz_atom_label(raw_note, split_line[0], index - 2)
            atoms.append(
                data_classes.Atom(
                    element=split_line[0],
                    cart_xyz=[
                        float(split_line[1]),
                        float(split_line[2]),
                        float(split_line[3]),
                    ],
                    atom_id=index - 2,
                    source_atom_id=index - 2,
                    site_label=label,
                    comment={"raw_xyz_note": raw_note},
                )
            )

    if number_of_atoms != len(atoms):
        print("Warning! The length of atoms don't match the number of atoms given")

    molecule = data_classes.Molecule(atoms=atoms, name=name, system_name=name)

    return molecule


def write_cif_file(crystal, sym=False, name="zcx"):
    """
    Accept crystal class, give the cif file out
    :param crystal: crystal class
    :param coordinates: frac or cart
    :param sym: False:give all atoms out; True:with symmetry TODO
    :param name: file name
    :param name: file name
    :return: cif_out
    cif file in list format should be print using the following function:
                                        target=open("D:\\zcx.cif",'w')
                                        target.writelines(cif_out)
                                        target.close()
    """
    if crystal.system_name != "unknown":
        name = crystal.system_name
    cif_file = []

    cif_file.append("data_" + str(name) + "\n")
    if sym == False:
        if crystal.space_group == 1:
            crystal_temp = crystal
        else:
            crystal_temp = copy.deepcopy(crystal)
            crystal_temp.make_p1()
        cif_file.append("_symmetry_space_group_name_H-M    'P1'" + "\n")
        cif_file.append("_symmetry_Int_Tables_number       1" + "\n")

        cif_file.append("loop_" + "\n")
        cif_file.append("_symmetry_equiv_pos_site_id" + "\n")
        cif_file.append("_symmetry_equiv_pos_as_xyz" + "\n")
        cif_file.append("1 x,y,z" + "\n")
        cif_file.append(
            "_cell_length_a                   "
            + str(crystal_temp.cell_para[0][0])
            + "\n"
        )
        cif_file.append(
            "_cell_length_b                   "
            + str(crystal_temp.cell_para[0][1])
            + "\n"
        )
        cif_file.append(
            "_cell_length_c                   "
            + str(crystal_temp.cell_para[0][2])
            + "\n"
        )
        cif_file.append(
            "_cell_angle_alpha                "
            + str(crystal_temp.cell_para[1][0])
            + "\n"
        )
        cif_file.append(
            "_cell_angle_beta                 "
            + str(crystal_temp.cell_para[1][1])
            + "\n"
        )
        cif_file.append(
            "_cell_angle_gamma                "
            + str(crystal_temp.cell_para[1][2])
            + "\n"
        )

        cif_file.append("loop_" + "\n")
        cif_file.append("_atom_site_label" + "\n")
        cif_file.append("_atom_site_type_symbol" + "\n")
        cif_file.append("_atom_site_fract_x" + "\n")
        cif_file.append("_atom_site_fract_y" + "\n")
        cif_file.append("_atom_site_fract_z" + "\n")
        for i in range(0, len(crystal_temp.atoms)):
            label = get_cif_atom_label(crystal_temp.atoms[i], i)
            cif_file.append(
                "{:12}  {:4} {:16.8f} {:16.8f} {:16.8f}\n".format(
                    label,
                    crystal_temp.atoms[i].element,
                    crystal_temp.atoms[i].frac_xyz[0],
                    crystal_temp.atoms[i].frac_xyz[1],
                    crystal_temp.atoms[i].frac_xyz[2],
                )
            )

        return cif_file

    elif sym == True:
        cif_file.append(
            "_symmetry_space_group_name_H-M    '{}'".format(
                chemical_knowledge.space_group[crystal.space_group][1]
            )
            + "\n"
        )
        cif_file.append(
            "_symmetry_Int_Tables_number       {}".format(crystal.space_group) + "\n"
        )

        cif_file.append("loop_" + "\n")
        cif_file.append("_symmetry_equiv_pos_site_id" + "\n")
        cif_file.append("_symmetry_equiv_pos_as_xyz" + "\n")
        for idx, SYMM in enumerate(crystal.SYMM):
            cif_file.append("{} {}".format(idx + 1, SYMM) + "\n")
        cif_file.append(
            "_cell_length_a                   " + str(crystal.cell_para[0][0]) + "\n"
        )
        cif_file.append(
            "_cell_length_b                   " + str(crystal.cell_para[0][1]) + "\n"
        )
        cif_file.append(
            "_cell_length_c                   " + str(crystal.cell_para[0][2]) + "\n"
        )
        cif_file.append(
            "_cell_angle_alpha                " + str(crystal.cell_para[1][0]) + "\n"
        )
        cif_file.append(
            "_cell_angle_beta                 " + str(crystal.cell_para[1][1]) + "\n"
        )
        cif_file.append(
            "_cell_angle_gamma                " + str(crystal.cell_para[1][2]) + "\n"
        )

        cif_file.append("loop_" + "\n")
        cif_file.append("_atom_site_label" + "\n")
        cif_file.append("_atom_site_type_symbol" + "\n")
        cif_file.append("_atom_site_fract_x" + "\n")
        cif_file.append("_atom_site_fract_y" + "\n")
        cif_file.append("_atom_site_fract_z" + "\n")
        for i in range(0, len(crystal.atoms)):
            label = get_cif_atom_label(crystal_temp.atoms[i], i)
            cif_file.append(
                "{:12}  {:4} {:16.8f} {:16.8f} {:16.8f}\n".format(
                    label,
                    crystal_temp.atoms[i].element,
                    crystal_temp.atoms[i].frac_xyz[0],
                    crystal_temp.atoms[i].frac_xyz[1],
                    crystal_temp.atoms[i].frac_xyz[2],
                )
            )

        return cif_file
