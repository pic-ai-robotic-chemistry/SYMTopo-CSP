import copy
import fractions
import re

import numpy as np

from basic_function import chemical_knowledge, others, unit_cell_parser


class Atom:
    """
    Arguments:

    must have:
    element --

    two choose one: cart_xyz or frac_xyz
    cart_xyz --
    frac_xyz --

    Optional:
    force --
    atom_charge --
    atom_energy --
    atom_id --
    atom_descriptor --
    molecule --
    """

    def __init__(self, **kwargs):
        self.element = kwargs.get("element", "unknown")
        self.cart_xyz = kwargs.get("cart_xyz", "unknown")
        self.frac_xyz = kwargs.get("frac_xyz", "unknown")
        self.atom_id = kwargs.get("atom_id", "unknown")
        self.descriptor = kwargs.get("descriptor", "unknown")
        self.force = kwargs.get("force_xyz", "unknown")
        self.atom_charge = kwargs.get("atom_charge", "unknown")
        self.atom_energy = kwargs.get("atom_energy", "unknown")
        self.molecule = kwargs.get("molecule", "unknown")
        self.bonded_atom = kwargs.get("bonded_atom", [])
        self.comment = kwargs.get("comment", {})

        self.site_label = kwargs.get("site_label", "unknown")
        self.source_atom_id = kwargs.get("source_atom_id", self.atom_id)
        self.symm_op_id = kwargs.get("symm_op_id", 0)


class Crystal:
    """
    Arguments:

    choose one: cell_vect or cell_para
    cell_vect -- list of [[lattice_vect_a_1, lattice_vect_a_2, lattice_vect_a_3],
                          [lattice_vect_b_1, lattice_vect_b_2, lattice_vect_b_3],
                          [lattice_vect_c_1, lattice_vect_c_2, lattice_vect_c_3]]
    cell_para -- list of [[length_a, length_b, length_c],
                          [   alpha,     beta,    gamma]]

    atoms -- list of class Atom, e.g. [Atom1, Atom2, Atom3......AtomN]

    Optional:
    Energy --
    Comment --
    Descriptor --

    Auto calculate:
    Density --
    Volume --
    """

    def __init__(self, **kwargs):
        self.cell_vect = kwargs.get("cell_vect", "unknown")
        self.cell_para = kwargs.get("cell_para", "unknown")
        self.atoms = kwargs.get("atoms", "unknown")
        self.energy = kwargs.get("energy", "unknown")
        self.comment = kwargs.get("comment", "unknown")
        self.descriptor = kwargs.get("descriptor", "unknown")
        self.molecule_number = kwargs.get("molecule_number", "unknown")
        self.system_name = kwargs.get("system_name", "unknown")
        self.virial = kwargs.get("virial", "unknown")
        self.lattice_and_atom_complete()
        self.other_properties = {}
        self.SYMM = kwargs.get("SYMM", ["x,y,z"])
        self.space_group = kwargs.get("space_group", 1)

    def lattice_and_atom_complete(self):
        # deal with the lattice
        if self.cell_vect != "unknown" and self.cell_para != "unknown":
            for index, value in enumerate(
                np.array(unit_cell_parser.cell_vect_to_para(self.cell_vect)).flatten()
            ):
                assert (
                    np.array(self.cell_para).flatten()[index] - 0.001
                    <= value
                    <= np.array(self.cell_para).flatten()[index] + 0.001
                ), "cell_para and cell_vect is inconsistent"
        elif self.cell_vect != "unknown" and self.cell_para == "unknown":
            self.cell_para = unit_cell_parser.cell_vect_to_para(self.cell_vect)
        elif self.cell_vect == "unknown" and self.cell_para != "unknown":
            self.cell_vect = unit_cell_parser.cell_para_to_vect(self.cell_para)
        else:
            raise Exception("Error: Cell information are not given")

        # deal with the atoms
        if self.atoms == "unknown":
            print("Warning: No atoms in the crystal")
        else:
            for atom in self.atoms:
                if "unknown" not in atom.cart_xyz and "unknown" not in atom.frac_xyz:
                    pass
                elif "unknown" in atom.cart_xyz:
                    atom.cart_xyz = unit_cell_parser.atom_frac_to_cart_by_cell_vect(
                        atom.frac_xyz, self.cell_vect
                    )
                elif "unknown" in atom.frac_xyz:
                    atom.frac_xyz = unit_cell_parser.atom_cart_to_frac_by_cell_vect(
                        atom.cart_xyz, self.cell_vect
                    )
                else:
                    raise Exception("Atom position is unknown")

        # calculate the volume
        self.volume = unit_cell_parser.calculate_volume(self.cell_para)

        # calculate the density
        mass = 0
        for atom in self.atoms:
            mass += chemical_knowledge.element_masses[atom.element]
        self.density = mass / (self.volume * ((10**-8) ** 3)) / (6.022140857 * (10**23))

    def update_cart_by_frac(self):
        for atom in self.atoms:
            atom.cart_xyz = unit_cell_parser.atom_frac_to_cart_by_cell_vect(
                atom.frac_xyz, self.cell_vect
            )

    def move_atom_into_cell(self):
        # move all the atoms into cell, which means all the atom's frac parameter will between 0 to 1
        for atom in self.atoms:
            for i in range(0, 3):
                while atom.frac_xyz[i] > 1:
                    atom.frac_xyz[i] = atom.frac_xyz[i] - 1
                while atom.frac_xyz[i] < 0:
                    atom.frac_xyz[i] = atom.frac_xyz[i] + 1
            atom.cart_xyz = unit_cell_parser.atom_frac_to_cart_by_cell_vect(
                atom.frac_xyz, self.cell_vect
            )

    def make_p1(self):
        all_ele, all_frac = self.get_ele_and_frac()
        all_reflect_position = []
        all_matrix_M = []
        all_matrix_C = []
        for sym_opt in self.SYMM:
            sym_opt_ele = sym_opt.lower().replace(" ", "").split(",")
            matrix_M = np.zeros((3, 3))
            matrix_C = np.zeros((1, 3))
            for idx, word in enumerate(sym_opt_ele):
                sym_opt_ele_split = re.findall(r".*?([+-]*[xyz0-9\/\.]+)", word)
                for sym_opt_frag in sym_opt_ele_split:
                    if sym_opt_frag == "x" or sym_opt_frag == "+x":
                        matrix_M[0][idx] = 1
                    elif str(sym_opt_frag) == "-x":
                        matrix_M[0][idx] = -1
                    elif sym_opt_frag == "y" or sym_opt_frag == "+y":
                        matrix_M[1][idx] = 1
                    elif sym_opt_frag == "-y":
                        matrix_M[1][idx] = -1
                    elif sym_opt_frag == "z" or sym_opt_frag == "+z":
                        matrix_M[2][idx] = 1
                    elif sym_opt_frag == "-z":
                        matrix_M[2][idx] = -1
                    elif others.is_number(sym_opt_frag) is True:
                        matrix_C[0][idx] = float(fractions.Fraction(sym_opt_frag))
                    else:
                        raise Exception("wrong sym opt of" + sym_opt_frag)

            all_matrix_M.append(matrix_M)
            all_matrix_C.append(matrix_C)

        new_atoms = []
        idx = 0
        for symm_idx, (matrix_M, matrix_C) in enumerate(
            zip(all_matrix_M, all_matrix_C)
        ):
            new_positions = (np.dot(np.array([all_frac]), matrix_M) + matrix_C).reshape(
                -1, 3
            )
            for i, atom in enumerate(self.atoms):
                frac_xyz = new_positions[i]
                base_label = others.normalize_site_label(
                    atom.site_label,
                    atom.element,
                    atom.atom_id if atom.atom_id != "unknown" else idx,
                )

                new_atoms.append(
                    Atom(
                        element=atom.element,
                        frac_xyz=frac_xyz,
                        atom_id=idx,
                        source_atom_id=atom.source_atom_id,
                        symm_op_id=symm_idx,
                        site_label=others.make_symm_copy_label(
                            base_label, symm_idx, ensure_unique=True
                        ),
                        comment=copy.deepcopy(atom.comment),
                    )
                )
                idx += 1

        self.SYMM = "[x,y,z]"
        self.space_group = 1
        self.atoms = new_atoms
        self.update_cart_by_frac()

    def get_ele_and_cart(self):
        all_carts = np.array([atom.cart_xyz for atom in self.atoms])
        all_ele = [atom.element for atom in self.atoms]
        return all_ele, all_carts

    def get_ele_and_frac(self):
        all_frac = np.array([atom.frac_xyz for atom in self.atoms])
        all_ele = [atom.element for atom in self.atoms]
        return all_ele, all_frac


class Molecule:

    def __init__(self, **kwargs):
        self.atoms = kwargs.get("atoms", "unknown")
        self.energy = kwargs.get("energy", "unknown")
        self.comment = kwargs.get("comment", "unknown")
        self.descriptor = kwargs.get("descriptor", "unknown")
        self.name = kwargs.get("name", "unknown")
        self.system_name = kwargs.get("system_name", "unknown")

        if self.atoms == "unknown":
            print("Warning: No atoms in the molecule")

    def get_ele_and_cart(self):
        all_carts = np.array([atom.cart_xyz for atom in self.atoms])
        all_ele = [atom.element for atom in self.atoms]
        return all_ele, all_carts

    def put_ele_cart_back(self, all_ele, all_carts):
        for idx, atom in enumerate(self.atoms):
            atom.element = all_ele[idx]
            atom.cart_xyz = all_carts[idx]

    def get_mass(self):
        mass = 0
        for atom in self.atoms:
            mass += chemical_knowledge.element_masses[atom.element]
        return mass

    def get_center_of_mass(self):
        mass = self.get_mass()
        all_ele, all_carts = self.get_ele_and_cart()
        masses = np.array([chemical_knowledge.element_masses[x] for x in all_ele])
        center_of_mass = np.sum(
            all_carts * masses.reshape(len(masses), 1) / mass, axis=0
        )
        return center_of_mass
