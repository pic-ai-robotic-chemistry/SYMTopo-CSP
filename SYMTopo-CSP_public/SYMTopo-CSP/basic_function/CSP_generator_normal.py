"""Symmetry-guided quasi-random generator for periodic crystal candidates."""

import itertools

import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.stats import qmc

from basic_function import (
    CSP_function,
    chemical_knowledge,
    data_classes,
    operation_new,
    others,
)


class CrystalGenerator:
    """Generate collision-free asymmetric units under a selected space group."""

    def __init__(
        self,
        molecules: list[data_classes.Molecule],
        space_group: int = 1,
        angles: tuple[float, float] = (45.0, 135.0),
    ) -> None:
        self.angles = angles
        self.space_group = space_group
        self.SYMM = chemical_knowledge.space_group[space_group][0]
        self.point_group = chemical_knowledge.space_group[space_group][2]
        self.molecules = molecules
        self.n_molecule = len(self.SYMM) * len(self.molecules)
        self.n_atoms = []
        self.search_dimension = "unknown"
        self.search_dimension_shape = "unknown"
        self.max_vdw = "unknown"
        self.find_max_vdw()
        self.find_search_dimensions()
        self.find_n_atoms()
        self.super_cell_code_frac = np.array(
            sorted(
                list(itertools.product(np.arange(-2, 3), repeat=3)),
                key=lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2,
            )
        )[:]
        self.orient_all_molecules()
        self.estimate_volume_function()

    def find_n_atoms(self):
        self.n_atoms = []
        for molecule in self.molecules:
            self.n_atoms.append(len(molecule.atoms))

    def orient_all_molecules(self):
        """Orient the molecules to an axis.
        http://sobereva.com/426"""

        for idx, molecule in enumerate(self.molecules):
            if len(molecule.atoms) == 1:
                continue
            else:
                self.molecules[idx] = operation_new.orient_molecule(molecule)

    def find_max_vdw(self):
        vdw_max = 0
        for molecule in self.molecules:
            all_ele, all_cart = molecule.get_ele_and_cart()
            for ele in set(all_ele):
                vdw_max = max(vdw_max, chemical_knowledge.element_vdw_radii[ele])
        self.max_vdw = vdw_max

    def find_search_dimensions(self):
        dim_cell_length = 3
        dim_cell_angle = 3
        dim_move_part = 3 * len(self.molecules)
        dim_rotate_part = 3 * len(self.molecules)
        dimension = dim_cell_length + dim_cell_angle + dim_move_part + dim_rotate_part
        self.search_dimensions = dimension
        self.search_dimension_shape = [
            dim_cell_length,
            dim_cell_angle,
            dim_move_part,
            dim_rotate_part,
        ]

    def random_to_angle(self, value: float) -> float:
        min_angle = self.angles[0]
        max_angle = self.angles[1]
        angle_range = max_angle - min_angle
        a = np.arcsin(2 * value - 1.0) / np.pi
        return (0.5 + a) * angle_range + min_angle

    def cell_angle(self, values):
        angle_candidate = []
        for value in values:
            angle_candidate.append(self.random_to_angle(value))
        if self.point_group == "Triclinic":
            return angle_candidate[0], angle_candidate[1], angle_candidate[2]
        elif self.point_group == "Monoclinic":
            return 90, angle_candidate[1], 90
        elif self.point_group in ["Orthorhombic", "Tetragonal", "Cubic"]:
            return 90, 90, 90
        elif self.point_group == "Hexagonal":
            return 90, 90, 120
        elif self.point_group == "Trigonal":
            return angle_candidate[0], angle_candidate[0], angle_candidate[0]

    def estimate_volume_function(self):
        total_volume = 0
        for molecule in self.molecules:
            all_ele, all_cart = molecule.get_ele_and_cart()
            vdws = np.array([chemical_knowledge.element_vdw_radii[x] for x in all_ele])
            volumes = (4 / 3) * np.pi * vdws**3
            total_volume += np.sum(volumes)
        self.estimate_volume = total_volume

    def cell_length(self, values, cell_angles, new_all_cart):
        s_mins = np.zeros(3)
        for all_cart in new_all_cart:
            i = all_cart @ operation_new.c2f_matrix([[1, 1, 1], cell_angles])
            diff = np.max(i, axis=0) - np.min(i, axis=0)
            s_mins = np.maximum(s_mins, diff)
        s_mins += self.max_vdw * 2
        if self.point_group in ["Triclinic", "Monoclinic", "Orthorhombic"]:
            a = s_mins[0] + values[0] * (self.n_molecule * s_mins[0])
            b = s_mins[1] + values[1] * (self.n_molecule * s_mins[1])
            c = s_mins[2] + values[2] * (self.n_molecule * s_mins[2])
            return a, b, c
        elif self.point_group in ["Tetragonal", "Hexagonal"]:
            a = s_mins[0] + values[0] * (self.n_molecule * s_mins[0])
            c = s_mins[2] + values[2] * (self.n_molecule * s_mins[2])
            return a, a, c
        elif self.point_group in ["Trigonal", "Cubic"]:
            a = s_mins[0] + values[0] * (self.n_molecule * s_mins[0])
            return a, a, a

    def generate(
        self,
        seed: int | str = "unknown",
        test: bool = False,
        densely_pack_method: bool = False,
        frame_tolerance: float = 1.5,
    ) -> data_classes.Crystal | None:
        print(seed)
        if seed == "unknown":
            raise Exception("error! no seed is given")
        sobol_gen = qmc.Sobol(d=self.search_dimensions, seed=seed)
        # Sample cell angles, lengths, molecular translations, and rotations.
        samples = sobol_gen.random(n=1)
        samples = samples.flatten()
        cell_angle_seed = samples[0:3]
        cell_length_seed = samples[3:6]
        move_part_seed = samples[6 : 6 + len(self.molecules) * 3]
        rotate_part_seed = samples[6 + len(self.molecules) * 3 :]

        alpha, beta, gamma = self.cell_angle(cell_angle_seed)

        # Compute the angular contribution to the parallelepiped volume.
        ca = np.cos(alpha / 180 * np.pi)
        cb = np.cos(beta / 180 * np.pi)
        cg = np.cos(gamma / 180 * np.pi)
        sg = np.sin(gamma / 180 * np.pi)
        x = 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg
        if x <= 0:
            print("fail because of length and angle could not build a unitcell")
            return None
        sqrt_x = np.sqrt(x)

        new_all_cart = []
        new_all_ele = []
        new_all_labels = []
        new_all_source_ids = []
        rotate_part = rotate_part_seed.reshape(-1, 3)

        for r_part, molecule in zip(rotate_part, self.molecules):
            all_ele, all_cart = molecule.get_ele_and_cart()
            new_cart = all_cart @ operation_new.get_rotate_matrix(r_part)

            new_all_cart.append(new_cart)
            new_all_ele.append(all_ele)
            new_all_labels.append(
                [
                    others.normalize_site_label(
                        atom.site_label, atom.element, atom.atom_id
                    )
                    for atom in molecule.atoms
                ]
            )
            new_all_source_ids.append(
                [
                    (
                        atom.source_atom_id
                        if atom.source_atom_id != "unknown"
                        else atom.atom_id
                    )
                    for atom in molecule.atoms
                ]
            )

        a, b, c = self.cell_length(cell_length_seed, [alpha, beta, gamma], new_all_cart)

        crystal_volume = a * b * c * sqrt_x
        if densely_pack_method == True:
            if crystal_volume > self.estimate_volume * 20:
                c = self.estimate_volume * 20 / (a * b * sqrt_x)

        # Contract the unit cell until further contraction causes a collision.
        cell_limit = [False, False, False]

        cycle_no = 0
        while True:
            if a < 0 or b < 0 or c < 0:
                print(
                    "BUG:stoped because of a or b or c < 0, sg = {} seed = {}".format(
                        self.space_group, seed
                    )
                )
                return None
            if cycle_no > 1000:
                print("stoped because of max cycle reached, seed = {}".format(seed))
                return None
            cell_para = [[a, b, c], [alpha, beta, gamma]]
            f2c = operation_new.f2c_matrix(cell_para)
            c2f = operation_new.c2f_matrix(cell_para)

            asym_frac = []
            asym_ele = []
            asym_label = []
            asym_source_id = []
            sc_cart = np.empty((0, 3))
            sc_ele = np.empty((0, 1))

            super_cell_code_cart = self.super_cell_code_frac @ f2c

            for id_mol, all_cart in enumerate(new_all_cart):
                sc_cart_temp = []
                all_frac = (
                    all_cart @ c2f + move_part_seed[id_mol * 3 : (id_mol + 1) * 3]
                )
                asym_frac.append(all_frac)
                asym_ele.append(new_all_ele[id_mol])
                asym_label.append(new_all_labels[id_mol])
                asym_source_id.append(new_all_source_ids[id_mol])

                all_cart_crystal = operation_new.apply_SYMM(all_frac, self.SYMM) @ f2c
                all_ele_crystal = [new_all_ele[id_mol]] * len(self.SYMM)
                centroid_1 = np.mean(all_frac, axis=0)
                centroids_frac = operation_new.apply_SYMM(centroid_1, self.SYMM)

                for j, cent in enumerate(centroids_frac):
                    move_temp = (np.mod(cent, 1) - cent) @ f2c
                    all_cart_crystal[j] = all_cart_crystal[j] + move_temp

                for code in super_cell_code_cart:
                    sc_cart_temp.append(
                        np.array(all_cart_crystal + code).reshape(-1, 3)
                    )
                sc_cart = np.vstack((sc_cart, np.array(sc_cart_temp).reshape(-1, 3)))
                sc_ele_temp = np.array(
                    all_ele_crystal * len(super_cell_code_cart)
                ).reshape(-1, 1)
                sc_ele = np.vstack((sc_ele, sc_ele_temp))

            if test:
                print("cycle no", cycle_no)
            crush = self.collision_test(sc_ele, sc_cart)
            if crush == True and cycle_no == 0:
                print("fail because the crush test not path at the first step")
                return None

            elif cycle_no != 0 and cell_limit == [True, True, True]:
                concatenated_all_ele = np.concatenate(asym_ele, axis=0)
                concatenated_all_frac = np.concatenate(asym_frac, axis=0)
                concatenated_all_label = np.concatenate(asym_label, axis=0)
                concatenated_all_source_id = np.concatenate(asym_source_id, axis=0)
                all_ele = concatenated_all_ele.reshape(-1, 1)
                all_frac = concatenated_all_frac.reshape(-1, 3)
                all_label = np.array(concatenated_all_label).reshape(-1)
                all_source_id = np.array(concatenated_all_source_id).reshape(-1)
                atoms = []
                for idx, (ele, frac, label, src_id) in enumerate(
                    zip(all_ele, all_frac, all_label, all_source_id)
                ):
                    atoms.append(
                        data_classes.Atom(
                            element=ele.item(),
                            frac_xyz=frac,
                            atom_id=idx,
                            source_atom_id=int(src_id),
                            symm_op_id=0,
                            site_label=others.normalize_site_label(
                                label, ele.item(), idx
                            ),
                        )
                    )
                generated_crystal = data_classes.Crystal(
                    cell_para=[[a, b, c], [alpha, beta, gamma]],
                    atoms=atoms,
                    comment=seed,
                    system_name=seed,
                    space_group=self.space_group,
                    SYMM=self.SYMM,
                )

                if (
                    CSP_function.detect_is_frame_vdw_new(
                        generated_crystal, tolerance=frame_tolerance
                    )
                    == False
                ):
                    print("fail because it is a slab")
                    return None
                return generated_crystal

            elif crush == True and cycle_no != 0 and cell_limit != [True, True, True]:
                a = old_a
                b = old_b
                c = old_c
                for temp_change in last_change:
                    cell_limit[temp_change] = True
            elif crush == False:
                old_a = a
                old_b = b
                old_c = c
                a, b, c, last_change = self.optimise_abc(a, b, c, cell_limit)

            cycle_no += 1

    def collision_test(self, sc_ele, sc_cart):
        sc_ele_temp = sc_ele
        sc_cart_temp = sc_cart
        sc_vdws = [chemical_knowledge.element_vdw_radii[x.item()] for x in sc_ele_temp]

        start = 0
        for i in range(0, len(self.molecules)):
            end = start + self.n_atoms[i]
            asym_pos = sc_cart_temp[start:end]
            asym_vdws = sc_vdws[start:end]
            neighbors = sc_cart_temp[end:]
            neigh_vdws = np.array(sc_vdws[end:])
            start += self.n_atoms[i] * 125 * len(self.SYMM)

            mol_min = np.min(asym_pos, axis=0) - self.max_vdw * 2 * 0.9
            mol_max = np.max(asym_pos, axis=0) + self.max_vdw * 2 * 0.9
            box_idxs = np.all(
                np.logical_and(mol_min < neighbors, neighbors < mol_max), axis=1
            )
            if not np.any(box_idxs):
                continue
            neighbors = neighbors[box_idxs]
            neigh_vdws = neigh_vdws[box_idxs]

            tree_1 = KDTree(asym_pos, compact_nodes=False, balanced_tree=False)
            tree_2 = KDTree(neighbors, compact_nodes=False, balanced_tree=False)

            contact = tree_1.query_ball_tree(tree_2, self.max_vdw * 2 * 0.9)
            for j, idxs in enumerate(contact):
                if len(idxs) == 0:
                    continue
                # Compute batched Euclidean distances without temporary norms.
                diff = asym_pos[j] - neighbors[idxs]
                dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
                sum_radii = (asym_vdws[j] + neigh_vdws[idxs]) * 0.9
                if np.any(dist < sum_radii):
                    return True
        return False

    def optimise_abc(self, a, b, c, cell_limit):
        if self.point_group in ["Triclinic", "Monoclinic", "Orthorhombic"]:
            lengths = [item for item, keep in zip([a, b, c], cell_limit) if not keep]
            if lengths == []:
                return a, b, c, 3
            max_length = max(lengths)
            if a == max_length:
                a -= 1
                last_change = [0]
            elif b == max_length:
                b -= 1
                last_change = [1]
            elif c == max_length:
                c -= 1
                last_change = [2]

        elif self.point_group in ["Tetragonal", "Hexagonal"]:
            lengths = [item for item, keep in zip([a, b, c], cell_limit) if not keep]
            if lengths == []:
                return a, b, c, 3
            max_length = max(lengths)
            if a == max_length or b == max_length:
                a -= 1
                b -= 1
                last_change = [0, 1]
            elif c == max_length:
                c -= 1
                last_change = [2]

        elif self.point_group in ["Trigonal", "Cubic"]:
            lengths = [item for item, keep in zip([a, b, c], cell_limit) if not keep]
            if lengths == []:
                return a, b, c, 3
            max_length = max(lengths)
            if a == max_length or b == max_length or c == max_length:
                a -= 1
                b -= 1
                c -= 1
                last_change = [0, 1, 2]

        return a, b, c, last_change
