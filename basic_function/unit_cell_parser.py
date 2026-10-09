import numpy as np


def cell_para_to_vect(cell_para, check=False):
    if check:
        shape_check = np.array(cell_para)
        assert shape_check.shape == (2, 3), "Wrong input cell para"

    length_a = cell_para[0][0]
    length_b = cell_para[0][1]
    length_c = cell_para[0][2]
    angle_alpha = cell_para[1][0]
    angle_beta = cell_para[1][1]
    angle_gamma = cell_para[1][2]
    rad_alpha = np.pi * angle_alpha / 180
    rad_beta = np.pi * angle_beta / 180
    rad_gamma = np.pi * angle_gamma / 180

    cell_vect = [
        [length_a, 0, 0],
        [length_b * np.cos(rad_gamma), length_b * np.sin(rad_gamma), 0],
        [
            length_c * np.cos(rad_beta),
            length_c
            * (
                (np.cos(rad_alpha) - np.cos(rad_beta) * np.cos(rad_gamma))
                / np.sin(rad_gamma)
            ),
            length_c
            * (
                np.sqrt(
                    1
                    + 2 * np.cos(rad_alpha) * np.cos(rad_beta) * np.cos(rad_gamma)
                    - np.square(np.cos(rad_alpha))
                    - np.square(np.cos(rad_beta))
                    - np.square(np.cos(rad_gamma))
                )
            )
            / np.sin(rad_gamma),
        ],
    ]
    return cell_vect


def cell_vect_to_para(cell_vect, check=False):
    if check:
        shape_check = np.array(cell_vect)
        assert shape_check.shape == (3, 3), "Wrong input cell vect"

    vector_a = np.array([cell_vect[0][0], cell_vect[0][1], cell_vect[0][2]])
    vector_b = np.array([cell_vect[1][0], cell_vect[1][1], cell_vect[1][2]])
    vector_c = np.array([cell_vect[2][0], cell_vect[2][1], cell_vect[2][2]])
    length_a = (
        cell_vect[0][0] ** 2 + cell_vect[0][1] ** 2 + cell_vect[0][2] ** 2
    ) ** 0.5
    length_b = (
        cell_vect[1][0] ** 2 + cell_vect[1][1] ** 2 + cell_vect[1][2] ** 2
    ) ** 0.5
    length_c = (
        cell_vect[2][0] ** 2 + cell_vect[2][1] ** 2 + cell_vect[2][2] ** 2
    ) ** 0.5
    rad_alpha = np.arccos(
        np.dot(vector_b, vector_c)
        / (np.linalg.norm(vector_b) * np.linalg.norm(vector_c))
    )
    rad_beta = np.arccos(
        np.dot(vector_a, vector_c)
        / (np.linalg.norm(vector_a) * np.linalg.norm(vector_c))
    )
    rad_gamma = np.arccos(
        np.dot(vector_a, vector_b)
        / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    )
    alpha = rad_alpha / np.pi * 180
    beta = rad_beta / np.pi * 180
    gamma = rad_gamma / np.pi * 180
    cell_para = [[length_a, length_b, length_c], [alpha, beta, gamma]]
    return cell_para


def atom_frac_to_cart_by_cell_vect(atom_frac, cell_vect, check=False):
    if check:
        shape_check = np.array(cell_vect)
        assert shape_check.shape == (3, 3), "Wrong input cell vect"
        assert len(atom_frac) == 3, "Wrong input atom frac"

    x_cart = (
        atom_frac[0] * cell_vect[0][0]
        + atom_frac[1] * cell_vect[1][0]
        + atom_frac[2] * cell_vect[2][0]
    )
    y_cart = (
        atom_frac[0] * cell_vect[0][1]
        + atom_frac[1] * cell_vect[1][1]
        + atom_frac[2] * cell_vect[2][1]
    )
    z_cart = (
        atom_frac[0] * cell_vect[0][2]
        + atom_frac[1] * cell_vect[1][2]
        + atom_frac[2] * cell_vect[2][2]
    )
    atom_cart = [x_cart, y_cart, z_cart]
    return atom_cart


def atom_cart_to_frac_by_cell_vect(atom_cart, cell_vect, check=False):
    if check:
        shape_check = np.array(cell_vect)
        assert shape_check.shape == (3, 3), "Wrong input cell vect"
        assert len(atom_cart) == 3, "Wrong input atom cart"

    lattice = np.array(cell_vect)
    lattice_inv = np.linalg.pinv(lattice)
    atom_cart_np = np.array(atom_cart)
    atom_frac_np = np.matmul(atom_cart_np, lattice_inv)
    atom_frac = atom_frac_np.tolist()
    return atom_frac


def calculate_volume(cell_info):
    if len(np.array(cell_info).flatten()) == 6:
        cell_para = cell_info
    elif len(np.array(cell_info).flatten()) == 9:
        cell_para = cell_vect_to_para(cell_info)
    else:
        raise Exception("Can't understand input during calculate_volume")

    a = cell_para[0][0]
    b = cell_para[0][1]
    c = cell_para[0][2]
    angle_alpha = cell_para[1][0]
    angle_beta = cell_para[1][1]
    angle_gamma = cell_para[1][2]
    rad_alpha = np.pi * angle_alpha / 180
    rad_beta = np.pi * angle_beta / 180
    rad_gamma = np.pi * angle_gamma / 180

    volume = (
        a
        * b
        * c
        * (
            (
                (
                    1
                    - np.cos(rad_alpha) ** 2
                    - np.cos(rad_beta) ** 2
                    - np.cos(rad_gamma) ** 2
                )
                + 2 * (np.cos(rad_alpha) * np.cos(rad_beta) * np.cos(rad_gamma))
            )
            ** 0.5
        )
    )

    return volume
