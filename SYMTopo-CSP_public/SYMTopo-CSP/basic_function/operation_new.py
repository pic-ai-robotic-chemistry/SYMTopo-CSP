import fractions
import re

import numpy as np

from basic_function import chemical_knowledge, others


def orient_molecule(molecule):
    "using Moment of Inertia to orient the molecule: http://sobereva.com/426"
    all_ele, all_cart = molecule.get_ele_and_cart()

    if len(all_cart) == 1:
        return np.array([[0.0, 0.0, 0.0]])

    masses = np.array([chemical_knowledge.element_masses[x] for x in all_ele])
    relative_position = all_cart - molecule.get_center_of_mass()
    I_matrix = np.zeros((3, 3))

    for mass, cart_xyz in zip(masses, relative_position):
        I_matrix[0][0] += mass * (cart_xyz[1] ** 2 + cart_xyz[2] ** 2)
        I_matrix[1][1] += mass * (cart_xyz[0] ** 2 + cart_xyz[2] ** 2)
        I_matrix[2][2] += mass * (cart_xyz[0] ** 2 + cart_xyz[1] ** 2)
        I_matrix[0][1] += -(mass * cart_xyz[0] * cart_xyz[1])
        I_matrix[0][2] += -(mass * cart_xyz[0] * cart_xyz[2])
        I_matrix[1][2] += -(mass * cart_xyz[1] * cart_xyz[2])

    I_matrix = (
        I_matrix + I_matrix.T - np.diag(np.diag(I_matrix))
    )  # study from Peter, faster

    axes = np.linalg.eigh(I_matrix)[
        1
    ].T  # I_matrix is Moment of Inertia matrix, np.linalg.eigh calculates eigenvalues and eigenvectors, returning a tuple containing two elements. The first element is a one-dimensional array containing the eigenvalues of the matrix; the second element is a two-dimensional array, where each column corresponds to an eigenvector associated with an eigenvalue. Therefore, [1].T is used.
    new_positions = np.dot(relative_position, axes.T)
    molecule.put_ele_cart_back(all_ele, new_positions)
    return molecule


def get_rotate_matrix(v):
    """
    This function is using a len(v)==3 vector to make a rotation matrix.
    The vector is first transformed in to a Quaternion, then using that Quaternion to construct a rotation matrix
    This matrix will not suffer from Gimbal lock
    left hand universe is used here
    :param v:
    :return:
    """
    x = (1.0 - v[0]) ** 0.5 * np.sin(2.0 * np.pi * v[1])
    y = (1.0 - v[0]) ** 0.5 * np.cos(2.0 * np.pi * v[1])
    z = v[0] ** 0.5 * np.sin(2.0 * np.pi * v[2])
    w = v[0] ** 0.5 * np.cos(2.0 * np.pi * v[2])
    return np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y + 2 * z * w, 2 * x * z - 2 * y * w],
            [2 * x * y - 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z + 2 * x * w],
            [2 * x * z + 2 * y * w, 2 * y * z - 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )


def f2c_matrix(cell_para):
    a = cell_para[0][0]
    b = cell_para[0][1]
    c = cell_para[0][2]
    alpha = cell_para[1][0]
    beta = cell_para[1][1]
    gamma = cell_para[1][2]
    ca = np.cos(alpha / 180 * np.pi)
    cb = np.cos(beta / 180 * np.pi)
    cg = np.cos(gamma / 180 * np.pi)
    sg = np.sin(gamma / 180 * np.pi)
    X = 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg
    if X < 0:
        return None
    v = a * b * c * np.sqrt(X)
    return np.transpose(
        [
            [a, b * cg, c * cb],
            [0, b * sg, c * (ca - cb * cg) / sg],
            [0, 0, v / (a * b * sg)],
        ]
    )


def c2f_matrix(cell_para):
    a = cell_para[0][0]
    b = cell_para[0][1]
    c = cell_para[0][2]
    alpha = cell_para[1][0]
    beta = cell_para[1][1]
    gamma = cell_para[1][2]
    ca = np.cos(alpha / 180 * np.pi)
    cb = np.cos(beta / 180 * np.pi)
    cg = np.cos(gamma / 180 * np.pi)
    sg = np.sin(gamma / 180 * np.pi)
    X = 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg
    if X < 0:
        return None
    v = a * b * c * np.sqrt(X)
    return np.array(
        [
            [1 / a, 0.0, 0.0],
            [-cg / (a * sg), 1 / (b * sg), 0],
            [
                b * c * (ca * cg - cb) / v / sg,
                a * c * (cb * cg - ca) / v / sg,
                a * b * sg / v,
            ],
        ]
    )


def apply_SYMM(frac_xyzs, SYMM):
    all_reflect_position = []
    all_matrix_M = []
    all_matrix_C = []
    for sym_opt in SYMM:
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

    for j in range(0, len(all_matrix_M)):
        new_positions = np.dot(np.array([frac_xyzs]), all_matrix_M[j]) + all_matrix_C[j]
        all_reflect_position.append(new_positions.squeeze())

    return np.array(all_reflect_position)
