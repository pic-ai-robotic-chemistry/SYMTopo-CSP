import copy

import networkx as nx
from scipy.spatial import cKDTree as KDTree

from basic_function import chemical_knowledge, operation


def detect_is_frame_vdw_new(crystal, tolerance: float = 1.2) -> bool:
    """Return whether the periodic van der Waals contact graph is connected."""
    crystal_temp = copy.deepcopy(crystal)
    crystal_temp.make_p1()
    crystal_temp.move_atom_into_cell()
    crystal_new = operation.super_cell(
        crystal_temp, cell_range=[[-1, 1], [-1, 1], [-1, 1]]
    )
    all_ele, all_carts = crystal_new.get_ele_and_cart()
    vdw_max = 0
    for ele in set(all_ele):
        vdw_max = max(vdw_max, chemical_knowledge.element_vdw_radii[ele])
    distance_threshold = vdw_max * tolerance * 2
    points = all_carts
    # Query candidate contacts efficiently with a k-d tree.
    tree = KDTree(points)
    pairs = tree.query_pairs(distance_threshold)
    graph = nx.Graph()
    graph.add_edges_from(pairs)
    largest_cc = max(nx.connected_components(graph), key=len)
    return len(largest_cc) > 9 * len(crystal_temp.atoms)
