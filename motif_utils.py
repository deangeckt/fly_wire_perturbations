

from collections import defaultdict
import collections
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
from typing import Optional, Union
import h5py
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from tqdm import tqdm
from matplotlib.patches import ArrowStyle
import seaborn as sns

class Motif:
    def __init__(self, id: Union[int, str],
                 name: str,
                 adj_mat: np.ndarray,
                 role_pattern: list[tuple],
                 n_real: Optional[int] = 0):
        self.id = id
        self.name = name
        self.adj_mat = adj_mat
        self.role_pattern = role_pattern
        self.n_real = n_real

        self.sub_graphs: Optional[list[tuple[tuple]]] = []
        # dict of dicts, key=role. value = dict where keys are node name and value are their freq.
        self.node_roles: Optional[dict] = {}


triplets_names = {
    14: "mutual out",
    38: "feed forward",
    46: "regulating mutual",
    74: "mutual in",
    78: "bi-mutual",
    102: "mutual cascade",
    108: "regulated mutual",
    110: "semi clique",
    6: "fan out",
    12: "cascade",
    36: "fan in",
    98: "feed backward",
    238: "clique"
}


def get_adjacency_matrix(graph: nx.DiGraph) -> np.ndarray:
    nodes = list(graph.nodes)
    nodes.sort()
    return nx.adjacency_matrix(graph, nodelist=nodes).todense()


def get_decimal_from_bin_vec(vec: list[int]) -> int:
    decimal = 0
    for i, bit in enumerate(vec):
        decimal += bit * (2 ** i)
    return decimal


def get_id(graph: nx.DiGraph) -> int:
    adj_mat = get_adjacency_matrix(graph)
    vec = adj_mat.flatten()
    return get_decimal_from_bin_vec(list(vec))


def get_bin_vec_from_decimal(decimal: int, pad_to: int) -> list[int]:
    bin_digits = [int(d) for d in str(bin(decimal))[2:]]
    pad_amount = pad_to - len(bin_digits)
    padding = pad_amount * [0]
    bin_digits = padding + bin_digits
    return bin_digits


def get_sub_graph_from_id(decimal: int, k: int) -> nx.DiGraph:
    bin_digits = get_bin_vec_from_decimal(decimal=decimal, pad_to=k ** 2)
    bin_digits.reverse()
    adj_mat = np.array(bin_digits).reshape(k, k)
    return nx.DiGraph(adj_mat)


def get_role_pattern(adj_mat: np.ndarray) -> list[tuple]:
    """
    :param adj_mat: the adjacency matrix of the motif
    :return: list of tuples with roles of the motif, in the format: (a,b), (b,c)
    """
    ascii_start = 97
    roles: list[tuple] = []
    for src, arr in enumerate(adj_mat):
        for tar, val in enumerate(arr):
            if val != 1:
                continue
            roles.append((chr(src + ascii_start), chr(tar + ascii_start)))
    return roles


def create_base_motif(sub_id: int, k=3) -> Motif:
    name = triplets_names.get(sub_id, f"unknown-{sub_id}")
    sub_graph = get_sub_graph_from_id(decimal=sub_id, k=k)
    adj_mat = nx.adjacency_matrix(sub_graph).todense()
    role_pattern = get_role_pattern(adj_mat)
    return Motif(id=sub_id, name=name, adj_mat=adj_mat, role_pattern=role_pattern)


"""Utilities for saving/loading motif data to/from HDF5 files"""
def save_motif_participation_nodes_h5(fsl_fully_mapped, filename):
    with h5py.File(filename, 'w') as f:
        # Create a group for motifs
        motifs_group = f.create_group("motifs")

        # Store each motif's data in its own dataset
        for motif_id, triplets in fsl_fully_mapped.items():
            # Convert to numpy array if not empty
            if triplets:
                # can save values (i.e., node id) from 0 to 65535
                data = np.array(triplets, dtype=np.uint16)
            else:
                # Create empty array with correct shape for empty triplets
                data = np.empty((0, 3), dtype=np.uint16)

            # Create dataset named by the motif_id
            motifs_group.create_dataset(str(motif_id), data=data,
                                        compression="gzip", compression_opts=9)

    print(f"Data saved to {filename}")


def load_motif_participation_nodes_h5(filename, motif_id):
    with h5py.File(filename, 'r') as f:
        motif_key = str(motif_id)

        if motif_key in f["motifs"]:
            # Load the dataset
            data = f["motifs"][motif_key][:]
            # Convert to list of lists and return
            return data.tolist()
        else:
            print(f"Motif ID {motif_id} not found")
            return []


def populate_motif_data(sub_graphs, role_nodes, motif_id, filename):
    """
    Save sub_graphs and role_nodes data for a specific motif ID to an HDF5 file.
    Creates the file if it doesn't exist, or updates it if it does.

    Args:
        sub_graphs: List of tuples of tuples representing edges, with variable number of edges
        role_nodes: Dict with keys 'a', 'b', 'c', each containing {node_id: frequency}
        motif_id: The specific motif ID
        filename: Name of the HDF5 file to save to
    """
    with h5py.File(filename, 'a') as f:  # 'a' mode = append (create if doesn't exist)
        motif_str = str(motif_id)

        # Create or get the group for this motif
        if motif_str in f:
            motif_group = f[motif_str]
        else:
            motif_group = f.create_group(motif_str)

        # sub_graphs
        sg_list = [[list(edge) for edge in sg] for sg in sub_graphs]
        sg_array = np.array(sg_list, dtype=np.uint16)

        if 'sub_graphs' in motif_group:
            del motif_group['sub_graphs']  # Delete existing dataset

        motif_group.create_dataset('sub_graphs', data=sg_array, compression="gzip", compression_opts=9)

        # Handle role_nodes
        if 'role_nodes' in motif_group:
            del motif_group['role_nodes']  # Delete existing group

        roles_group = motif_group.create_group('role_nodes')

        for role in ['a', 'b', 'c']:
            if role not in role_nodes:
                continue

            node_dict = role_nodes[role]
            nodes = np.array(list(node_dict.keys()), dtype=np.uint16)
            freqs = np.array(list(node_dict.values()), dtype=np.uint32)  # 32-bit for frequencies

            roles_group.create_dataset(f'{role}_nodes', data=nodes, compression="gzip", compression_opts=9)
            roles_group.create_dataset(f'{role}_freqs', data=freqs, compression="gzip", compression_opts=9)

    print(f"Data for motif {motif_id} saved to {filename}")


def load_motif_data(motif_id, filename):
    """
    Load sub_graphs and role_nodes data for a specific motif ID from an HDF5 file.

    Args:
        motif_id: The specific motif ID to load
        filename: Name of the HDF5 file

    Returns:
        Tuple (sub_graphs, role_nodes) where:
        - sub_graphs is a list of tuples of tuples (each tuple being an edge)
        - role_nodes is a dict with keys 'a', 'b', 'c', each containing {node_id: frequency}
    """
    sub_graphs = []
    role_nodes = {'a': {}, 'b': {}, 'c': {}}

    with h5py.File(filename, 'r') as f:
        motif_str = str(motif_id)

        if motif_str not in f:
            print(f"Motif {motif_id} not found in {filename}")
            return sub_graphs, role_nodes

        motif_group = f[motif_str]

        sg_array = motif_group['sub_graphs'][:]
        sub_graphs = [tuple(tuple(edge) for edge in sg) for sg in sg_array]

        # Load role_nodes if available
        if 'role_nodes' in motif_group:
            roles_group = motif_group['role_nodes']

            for role in ['a', 'b', 'c']:
                node_key = f'{role}_nodes'
                freq_key = f'{role}_freqs'

                if node_key in roles_group and freq_key in roles_group:
                    nodes = roles_group[node_key][:]
                    freqs = roles_group[freq_key][:]

                    # Create the dictionary for this role
                    role_nodes[role] = {int(nodes[i]): int(freqs[i]) for i in range(len(nodes))}

    return sub_graphs, role_nodes


""""Utilities processing raw data"""
def sort_dict_freq(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))


def get_sub_graph_mapping_to_motif(sub_graph: tuple[tuple],
                                   motif_roles: list[tuple],
                                   ) -> dict:
    """
    :param sub_graph: tuple of tuples - the edges in the subgraph
    :param motif_roles: list of tuples with roles of a motif, in the format: (a,b), (b,c)
    :return: mapping of each role per node: e.g: {'a' : node_id, 'b' : node_id}
    """
    graph = nx.DiGraph(sub_graph)
    motif_graph = nx.DiGraph(motif_roles)
    matcher = isomorphism.DiGraphMatcher(motif_graph, graph)

    if not (matcher.is_isomorphic()):
        raise Exception('The sub graph is not isomorphic to the motif')
    return dict(matcher.mapping)


def sort_node_roles_in_sub_graph(appearances: list[tuple[tuple]],
                                 neuron_names: list,
                                 motif: Motif
                                 ) -> dict[str, dict]:
    """
    :param appearances: the sub graphs appearances of a given motif
    :param neuron_names: list of neurons names for neural network or an empty list otherwise
    :param motif: motif object with roles: list of tuples with the pattern of roles of the motif
    :return: dict, where each key is role, and the value is a sorted dict based on appearances of that role
    """
    node_roles = defaultdict(list)
    for sub_graph in appearances:
        nodes_in_sub_graph = get_sub_graph_mapping_to_motif(sub_graph, motif.role_pattern)
        for role, n in nodes_in_sub_graph.items():
            node = neuron_names[n] if neuron_names else n
            node_roles[role].append(node)

    freq_node_roles = {}
    for role in node_roles:
        freq_node_roles[str(role)] = sort_dict_freq(dict(collections.Counter(node_roles[role])))

    return freq_node_roles




"""Utilities for visualization of motifs"""
def draw_motif_single(motif: Motif, title: str, ax, font_size=20, node_size=1000, arrowsize=7):
    as_ = ArrowStyle("simple", head_length=2.5, head_width=2.5, tail_width=.4)
    ax.axis('off')
    ax.set_title(title, fontsize=12)
    graph_ = nx.DiGraph(motif.role_pattern)
    pos = nx.circular_layout(graph_)
    nx.draw_networkx(graph_, pos, ax=ax, arrowsize=arrowsize, arrowstyle=as_, node_size=node_size, font_size=font_size)



