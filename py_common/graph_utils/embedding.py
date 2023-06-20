from typing import List, Callable, Union
import networkx
import numpy as np
from .utils import AdjUtil
from .fastwl import FastWL
from .canon_wrapper import canonical


def centrality(graph: Union[np.ndarray, networkx.Graph], cent_func: Callable):
    if not isinstance(graph, networkx.Graph):
        graph = networkx.from_numpy_array(graph)
    centrality_dict = cent_func(graph)
    centrality_labels = np.asarray(list(centrality_dict.values()))
    return centrality_labels


def centrality_group(graph: np.ndarray, cent_list):
    cent_label_list = []
    for cent_name in cent_list:
        centrality_labels = centrality(graph, cent_name)
        cent_label_list.append(centrality_labels)
    return cent_label_list


def weisfeiler_lehman_1d(adj_mat: np.ndarray):
    adj_mat = AdjUtil.to_symmetric_adj(adj_mat)
    wl = FastWL(adj_mat.astype(np.float32))
    return wl.equivalent_class().ravel()


def wl_canon(adj_mat: np.ndarray, centrality_list):
    node_order = weisfeiler_lehman_1d(adj_mat).astype(np.float64)
    # this ensures that the adjacency matrix is not triangular
    adj_mat = AdjUtil.to_symmetric_adj(adj_mat)
    c_labels = canonical(adj_mat, classes_old=node_order)
    cent_label_group: List = centrality_group(adj_mat, centrality_list)
    sort_keys = [c_labels, node_order] + cent_label_group
    return np.lexsort(sort_keys), sort_keys
