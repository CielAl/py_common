# wrapper
import ctypes
import logging
import os
import sys

import numpy as np
from numpy.ctypeslib import ndpointer

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

__LIB_NAME__: str = 'canon'
__FUNC_NAME__: str = 'canonical'
__WIN32_NAME__: str = 'win32'
__dir_name__ = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'compiled')


def get_library(directory: str, lib_name: str, func_name: str):
    if sys.platform == __WIN32_NAME__:
        lib_getter = ctypes.WinDLL
        ext = '.dll'
    else:
        lib_getter = ctypes.cdll.LoadLibrary
        ext = '.so'
    lib_basename = f"{lib_name}{ext}"
    lib_fullname = os.path.join(directory, lib_basename)
    logger.debug(lib_fullname)
    lib = lib_getter(os.path.abspath(lib_fullname))
    return getattr(lib, func_name)


def validate_symmetric(mat: np.ndarray):
    if np.allclose(mat, np.triu(mat)) or np.allclose(mat, np.tril(mat)):
        mat = mat + mat.transpose()
    return mat


def canonical(sub_graph_old: np.ndarray, classes_old=None, lib_dir: str = __dir_name__, flatten: bool = True):
    sub_graph_old = sub_graph_old.astype(np.float64)
    sub_graph_old = validate_symmetric(sub_graph_old)
    assert np.allclose(sub_graph_old, sub_graph_old.T, rtol=1e-05, atol=1e-08), f'Not symmetric: Nauty requires' \
                                                                                f' symmetric matrix, ' \
                                                                                f'not a triangular one.'
    num_node = sub_graph_old.shape[0]
    if classes_old is None:
        classes_old = np.ones([num_node])
    classes_old = classes_old.astype(np.float64)
    order: np.ndarray = np.argsort(classes_old)

    classes = classes_old[order]
    sub_graph = sub_graph_old[order][order]
    # must be a binary graph. unweighted
    sub_graph = (sub_graph > 0).astype(np.float64)
    # classes = np.concatenate([classes,classes[-1]+1])
    classes = np.append(classes, classes[-1] + 1)
    colors_nauty = (1 - np.diff(classes, axis=0)).astype(np.float64)
    num_edges = int(np.count_nonzero(sub_graph))

    degrees = (sub_graph > 0).sum(axis=1, keepdims=True).astype(np.float64)
    canonical_func = get_library(directory=lib_dir,
                                 lib_name=__LIB_NAME__,
                                 func_name=__FUNC_NAME__)
    canonical_func.restype = ctypes.c_int
    logging.debug(str(canonical_func))
    # somehow they are all double while passed to C whether or not converted the type
    clabels = np.zeros([num_node, 1]).astype(np.float64)  # must be float64 for ndpointer

    # corresponding code treat input as double*
    canonical_func.argtypes = [ctypes.c_int,
                               ctypes.c_int,
                               ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                               ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                               ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                               ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    # out_pointer = ctypes.c_void_p(clabels.astype(float).ctypes.data)
    status = canonical_func(num_edges,
                            num_node,  # .ctypes.data
                            sub_graph,
                            degrees,
                            colors_nauty,
                            clabels
                            )
    assert status == 1, "Nauty_Failure"
    clabels = clabels.astype(np.int)
    order = order[clabels]
    if flatten:
        order = order.ravel()
    return order
# main	 clabels = canonical(subgraph1, num_edges, degrees, colors_nauty);
