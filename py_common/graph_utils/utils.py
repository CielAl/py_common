import numpy as np


class AdjUtil:

    @staticmethod
    def check_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08):
        """
        Credit: https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
        Args:
            a:
            rtol:
            atol:

        Returns:

        """
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @staticmethod
    def is_upper(mat: np.ndarray):
        return np.allclose(mat, np.triu(mat))  # check if upper triangular

    @staticmethod
    def is_lower(mat: np.ndarray):
        return np.allclose(mat, np.tril(mat))  # check if lower triangular

    @staticmethod
    def is_symmetric(mat: np.ndarray):
        return mat.ndim == 2 and mat.shape[0] == mat.shape[1]

    @staticmethod
    def is_triangular(mat: np.ndarray):
        return AdjUtil.is_upper(mat) or AdjUtil.is_lower(mat)

    @staticmethod
    def to_symmetric_adj(g: np.ndarray):
        assert isinstance(g, np.ndarray)
        if AdjUtil.is_triangular(g):
            g = g + g.transpose()
        assert AdjUtil.is_symmetric(g)
        return g
