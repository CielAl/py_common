"""wrapper function of loadmat interfaces."""
from scipy.io import loadmat as loadmat_sio
from ..functional import HDF5Mat
from typing import List, Optional


def loadmat(fp: str, *, squeeze_me: Optional[bool] = False, encoding_h5: Optional[str] = 'utf-16',
            exclude_h5: List[str],
            **loadmat_kwargs):
    """unified interface to read v7.3+ or below mat files.

    For detail signature of mat below V7.3
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

    Args:
        fp: fname
        squeeze_me: whether to squeeze the matrix
        encoding_h5: what encoding the HDF5 uses in v7.3 mat.
        exclude_h5: list of paths (i.e., variables or nested fields) to exclude from the hdf5 root group.
        **loadmat_kwargs: keywords argument besides squeeze_me for scipy.io.loadmat.

    Returns:

    """
    try:
        return loadmat_sio(fp, squeeze_me=squeeze_me, **loadmat_kwargs)
    except NotImplementedError:
        return HDF5Mat.loadmat_v7_3(fp, squeeze=squeeze_me, exclude_list=exclude_h5, encoding=encoding_h5)

