"""
Project-specific helpers to parse existing data so they could be added into the h5 datasets.
A collection of implementation for different side projects -- document and re-organize later.
TODO

"""
import imageio
import numpy as np
from scipy import io as sio
from .base import DataParser
from typing import Sequence, Dict
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SimpleDataReader(DataParser):
    """Compile the data for H5py dataset creation from a previous messy dataset.
    root/phase/feat_name/***.mat
    root/phase/CT_slices/***.jpg
    Simply a wrapper to read from the dataset

    Completely expose the iteration of each data point to the main script.
    It only simplify the codes in the script as the helper
    to get all feature arrays aligned with the given slice/patch number,

    """
    KEY_VALUE: str = 'value'
    _data_root: str
    _slice_folder_name: str
    _feat_names: Sequence[str]
    _phase: str
    _slice_list: Sequence[str]
    _feat_list_dict: Dict[str, Sequence[str]]
    _img_ext: str
    __slice_patch_idx_list: Sequence[str]

    @classmethod
    def build(cls, slice_folder_name, feat_names, current_phase, slice_list, feat_list_dict):
        p = slice_list[0]
        _, img_ext = os.path.splitext(p)
        slice_folder, image_name = os.path.split(p)
        phase_folder, slice_base = os.path.split(slice_folder)
        root_folder, phase = os.path.split(phase_folder)

        assert slice_base == slice_folder_name, f"{slice_base} vs. {slice_folder_name}"
        assert phase == current_phase, f"{phase} vs. {current_phase}"
        assert set(feat_names) == set(feat_list_dict.keys())

        return cls(data_root=root_folder,
                   slice_folder_name=slice_base,
                   feat_names=feat_names,
                   phase=current_phase,
                   img_ext=img_ext,
                   slice_list=slice_list,
                   feat_list_dict=feat_list_dict)

    # @property
    # def slice_patch_idx_list(self):
    #     return self.__slice_patch_idx_list

    def __init__(self,
                 data_root: str,
                 slice_folder_name: str,
                 feat_names: str,
                 phase: str,
                 img_ext: str,
                 slice_list: Sequence[str],
                 feat_list_dict: Dict[str, Sequence[str]]):
        super().__init__()
        self._data_root = data_root
        self._slice_folder_name = slice_folder_name
        self._feat_names = feat_names
        self._phase = phase
        self._img_ext = img_ext
        self._slice_list = slice_list
        self._feat_list_dict = feat_list_dict
        self._init_slice_patch_idx_list(self._slice_list)

    @staticmethod
    def slice_patch_id(slice_loc):
        basename = os.path.basename(slice_loc)
        slice_patch_id, _ = os.path.splitext(basename)
        return slice_patch_id

    @staticmethod
    def feature_mat_array_helper(slice_patch_id, feat_name, data_root, phase, key_value=KEY_VALUE):
        filename = os.path.join(data_root, phase, feat_name, f"{slice_patch_id}.mat")
        mat = sio.loadmat(filename)[key_value]
        logger.debug(f"feat: {filename}")
        return mat

    @staticmethod
    def feature_mat_array(slice_patch_id, feat_name_list, data_root, phase, key_value=KEY_VALUE):
        array_list = []
        for feat_name in feat_name_list:
            feat_arr_single = SimpleDataReader.feature_mat_array_helper(slice_patch_id,
                                                                        feat_name,
                                                                        data_root,
                                                                        phase,
                                                                        key_value)
            array_list.append(feat_arr_single)
        return np.dstack(array_list)

    def feat_from_idx(self, slice_patch_id, key_value=KEY_VALUE):
        return SimpleDataReader.feature_mat_array(slice_patch_id,
                                                  feat_name_list=self._feat_names,
                                                  data_root=self._data_root,
                                                  phase=self._phase,
                                                  key_value=key_value)

    @staticmethod
    def slice_jpg_read_helper(slice_patch_id, data_root, phase, slice_folder_name, ext='.jpg'):
        filename = os.path.join(data_root, phase, slice_folder_name, f"{slice_patch_id}{ext}")
        logger.debug(f"slice: {filename}")
        return imageio.v2.imread(filename)

    def slice_from_idx(self, slice_patch_id):
        return SimpleDataReader.slice_jpg_read_helper(slice_patch_id,
                                                      data_root=self._data_root,
                                                      phase=self._phase,
                                                      slice_folder_name=self._slice_folder_name,
                                                      ext=self._img_ext)

    def data_point(self, slice_patch_id, key_value=KEY_VALUE):
        x = self.slice_from_idx(slice_patch_id)
        y = self.feat_from_idx(slice_patch_id, key_value=key_value)
        return x, y


class MatReader(DataParser):
    MAT_VAR_NAME = 'haralick_struct'
    KEY_FEAT_NAME = 'names'
    KEY_FEAT_MAP = 'img3'
    KEY_CT_SLICE = 'img_raw'
    KEY_CT_MASK = 'mask'
    KEY_GRAY_LEVEL = 'graylevels'
    KEY_WINDOW = 'hws'
    KEY_USE_ALL = 'use_all'

    def __init__(self, data_root, mat_data_list):
        super().__init__()
        self._data_root = data_root
        self._slice_list = mat_data_list
        self._init_slice_patch_idx_list(self._slice_list)

    @staticmethod
    def mat_read_helper(fname):
        data = sio.loadmat(fname)
        data_struct = data[MatReader.MAT_VAR_NAME][0][0]
        struct_field_names = list(data[MatReader.MAT_VAR_NAME][0][0].dtype.names)
        return data_struct, struct_field_names

        # feat_map_idx = struct_field_names.index(key_feat_map)
        # feature_maps = data_struct[feat_map_idx]
    @staticmethod
    def mat_read_data(fname, target_var_name):
        data_struct, struct_field_names = MatReader.mat_read_helper(fname)
        data_idx = struct_field_names.index(target_var_name)
        data_to_read = data_struct[data_idx]
        return data_to_read

    def feat_from_idx(self, slice_patch_id):
        fname = os.path.join(self._data_root, f"{slice_patch_id}.mat")
        # data_struct, struct_field_names = MatReader.mat_read_helper(fname)
        return MatReader.mat_read_data(fname, MatReader.KEY_FEAT_MAP)
        # return feature_maps

    def slice_from_idx(self, slice_patch_id):
        fname = os.path.join(self._data_root, f"{slice_patch_id}.mat")
        return MatReader.mat_read_data(fname, MatReader.KEY_CT_SLICE)

    def mask_from_idx(self, slice_patch_id):
        fname = os.path.join(self._data_root, f"{slice_patch_id}.mat")
        return MatReader.mat_read_data(fname, MatReader.KEY_CT_MASK)

    def data_point(self, slice_patch_id):
        x = self.slice_from_idx(slice_patch_id)
        y = self.feat_from_idx(slice_patch_id)
        mask = self.mask_from_idx(slice_patch_id)
        return x, y, mask
