import inspect
from tqdm import tqdm
import numpy as np
from .creation import H5Util
from typing import Tuple, List, Union, Generator, Callable, Dict, Sequence, Type, Any, Set

_type_func_str = Callable[[str, Union[Sequence, np.ndarray]], H5Util]
_type_func_arr = Callable[
    [str, Tuple[int, ...], Union[int, None], Union[int, None], Union[Type, None]]
    , H5Util]
_type_add_func = Union[_type_func_arr, _type_func_str]


class H5ParserCore:
    """A barebone parser to handle creation of hdf5 dataset.

    Contains an array dataset that is presumably to be the image or feature (preserved field),
    and a str dataset which serves as the identifier (e.g., the name of the wsi).

    Anything else can be written as HDF5 arrays. Please follow the h5py for dtypes. Note that types such as string
    are handled differently in HDF5 (e.g., h5py.string_dtype instead of str or numpy objects)

    """

    __h5util: H5Util
    __h5_name: str
    __patch_shape: Tuple[int, ...]
    __max_size: Union[int, None]
    __chunk_size: Union[int, None]

    CONST_DATASET_NAME: str = 'h5_name'
    CONST_TILE: str = 'tile'
    CONST_BBOX: str = 'bbox'
    CONST_LABEL: str = 'label'
    CONST_URI: str = 'uri'

    CONST_PRESERVED_FIELDS: List[str] = []  # CONST_DATASET_NAME, CONST_TILE
    TYPE_STR: str = "str"
    TYPE_ARRAY: str = "array"
    SUPPORTED_ARRAY_TYPES: Set[str] = {TYPE_STR, TYPE_ARRAY}

    ERROR_DUPLICATE: str = "Duplicate field name"
    ERROR_NOT_CALLABLE: str = "Not a Callable"
    ERROR_NOT_VALID_FIELD_TYPE: str = f"Invalid dataset type. Must be from {TYPE_ARRAY, TYPE_STR}"

    ERROR_H5_DUPLICATE: str = "H5Root already contain the field name."
    ERROR_H5_PRESERVED: str = "The field name is preserved."

    @property
    def h5util(self) -> H5Util:
        """The H5Util object to manage the creation/writing of HDF5 and the file handles

        Returns:

        """
        return self.__h5util

    @property
    def h5_name(self) -> str:
        """Name of dataset identifier

        Returns:
            dataset_name attribute (str)
        """
        return self.__h5_name

    @property
    def patch_shape(self):
        return self.__patch_shape

    @property
    def max_size(self) -> Union[int, None]:
        return self.__max_size

    @property
    def chunk_size(self) -> Union[int, None]:
        return self.__chunk_size

    @property
    def primary_dtype(self) -> Type:
        return self.__primary_dtype

    def __init__(self, h5util: H5Util,
                 h5_name: str,
                 data_shape: Tuple[int, ...],
                 max_size: Union[int, None] = None,
                 chunk_size: Union[int, None] = None,
                 primary_dtype: Union[Type, None] = None):
        """Init func.

        Args:
            h5util: corresponding h5util object
            h5_name: name/identifier of this h5 object, e.g., the name of WSI where data are extracted
            data_shape: shape of the primary data. See h5py_helper.H5Util.add_array_dataset for more details
            max_size: Optional. Max size of the data (e.g., total # of instances)
            chunk_size: chunk size for fast reading
            primary_dtype: Optional. The dtype of primary data. Note: either of data_shape and primary_dtype
                must be specified
        """
        self.__h5util = h5util
        self.__h5_name = h5_name
        self.__patch_shape = data_shape

        self.__max_size = max_size
        self.__chunk_size = chunk_size
        self.__primary_dtype = primary_dtype
        self.init_core_dataset()

    def validate_func(self, field_type: str) -> _type_add_func:
        """validate if the name of field_type is mapped to H5Util's functions for dataset creation.

        Args:
            field_type: must be H5ParserCore.TYPE_STR or H5ParserCore.TYPE_ARRAY

        Returns:
            The function to create h5py dataset.
        """
        func_dict: Dict[str, _type_add_func] = {
            H5ParserCore.TYPE_STR: self.h5util.add_str_dataset,
            H5ParserCore.TYPE_ARRAY: self.h5util.add_array_dataset,
        }
        func = func_dict.get(field_type, None)
        assert isinstance(func, Callable), f"{H5ParserCore.ERROR_NOT_CALLABLE}, but {type(func)}"
        assert func is not None, f"{H5ParserCore.ERROR_NOT_VALID_FIELD_TYPE}"
        return func

    def validate_field_name(self, is_preserved: bool, name: str):
        """Validate whether a field name can be used.

        If is_predefined is not set -- the function will also raise the exception if the field name is already
        a preserved field name.

        It will also raise the exception if the h5root already has a dataset with this name.

        Args:
            is_preserved: Whether to insert a preserved name. If set to False, the function will
                validate whether it collide with existing preserved names in `CONST_PRESERVED_FIELDS`
            name: field name (i.e., dataset name of h5py under the h5root)
        Returns:
            name
        """
        assert name not in self.h5util.h5root, f"{H5ParserCore.ERROR_H5_DUPLICATE}"
        assert is_preserved or name not in H5ParserCore.CONST_PRESERVED_FIELDS,\
            f"{H5ParserCore.ERROR_H5_PRESERVED}"
        return name

    def add_dataset(self, is_preserved: bool, field_type: str, *, name: str, **kwargs):
        """Wrapper of lower-level h5util operations but inspect if the field is already created under h5root.

        Args:
            is_preserved: whether the field is preserved
            field_type: array dataset or string dataset
            name: group name for h5py.create_dataset
            **kwargs: See H5util.add_str_dataset and add_array_dataset

        Returns:

        """

        data_add_func = self.validate_func(field_type)
        name = self.validate_field_name(is_preserved, name)
        data_add_func(name, **kwargs)
        return self

    def init_core_dataset(self):
        """Init the core dataset, which are the two preserved types by default: tiles and the dataset name.

        Returns:

        """
        self.add_dataset(True, H5ParserCore.TYPE_ARRAY,
                         name=H5ParserCore.CONST_TILE,
                         shape=self.patch_shape,
                         max_size=self.max_size,
                         dtype=self.primary_dtype,
                         chunk_size=self.chunk_size)\
            .add_dataset(True, H5ParserCore.TYPE_STR,
                         name=H5ParserCore.CONST_DATASET_NAME,
                         str_data_in=self.h5_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5util.__exit__(exc_type, exc_val, exc_tb)


class Parser:

    field_names: Union[List[str], None]
    dataset_types: Union[List[str], None]
    data_dtypes: Union[List[Union[Type, None]], None]
    data_shapes: Union[List[Tuple[int, ...]], None]
    data_init_strs: Union[List[str], None]
    max_sizes: Union[List[Union[int, None]], None]
    chunk_sizes: Union[List[Union[int, None]], None]
    arg_list: List[Tuple[str, str, Dict[str, Any]]]
    ERROR_EITHER_ALL_OPTIONAL_OR_ALL_SPECIFIED: str = "Collated fields must be either all None (unspecified) or" \
                                                      "all specified as List of elements."
    ERROR_NUM_FIELDS_MISMATCH: str = "Number of elements in different arg list of fields do not match."
    ERROR_NONEXIST_FIELD_TO_WRITE: str = "Field does not exist"

    def __init__(self,
                 parser_core: H5ParserCore,
                 field_names: Union[List[str], None] = None,
                 data_shapes: Union[List[Tuple[int, ...]], None] = None,
                 dataset_types: Union[List[str], None] = None,
                 data_dtypes: Union[List[Union[Type, None]], None] = None,
                 max_sizes: Union[List[Union[int, None]], None] = None,
                 chunk_sizes: Union[List[Union[int, None]], None] = None,
                 data_init_strs: Union[List[str], None] = None,
                 delay_add_datasets: bool = False,
                 ):
        """Parser object. Please instantiate using the factory builder.

        Args:
            parser_core: The _H5ParserCore object
            field_names: names of the fields to add into the h5root other than the preserved ones in parser core
                None if nothing to add
            data_shapes: list of shapes corresponding to each field.
            dataset_types: list of dataset types, so far either _H5ParserCore.TYPE_STR or TYPE_ARR is supported
                See field_names
            data_dtypes:  list of dataset dtype. Can be optional. Aligned to field_names. If any individual field
                does not need dtype, leave it None in the corresponding position in the list
            max_sizes: see above and max_size in H5ParserCore
            chunk_sizes: see above and chunk_size in H5ParserCore
            data_init_strs: init value for str datasets
            delay_add_datasets: instantly add all datasets to h5root if False, otherwise manually instantiated.
        """
        field_names, data_shapes, dataset_types,\
            data_dtypes, max_sizes, chunk_sizes,\
            data_init_strs = Parser.validate_args(field_names,
                                                  data_shapes, dataset_types,
                                                  data_dtypes, max_sizes,
                                                  chunk_sizes, data_init_strs)

        self.parser_core = parser_core
        self.field_names = field_names
        self.dataset_types = dataset_types
        self.data_shapes = data_shapes
        self.data_dtypes = data_dtypes
        self.max_sizes = max_sizes
        self.chunk_sizes = chunk_sizes
        self.data_init_strs = data_init_strs

        self.arg_list = Parser.parsed_h5_args(field_names,
                                              data_shapes, dataset_types,
                                              data_dtypes, max_sizes,
                                              chunk_sizes, data_init_strs)

        if not delay_add_datasets:
            self.add_datasets()

    @staticmethod
    def add_datasets_helper(parser_core: H5ParserCore, arg_list: List[Tuple[str, str, Dict[str, Any]]]):
        """Helper function of `add_datasets`.

        Args:
            parser_core: H5ParserCore object.
            arg_list: vectorized arguments for multiple call of `add_dataset`

        Returns:

        """
        for name, dataset_type, kwargs in arg_list:
            parser_core.add_dataset(False, dataset_type, name=name, **kwargs)

    def add_datasets(self):
        """Vectorized operation of adding multiple datasets given by the arg_list, which is parsed by parsed_h5_args.

        The self.arg_list is parsed by parsed_h5_args from vectorized input arguments of __init__.

        Returns:
            self
        """
        Parser.add_datasets_helper(self.parser_core, self.arg_list)
        return self

    @staticmethod
    def validate_args(*args):
        """validate the input arguments.

        Args:
            See __init__. Arguments must either be all None or all specified. Otherwise generate a list of Nones.

        Returns:
            validated argument list: Arguments must either be all None or all specified.
            Otherwise generate a list of Nones.
        """
        # nothing is specified
        all_none = all(x is None for x in args)
        # everything is specified
        all_specified = all(x is not None for x in args)
        num_fields = 0 if all_none else [len(x) for x in args if x is not None][0]
        args_out = args
        if not all_specified:
            # get the first (or any) not None field list and get its length
            # replace None as list of Nones aligned to other args
            args_out = [x if x is not None else [None] * num_fields for x in args]

        all_none = all(x is None for x in args_out)
        # everything is specified
        all_specified = all(x is not None for x in args_out)
        assert all(len(x) == num_fields for x in args_out), Parser.ERROR_NUM_FIELDS_MISMATCH
        assert all_none or all_specified, Parser.ERROR_EITHER_ALL_OPTIONAL_OR_ALL_SPECIFIED
        # three situation
        # 1: all specified
        # 2: some not specified --> convert to List of Nones
        # 3: all none --> list of empty lists
        return args_out

    @staticmethod
    def parsed_h5_args_helper(name: str,
                              shape: Union[Tuple[int, ...]],
                              dataset_type: str,
                              dtype: Union[Type, None],
                              max_size: Union[int, None],
                              chunk_size: Union[int, None],
                              init_str: Union[str, None]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Helper function to get the args (name and kwargs in H5ParserCore.add_dataset)
        Args:
            name:
            shape:
            dataset_type:
            dtype:
            max_size:
            chunk_size:
            init_str:

        Returns:

        """

        assert dataset_type in H5ParserCore.SUPPORTED_ARRAY_TYPES, f"{H5ParserCore.ERROR_NOT_VALID_FIELD_TYPE}" \
                                                                    f"Got: {dataset_type}"
        core_array_func_args = \
            set(inspect.signature(H5Util.add_array_dataset).parameters.keys()) - {'self'} - {'name'}
        core_str_func_args = \
            set(inspect.signature(H5Util.add_str_dataset).parameters.keys()) - {'self'} - {'name'}
        if dataset_type == H5ParserCore.TYPE_STR:
            kwargs = {
                "str_data_in": init_str
            }
            assert set(kwargs.keys()).issubset(core_str_func_args)
        else:
            kwargs = {
                    "shape": shape,
                    "max_size": max_size,
                    "dtype": dtype,
                    "chunk_size": chunk_size
            }
            assert set(kwargs.keys()).issubset(core_array_func_args)
        return name, dataset_type, kwargs

    @staticmethod
    def parsed_h5_args(
                 field_names: Union[List[str], None],
                 data_shapes: Union[List[Tuple[int, ...]], None],
                 dataset_types: Union[List[str], None],
                 data_dtypes: Union[List[Union[Type, None]], None],
                 field_max_sizes: Union[List[Union[int, None]], None],
                 field_chunk_sizes: Union[List[Union[int, None]], None],
                 data_init_strs: Union[List[str], None]):
        arguments = Parser.validate_args(field_names,
                                         data_shapes, dataset_types,
                                         data_dtypes, field_max_sizes,
                                         field_chunk_sizes, data_init_strs)

        arg_list = []
        for (name, shape, dataset_type, dtype, max_size, chunk_size, init_str) in zip(*arguments):
            name, dataset_type, kwargs = Parser.parsed_h5_args_helper(name, shape,
                                                                      dataset_type, dtype,
                                                                      max_size, chunk_size, init_str)
            arg_list.append((name, dataset_type, kwargs))

        return arg_list

    @classmethod
    def build(cls,
              uri: str,
              mode: str,
              h5_name: str,
              primary_data_shape: Tuple[int, ...],
              primary_max_size: Union[int, None] = None,
              primary_chunk_size: Union[int, None] = None,
              primary_dtype: Union[Type, None] = None,
              field_names: Union[List[str], None] = None,
              data_shapes: Union[List[Tuple[int, ...]], None] = None,
              dataset_types: Union[List[str], None] = None,
              data_dtypes: Union[List[Union[Type, None]], None] = None,
              max_sizes: Union[List[Union[int, None]], None] = None,
              chunk_sizes: Union[List[Union[int, None]], None] = None,
              data_init_strs: Union[List[str], None] = None,
              delay_add_datasets: bool = False,
              ) -> "Parser":
        """Parser object. Please instantiate using the factory builder.

        Args:
            uri: where to export the hdf5 file
            mode: IO mode. (e.g., w, w-, a, etc.)
            h5_name: str,
            primary_data_shape: Tuple[int, ...]: Array shape for the preserved tile fields in H5ParserCore
            primary_max_size:  See primary_data_shape
            primary_chunk_size: See primary_data_shape
            primary_dtype: See primary_data_shape
            field_names: names of the fields to add into the h5root other than the preserved ones in parser core
                None if nothing to add
            data_shapes: list of shapes corresponding to each field.
            dataset_types: list of dataset types, so far either _H5ParserCore.TYPE_STR or TYPE_ARR is supported
                See field_names
            data_dtypes:  list of dataset dtype. Can be optional. Aligned to field_names. If any individual field
                does not need dtype, leave it None in the corresponding position in the list
            max_sizes: see above and max_size in H5ParserCore
            chunk_sizes: see above and chunk_size in H5ParserCore
            data_init_strs: init value for str datasets
            delay_add_datasets: instantly add all datasets to h5root if False, otherwise manually instantiated.

        Returns:
            Parser
        """
        h5util = H5Util.build(uri, mode)
        parser_core = H5ParserCore(h5util, h5_name,
                                   data_shape=primary_data_shape,
                                   max_size=primary_max_size,
                                   chunk_size=primary_chunk_size,
                                   primary_dtype=primary_dtype,
                                   )

        return cls(parser_core, field_names,
                   data_shapes,
                   dataset_types,
                   data_dtypes,
                   max_sizes,
                   chunk_sizes,
                   data_init_strs,
                   delay_add_datasets)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parser_core.__exit__(exc_type, exc_val, exc_tb)

    @property
    def h5util(self):
        return self.parser_core.h5util

    def write_array(self, field_order: List[str],
                    data_generator: Generator,
                    verbose: bool = False,
                    cutoff_size: Union[int, None] = None,
                    total_length: Union[int, None] = None):
        """Write/append array data into HDF5 file using generator, which is generic enough to be applied in several
         projects with different data needs/structure/layout

        However, this compromises the control of data preprocessing and filtering over simplicity of interfaces.
        With more complicated data processing (e.g., with on-the-fly color transform or tissue filtering),
        the complexity of generator will also increase and it would be difficult to troubleshoot. In this case,
        use the `write_single_data` below to unfold the writing procedure.

        Args:
            field_order: The ordered field name of data in each iteration.
            data_generator: A generator that yields different data (e.g., image/mask/label/etc.)
                Note that each data point follows the convention of (N, ....) wherein the leading dimension represents
                number of instance. E.g, for one image with HxW, a singleton leading dimension must be added beforehand.
            verbose: Whether print the progress bar
            cutoff_size: Size to return early. No cutoff if set to None or negative
            total_length: Total number of data points. Only for verbose (progress bar length) since generator itself
                does not have length

        Returns:

        """
        generator_work = data_generator if not verbose else tqdm(data_generator, total=total_length)
        invalid_fields = [x for x in field_order if x not in self.h5util.h5root]
        assert len(invalid_fields) == 0, f"{Parser.ERROR_NONEXIST_FIELD_TO_WRITE}: {invalid_fields}"
        for idx, data_points_single in enumerate(generator_work):
            # check cut off
            if cutoff_size is not None and 0 <= cutoff_size <= idx:
                break
            # for name, data in zip(field_order, data_points_single):
            #     self.h5util.add_data_to_array(name, data, insert_idx=idx)
            self.write_single_data(field_order, data_points_single, idx)

    def write_single_data(self, field_order: List[str], data_points_single: Sequence[np.ndarray],
                          insert_idx: int):
        """Write (insert) a single data corresponding to insert_idx.

        A single data point contains multiple field corresponding to the same insert_idx.
        HDF5 array's size is updated determined by the insert_idx

        Args:
            field_order: ordered list of field names.
            data_points_single: corresponding individual data aligned to each of the field in field_order
            insert_idx: index to assert in HDF5 array

        Returns:

        """
        for name, data in zip(field_order, data_points_single):
            self.h5util.add_data_to_array(name, data, insert_idx=insert_idx)