from typing import List, Dict, Optional, Set, Tuple, Sequence


TYPE_NODE = Dict[str, Optional['TYPE_NODE']]
PATH_STACK = Tuple[str, ...]


class PathTree:
    """A simple tree to parse the hierarchical path in HDF5 group.

    path tree as a nested dict.

    Each key of dict denotes a key of the HDF5 group/dataset of certain level.
    The value of the corresponding key is recursively the same type of dict that denotes all keys of the next level.

    An empty dict denote a leaf node.

    E.g.,

    {
    'root':
        'subfolder1': {'a': {}}
    }
    Assume all nodes are not None.

    """
    path_tree: TYPE_NODE
    path_set: Set[PATH_STACK]

    @staticmethod
    def sorted_copy(path_list: List[str]) -> List[str]:
        """Sort the list of path following lexicographic order.

        Simplify the tree traverse as the order of the paths will naturally be aligned with the DFS.

        Args:
            path_list: a list of HDF5 paths

        Returns:
            A copy of sorted list.
        """
        copied_list = [x for x in path_list]
        copied_list.sort()
        return copied_list

    @staticmethod
    def tree_parsed(path_list: List[str], delimiter: str = '/') -> Tuple[TYPE_NODE, Set[PATH_STACK]]:
        """A simplified implementation to parse the path tree as a nested dict.

        Each key of dict denotes a key of the HDF5 group/dataset of certain level.
        The value of the corresponding key is recursively the same type of dict that denotes all keys of the next level.

        An empty dict denote a leaf node.

        E.g.,

        {
        'root':
            'subfolder1': {'a': {}}
        }

        Args:
            path_list: A list of paths, e.g., ['root/a/b/c', 'root/a', '#refs#/c']. The list will be copied and
                lexicographically sorted for the parsing.
            delimiter: delimiter to split the components of a path. Default: '/'

        Returns:
            Parsed nested dict from the input list of paths.
        """
        root = dict()
        path_set = set()
        if not path_list:
            return root, path_set

        exclude_list_copy_sorted = PathTree.sorted_copy(path_list)
        for name in exclude_list_copy_sorted:
            # get each node name split by the delimiter
            component_list = name.split(delimiter)
            root = PathTree.parse_path(root, path_set, component_list)

        return root, path_set

    @staticmethod
    def parse_path(root: TYPE_NODE, path_set: Set[PATH_STACK], component_list: List[str]):
        """Helper function. For a given list of component derived from the path, register all nodes into the tree.

        Args:
            root: Tree root as a dict
            path_set: set of paths that are already registered. Since all paths are lexicographically sorted already
                the order of traverse will be DFS.
            component_list: list of components splitted from the input path by a given delimiter

        Returns:
            tree with nodes registered.
        """
        # tree root
        curr_node = root
        # depth first - along the path
        stack = tuple()
        for idx, part in enumerate(component_list):
            # set default part --> dict
            stack = stack + (part, )
            if stack in path_set:
                break
            curr_node.setdefault(part, dict())
            curr_node = curr_node[part]
            # the next part is the children of this part
        path_set.add(stack)
        return root

    @classmethod
    def build(cls, path_list: List[str], delimiter: str = '/') -> 'PathTree':
        """A factory method to instantiate the node by a list of paths and delimiter

        Args:
            path_list:
            delimiter:

        Returns:

        """

        path_tree, path_set = cls.tree_parsed(path_list, delimiter)
        return cls(path_tree, path_set)

    def __init__(self, path_tree: TYPE_NODE, path_set: Optional[Set[PATH_STACK]] = None):
        """ See `tree_parsed`.

        Args:
            path_tree: a nested dict mimicking a tree of paths - see `tree_parsed`
            path_set: set of paths.
        """
        self.path_tree = path_tree
        self.path_set = path_set

    def has_node_name(self, key: str) -> bool:
        """Query whether a key (subfolder/subgroup) exists in any levels.

        Args:
            key: key to query

        Returns:
            bool
        """
        # if at current level
        if self.is_empty():
            return False
        if key in self.path_tree:
            return True
        return any(PathTree(child_value).has_node_name(key) for child_value in self.path_tree.values())

    @staticmethod
    def _empty_helper(exclude_tree: TYPE_NODE):
        """Helper function to check if the tree is empty - either none or has no elements.

        Args:
            exclude_tree:

        Returns:
            bool
        """
        return exclude_tree is None or len(exclude_tree) == 0

    def is_empty(self):
        """Check if the current tree is empty.

        Returns:

        """
        return PathTree._empty_helper(self.path_tree)

    def query_name(self, key) -> Optional[TYPE_NODE]:
        """Return the child node (dict) of a given node name (e.g., subgroup)

        Since subgroups under different levels/parents may share the same name, return the first of query results
        in lexicographical order.

        Args:
            key: name of the node

        Returns:
            dict - the child node. None if query finds nothing.
        """
        if key not in self:
            return None
        if key in self.path_tree:
            return self.path_tree[key]
        out = [child_value for child_key, child_value in self.path_tree.items() if key in PathTree(child_value)]
        return out[0] if len(out) > 0 else None

    @staticmethod
    def empty_removed_list(p_list: Sequence[str]) -> List[str]:
        """Sanitizing the input list by remove any empty chars.

        Args:
            p_list:

        Returns:
            sanitized list
        """
        assert not isinstance(p_list, str)
        return [x for x in p_list if x != '']

    @staticmethod
    def _sanitize_path(path: Sequence[str] | str, delimiter: Optional[str] = '/') -> List[str]:
        """Sanitize the path. Remove empty chars and convert to a list of components split by a delimiter

        Args:
            path: The path can either be a list of components a str of full path.
            delimiter: delimiter to split the path if path is a full path str.

        Returns:
            list of components
        """
        if not isinstance(path, str):
            return PathTree.empty_removed_list(path)
        assert delimiter is not None
        return PathTree._sanitize_path(path.split(delimiter), None)

    def query_node(self, path: Sequence[str], delimiter: Optional[str] = '/') -> Optional[TYPE_NODE]:
        """Query whether a path exists in the tree and returns the child node

        Args:
            path: path as a list of components sanitized by `_sanitize_path`
            delimiter: delimiter to split the path if path is a full path str.

        Returns:
            child node dict. None if the query finds nothing.
        """
        # left -> right = top -> bottom
        path = PathTree._sanitize_path(path,  delimiter)
        path = tuple(path)
        root = self.path_tree
        for p in path:
            if root is None:
                return root
            root = root.get(p, None)
        return root

    def __contains__(self, path: Sequence[str]) -> bool:
        """Query whether a path exists in the tree.

        Args:
            path: path as a list of components sanitized by `_sanitize_path`

        Returns:
            child node dict.
        """
        return self.query_node(path) is not None

        # each
    def is_leaf(self, path: Sequence[str]) -> bool:
        """check if the given path points to a leaf node (e.g., end of the path in the path tree)

        Args:
            path: path as a list of components sanitized by `_sanitize_path`

        Returns:
            bool

        Raises:
            AssertionError if path does not exist in the tree at all.
        """
        path = PathTree._sanitize_path(path,  '/')

        node = self.query_node(path)
        assert node is not None
        return PathTree._empty_helper(node)
