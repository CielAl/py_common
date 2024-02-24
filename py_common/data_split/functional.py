from .base import Splitter
from typing import Optional, Iterable, Tuple, Dict, Set, List, TypeVar, Type
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

T = TypeVar('T')
K = TypeVar('K')


class StratifiedGroupShuffleSplit(Splitter):
    """Approximation of Stratified Group Shuffle Split

    If a group contains more than one label, only use the most frequent as the group-level label.
    The groups are first stratified and then data points are split accordingly.

    Attributes:
        n_splits (int): Number of re-shuffling & splitting iterations.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int, RandomState instance, or None): Controls the randomness of
            the training and test indices produced.

    Args:
        n_splits (int): Number of re-shuffling & splitting iterations.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int, RandomState instance, or None): Controls the randomness of
            the training and test indices produced.
    """

    n_splits: int
    test_size: float
    random_state: Optional[int]

    def __init__(self, n_splits: int = 10, test_size: float = 0.2, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, group=None) -> int:
        return self.n_splits

    @staticmethod
    def _validate_group_one_label(group_labels: Dict) -> Dict:
        if any(len(labels) > 1 for labels in group_labels.values()):
            raise ValueError("Each group must have only one unique label")
        return group_labels

    @staticmethod
    def _get_group_labels(group: List[K], y: List[T], *, unique_flag: bool) -> Dict[K, Set[T] | List[T]]:
        group_labels = defaultdict(set) if unique_flag else defaultdict(list)
        add_func = set.add if unique_flag else list.append
        for g, label in zip(group, y):
            # noinspection PyArgumentList
            add_func(group_labels[g], label)
        return group_labels

    @staticmethod
    def mode_np(arr: np.ndarray):
        unique, counts = np.unique(arr, return_counts=True)
        max_index = np.argmax(counts)  # Index of the most frequent item
        return unique[max_index]

    @staticmethod
    def group_label_top1(group_labels: Dict[K, List[T]]) -> Dict[K, T]:
        o_dict: Dict[K, T] = dict()
        for g, labels in group_labels.items():
            mode: T = StratifiedGroupShuffleSplit.mode_np(np.asarray(labels))
            o_dict[g] = mode
        return o_dict

    def split(self, X, y: List[T], groups: List[K]) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        # collect labels from y
        # group_labels_unique = StratifiedGroupShuffleSplit._get_group_labels(group, y, unique_flag=False)
        # group_labels = StratifiedGroupShuffleSplit._validate_group_one_label(group_labels)
        group_labels_all: Dict[K, List[T]] = StratifiedGroupShuffleSplit._get_group_labels(groups, y, unique_flag=False)

        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        group_label_dict = StratifiedGroupShuffleSplit.group_label_top1(group_labels_all)
        group_labels = [group_label_dict[g] for g in unique_groups]
        assert len(group_labels) == len(unique_groups)

        # split by group
        sss = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)

        for group_train_idx, group_test_idx in sss.split(unique_groups, group_labels):
            train_idx = np.where(np.isin(group_indices, group_train_idx))[0]
            test_idx = np.where(np.isin(group_indices, group_test_idx))[0]

            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set)
            assert train_set.union(test_set) == set(range(len(X)))
            yield train_idx, test_idx
