import numpy as np
from spatial_graph.logger import GlobalLoggers
logger = GlobalLoggers.instance().get_logger(__name__)


class FastWL(object):

    @property
    def adjacency(self):
        return self._adjacency

    @property
    def prime_list(self):
        if not hasattr(self, '_prime_list') or self._prime_list is None:
            self._prime_list = np.asarray([2, 3, 7, 19, 53, 131, 311, 719, 1619,
                                           3671, 8161, 17863, 38873, 84017, 180503, 386093,
                                           821641, 1742537, 3681131, 7754077, 16290047])
        return self._prime_list

    def __init__(self, adj):
        self._prime_list = None
        self._adjacency = adj

    '''
        labels: start from 0. 
    '''

    def transformation(self, labels: np.ndarray):
        # label: must be 1-based.
        labels = np.atleast_2d(labels)
        labels = labels.reshape(labels.size, 1)
        labels = labels.astype(np.float32)
        logger.info(f"shape: {labels.shape}")
        num_labels = labels.max().astype(int)  # label start from 0. plus 1 above to avoid -inf
        # remove plus 1 for the indices shifting
        prime_indices = np.ceil(np.log2(num_labels)).astype(int)
        primes_argument = self.prime_list[prime_indices]

        p = FastWL.primes(primes_argument)
        # print('p', p)
        logger.info(f"num_labels: {num_labels}")
        log_primes = np.log(p[0:num_labels])
        # transpose
        log_primes = log_primes.reshape([log_primes.shape[0], 1])
        # must ravel the label as the indices of primes.
        # Otherwise the output shape changes with the dimensionality of labels.
        # breakpoint()
        signatures = labels + np.matmul(self.adjacency.astype(np.float32), log_primes[labels.astype(int).ravel() - 1])
        # signatures -= 1
        # label is already shifted 1 to left.
        # No need to shift again. Besides, the returned reverse idx is already python based
        temp, new_labels = np.unique(signatures, return_inverse=True)
        new_labels += 1
        return new_labels

    def equivalent_class(self, labels=None):
        if labels is None:
            labels = np.ones([self.adjacency.shape[0], 1])
        labels = np.atleast_2d(labels)
        labels = labels.reshape(labels.size, 1)

        equivalence_classes = np.zeros_like(labels)
        while not FastWL.is_equivalent(labels, equivalence_classes):
            equivalence_classes = labels
            labels = self.transformation(labels)
        return equivalence_classes.ravel()

    @staticmethod
    def is_equivalent(labels_1: np.ndarray, labels_2: np.ndarray):
        """
            Whether the labeling are equivalent (permutation). Labels start from 1.
        :param labels_1:
        :type labels_1:
        :param labels_2:
        :type labels_2:
        :return:
        :rtype:
        """
        set_label_1 = set(labels_1.ravel())
        set_label_2 = set(labels_2.ravel())
        diff = set_label_1 - set_label_2
        return len(diff) == 0

    @staticmethod
    def primes(n):
        # if n.isdim
        assert np.isscalar(n) and n >= 0
        ub = int(n)
        primes = np.ones([np.ceil(ub / 2).astype(int)])
        num_search_space = primes.shape[0]
        ub = np.ceil(np.sqrt(n)).astype(int)

        for k in range(3, ub, 2):
            if primes[(k + 1) // 2 - 1] > 0:
                primes[((k * k + 1) // 2 - 1):num_search_space:k] = 0
        # python index start from 1 --> output of where +1
        primes = np.where(primes > 0)[0] + 1
        primes = primes * 2 - 1
        if primes.shape[0] > 0:
            primes[0] = 2
        return primes


'''
    matlab implementation:
        labels_1 = np.atleast_1d(labels_1)
        labels_2 = np.atleast_1d(labels_2)
        max_label_1 = max(labels_1).astype(int)
        max_label_2 = max(labels_2).astype(int)
        print("1", labels_1)
        print("2", labels_2)
        print("max_1", max_label_1)
        # num of labels
        if max_label_1 != max_label_2:
            return False

        # result = True
        for i in range(1, max_label_1 + 1):
            y: np.ndarray = np.asarray(labels_2[labels_1 == i])
            print('size', y.size, y)
            if any(y[0] != y):
                return False
'''
