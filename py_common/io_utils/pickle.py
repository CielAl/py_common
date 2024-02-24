import pickle


def write_pickle(fname: str, obj, **kwargs):
    with open(fname, 'wb') as root:
        pickle.dump(obj, root, **kwargs)


def load_pickle(fname: str, **kwargs):
    with open(fname, 'rb') as root:
        return pickle.load(root, **kwargs)

