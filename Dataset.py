import re
import copy
import ntpath
import os.path as osp
from numpy.random import permutation
from inspect import isgenerator


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())


class Dataset(object):
    r"""Dataset base class
    """
    
    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def len(self):
        raise NotImplementedError

    def get(self, idx):
        r"""Gets the data object at index :obj:`idx`."""
        raise NotImplementedError

    def __init__(self, path, transform=None, pre_transform=None):

        self.transform = transform
        self.path = path
        self.pre_transform = pre_transform
        self.__indices__ = None
        self.data = None

        if isinstance(self.path, str):
            raw_path, filename = osp.split(osp.abspath(self.path))
            if(osp.exists(self.path)):
                self._process()
            else:
                raise IOError("File not found")

    def indices(self):
        if self.__indices__ is not None:
            return self.__indices__
        else:
            return range(len(self))

    def _process(self):
        
        self.process()
        
        # if self.pre_transform is not None:
        #         data = map(self.pre_transform(), self.data)
        #         self.data = list(data) if(isgenerator(self.data)) else data
                

    def __len__(self):
        r"""The number of examples in the dataset."""
        if self.__indices__ is not None:
            return len(self.__indices__)
        return self.len()

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, int):
#            data = self.get(self.indices()[idx])
            data = self.get(idx)
            data = data if self.transform is None else self.transform(data)
            return data
        else:
            return self.index_select(idx)
    
    def index_select(self, idx):
        indices = self.indices()

        if isinstance(idx, slice):
            indices = indices[idx]
            ##
            idx = list(range(idx.start,idx.stop))
            ##
        elif isinstance(idx, list) or isinstance(idx, tuple):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                'Only integers, slices (`:`), list, tuples, and long or bool '
                'tensors are valid indices (got {}).'.format(
                    type(idx).__name__))

        dataset = copy.copy(self)
#        dataset.data = [self.get(item) for item in indices]
        dataset.data = [self.get(item) for item in idx]                        
        dataset.__indices__ = indices

        return dataset

    def shuffle(self, return_perm=False):
        r"""Randomly shuffles the examples in the dataset.

        Args:
            return_perm (bool, optional): If set to :obj:`True`, will
                additionally return the random permutation used to shuffle the
                dataset. (default: :obj:`False`)
        """
        perm = (permutation(len(self))).tolist()
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)})'