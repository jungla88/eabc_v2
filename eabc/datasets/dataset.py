import copy
import os.path as osp
import numpy

class Dataset(object):
    r"""Dataset base class
    """
    
    seed = None
    _rng = numpy.random.default_rng(seed)     
    
    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError
        
    def add_keyVal(self,idx,data):
        raise NotImplementedError


    def __len__(self):
        r"""The number of examples in the dataset."""
        return len(self._indices)

    
        
    def __init__(self, targetObj, idx = None, transform=None, pre_transform=None, seed = None):    

        #Q: are really worth mantain pre_transform or trasform?

        self.transform = transform
        self.pre_transform = pre_transform
        self.idx =idx
        
        
        self._data = []
        self._indices = []
        
        self._rng = numpy.random.default_rng(seed)


        if isinstance(targetObj, str):
            self.path = targetObj
            raw_path, filename = osp.split(osp.abspath(self.path))
            if(not osp.exists(self.path)):
                raise IOError("File not found")
        elif not isinstance(targetObj,(numpy.ndarray,list,tuple,dict)):
            raise TypeError("Can't deal with {}".format(type(targetObj)))
        
        self._process()
            
    @property
    #Return the keys of data object assigned when loading data
    def indices(self):
        return self._indices
    #TODO: check if worth after fresh_dpcopy
    @indices.deleter
    def indices(self):
        self._indices.clear()            
    
    @property
    #Return all the raw data in dataset. It can be specified in the derived dataset. 
    def data(self):
        return list(map(lambda x: x.x, self._data))
    @data.deleter
   #TODO: check if worth after fresh_dpcopy
    def data(self):
        self._data.clear()

        
    #Return all labels in dataset
    @property
    def labels(self):
        return sorted(list(map(lambda x: x.y, self._data)))
    
    @property
    def unique_labels(self):
        return sorted(list(set(map(lambda x: x.y, self._data))))
    
    def to_key(self,idx):
        return self._indices[idx]
    
    def _process(self):
                
        self.process()
        
        if self.pre_transform is not None:            
                self._data = list(map(self.pre_transform, self._data))
                
        self._indices= list(range(len(self._data))) if self.idx is None else self.idx
        
        if self.idx is None:
            self._indices = list(range(len(self._data)))
        else:
            if isinstance(self.idx,numpy.ndarray):
                self._indices = self.idx.tolist()
            
        #debug
        assert(len(self._data)==len(self._indices))
                
        #self._indices= list(range(len(self._data)))
        
        
    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, (int, numpy.int64,numpy.int32)):
            data = self._data[idx] if self.transform is None else self.transform(copy.copy(self._data[idx])) #Need shallow copy?
            return data
        else:
            return self.index_select(idx)
    
    def index_select(self, idx):
        indices = self._indices

        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self._data)
            
            #Select indices in the list because self._indices might not be order (shuffled dataset)
            indices = indices[idx]
            #Unroll the slice
            idx = list(range(start,stop))

        #FIXME: with boolean list return element with index 0 or 1
        elif isinstance(idx, list) or isinstance(idx, tuple):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                'Only integers, slices (`:`), list, tuples,'
                '(got {}).'.format(
                    type(idx).__name__))

        dataset = copy.copy(self)
        dataset._data = [self._data[item] for item in idx]                        
        dataset._indices = indices

        return dataset

    def shuffle(self, return_perm=False):
        r"""Randomly shuffles the examples in the dataset.

        Args:
            return_perm (bool, optional): If set to :obj:`True`, will
                additionally return the random permutation used to shuffle the
                dataset. (default: :obj:`False`)
        """

        perm = (self._rng.permutation(len(self))).tolist()
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset
    
    def fresh_dpcopy(self):
        r""" Create new Dataset object with null data and indices attribute.
        pre_transform() is invalidated and can not be used in the copied dataset.
        
        Return a Dataset with empty data. Only transform and seed is copied from the original Dataset       
        """        
        
        dataset = self.__new__(self.__class__)
        
        dataset.transform = self.transform

        #
        dataset._data = []
        dataset._indices = []     

        dataset._rng = self._rng               
        
        return dataset
