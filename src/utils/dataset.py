import h5py
import functools
import numpy as np

from torch.utils.data import Dataset

def hdf5_to_dict(hdf5_file, cache_in_mem=False):
    data = {**hdf5_file.attrs}  # 存储文件属性
    for key in hdf5_file.keys():
        if isinstance(hdf5_file[key], h5py.Group):
            data[key] = hdf5_to_dict(hdf5_file[key])
        elif isinstance(hdf5_file[key], h5py.Dataset):
            if cache_in_mem:
                data[key] = hdf5_file[key][:] # Load data to ndarray and save in mem
            else:
                data[key] = hdf5_file[key]   # keep using h5py.Datset, which loads data from disk
    return data

class FileDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.files = {}
    
    @functools.cache
    def load_hdf5(self, path:str, cache_in_mem=False):
        assert path.endswith(".hdf5")
        self.files[path] = h5py.File(path, 'r') # Kept the file open
        return hdf5_to_dict(self.files[path], cache_in_mem) 
    
    def load_npy(self, path:str):
        assert path.endswith(".npy")
        return np.load(path, allow_pickle=True)