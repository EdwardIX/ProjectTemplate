import torch
import numpy as np
import functools

def cache_array(func):
    """
    a decorator to cache functions with array input
    NOTE: the cache is based on value hashing... maybe change to some better methods?
    TODO: use functools lru_cache / cache for implementation
    """
    cache = {}
    sentinel = object()

    @functools.wraps(func)
    def wrapper(array):
        if isinstance(array, np.ndarray):
            # key = (*array.shape, np.sin(array).sum().item(), np.cos(array).sum().item(), array.sum().item(), (array * np.arange(array.size).reshape(array.shape)).mean().item())
            key = (*array.shape, array.dtype, hash(array.tobytes()))
        elif isinstance(array, torch.Tensor):
            # key = (*array.shape, torch.sin(array).sum().item(), torch.cos(array).sum().item(), array.sum().item(), (array * torch.arange(array.numel()).reshape(array.shape)).mean().item())
            key = (*array.shape, array.dtype, hash(array.numpy().tobytes()))
        else:
            raise TypeError("Expected np.ndarray or torch.Tensor, got {}".format(type(array)))
        result = cache.get(key, sentinel)
        if result is sentinel:
            result = func(array)
            cache[key] = result
        return result

    return wrapper

def to_dtype(data, dtype):
    """
    convert all tensor / numpy in data to specific dtype
    """
    if isinstance(data, dict):
        return {k:to_dtype(v, dtype) for k, v in data.items()}
    elif isinstance(data, list):
        return list(to_dtype(v, dtype) for v in data)
    elif isinstance(data, tuple):
        return tuple(to_dtype(v, dtype) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(dtype)
    elif isinstance(data, np.ndarray):
        return data.astype(dtype)
    else:
        raise NotImplementedError(f"Unsupported Data Type: {type(data)}")

def to_backend(data, backend='np'):
    """
    convert all tensor / numpy in data to specific backend
    """
    if isinstance(data, dict):
        return {k:to_backend(v, backend) for k, v in data.items()}
    elif isinstance(data, list):
        return list(to_backend(v, backend) for v in data)
    elif isinstance(data, tuple):
        return tuple(to_backend(v, backend) for v in data)
    else:
        if backend == 'np' or backend == 'numpy':
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            else:
                return np.array(data)
        elif backend == 'torch':
            return torch.as_tensor(data)
        else:
            raise NotImplementedError(f"Unsupported Backend {backend}")

def to_numpy(data):
    return to_backend(data, backend='np')

def meshgrid(*arrays, keepdims=False):
    """
    mesh multiple arrays
    return shape: 
        if keepdims = False: (prod of arrays length, num of arrays) 
        if keepdims = True: (array1.length, ..., arrayn.length, num of arrays)
    """
    if isinstance(arrays[0], torch.Tensor):
        arrays = list(torch.reshape(arr, (-1,)) for arr in arrays)
        arrays = torch.meshgrid(*arrays, indexing='ij')
        if not keepdims:
            arrays = list(torch.reshape(arr, (-1,)) for arr in arrays)
        return torch.stack(arrays, dim=-1)
    else:
        arrays = list(np.reshape(arr, (-1,)) for arr in arrays)
        arrays = np.meshgrid(*arrays, indexing='ij')
        if not keepdims:
            arrays = list(np.reshape(arr, (-1,)) for arr in arrays)
        return np.stack(arrays, axis=-1)

def grid(shape, keepdims=False, use_torch=False):
    if use_torch:
        return meshgrid(*[torch.arange(s) for s in shape], keepdims=keepdims)
    else:
        return meshgrid(*[np.arange(s) for s in shape], keepdims=keepdims)

def stack(data, reduce='none'):
    """
    stack the tree-structured data into arrays
    """
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"can only stack list or tuple objects, but got {type(data)}")
    if len(data) == 0:
        raise ValueError("stack got an empty list")
    
    if isinstance(data[0], dict):
        return {key:stack([d[key] for d in data], reduce) for key in data[0].keys()}
    elif isinstance(data[0], (list, tuple)):
        return [stack([d[key] for d in data], reduce) for key in range(len(data[0]))]
    elif isinstance(data[0], (np.ndarray, int, float, complex)):
        return {
            'mean': np.mean,
            'sum': np.sum,
            'var': np.var,
            'none': np.stack,
        }[reduce](data, axis=0)
    elif isinstance(data[0], torch.Tensor):
        data = torch.stack(data, dim=0)
        return {
            'mean': torch.mean,
            'sum': torch.sum,
            'var': torch.var,
            'none': (lambda d,**_:d),
        }[reduce](data, dim=0)
    else:
        return data
    
def concat(data, reduce='none'):
    """
    concat the tree-structured data into arrays
    """
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"can only concat list or tuple objects, but got {type(data)}")
    if len(data) == 0:
        raise ValueError("concat got an empty list")
    
    if isinstance(data[0], dict):
        return {key:concat([d[key] for d in data], reduce) for key in data[0].keys()}
    elif isinstance(data[0], (list, tuple)):
        return [concat([d[key] for d in data], reduce) for key in range(len(data[0]))]
    elif isinstance(data[0], (int, float, complex)):
        return {
            'mean': np.mean,
            'sum': np.sum,
            'var': np.var,
            'none': np.stack,
        }[reduce](data, axis=0)
    elif isinstance(data[0], np.ndarray):
        data = np.concatenate(data, axis=0)
        return {
            'mean': np.mean,
            'sum': np.sum,
            'var': np.var,
            'none': (lambda d,**_:d),
        }[reduce](data, axis=0)
    elif isinstance(data[0], torch.Tensor):
        data = torch.cat(data, dim=0)
        return {
            'mean': torch.mean,
            'sum': torch.sum,
            'var': torch.var,
            'none': (lambda d,**_:d),
        }[reduce](data, dim=0)
    else:
        return data
    
def lnerr(output, reference, batch_dim=[], p=2, root=True, mean=True, mask=None):
    """
    calculate Lp error (|output - reference|_{L^p}) between output and reference
    if root=False, then calculate |output - reference|_{L^p}^p
    if mean=True: The inputs are view as functions, so sampling points does not affact integral in Lp norm
            False: The inputs are view as vectors, so the dimension does effect norm.
    mask: array of bool type: will ignore the posisions with True value
    """
    assert output.shape == reference.shape, f"Shape of output ({output.shape}) and reference ({reference.shape}) doesn't match"
    assert p > 0
    
    if isinstance(batch_dim, int): 
        batch_dim = [batch_dim]
    op_dim = tuple(sorted(set(range(len(output.shape))) - set(batch_dim + [d+len(output.shape) for d in batch_dim])))
    num = np.prod([output.shape[d] for d in op_dim]) if mean else 1

    if mask is not None:
        assert mask.shape == output.shape, f"Shape of output ({output.shape}) and mask ({mask.shape}) doesn't match"
        if mean: # Need to correct num
            if isinstance(mask, torch.Tensor):
                num = torch.sum(~mask, dim=op_dim)
            elif isinstance(mask, np.ndarray):
                num = np.sum(~mask, axis=op_dim)
            else:
                raise NotImplementedError(f"Unsupported Mask Type {type(mask)}")
    
    if isinstance(output, torch.Tensor) and isinstance(reference, torch.Tensor):
        diff = output - reference
        if mask is not None: diff[mask] = 0
        if p == np.inf:
            return torch.norm(diff, p, dim=op_dim)
        elif root:
            return torch.norm(diff, p, dim=op_dim) / (num ** (1./p))
        else:
            return torch.norm(diff, p, dim=op_dim) ** p / num
    elif isinstance(output, np.ndarray) and isinstance(reference, np.ndarray):
        diff = output - reference
        if mask is not None: diff[mask] = 0
        if p == np.inf:
            return np.linalg.norm(diff, p, axis=op_dim)
        elif root:
            return np.linalg.norm(diff, p, axis=op_dim) / (num ** (1./p))
        else:
            return np.linalg.norm(diff, p, axis=op_dim) ** p / num
    else:
        raise NotImplementedError(f"Unsupported Param Types: {type(output)} and {type(reference)}")

def relerr(output, reference, batch_dim=[], p=2, root=True, mask=None):
    """
    calculate Lp relative error (|output - reference|_{L^p} / |reference|_{L^p}) between output and reference
    if root=False, then caluculate (|output - reference|_{L^p}^p / |reference|_{L^p}^p)
    mask: array of bool type: will ignore the posisions with True value
    """
    assert output.shape == reference.shape, f"Shape of output ({output.shape}) and reference ({reference.shape}) doesn't match"
    if isinstance(batch_dim, int): 
        batch_dim = [batch_dim]
    op_dim = tuple(sorted(set(range(len(output.shape))) - set(batch_dim + [d+len(output.shape) for d in batch_dim])))

    if mask is not None:
        assert mask.shape == output.shape, f"Shape of output ({output.shape}) and mask ({mask.shape}) doesn't match"
        if isinstance(reference, torch.Tensor): # exclude nan in reference
            reference = torch.where(mask, torch.tensor(0.), reference)
        elif isinstance(reference, np.ndarray):
            reference = np.where(mask, 0., reference)
        else:
            raise NotImplementedError(f"Unsupported Reference Type: {type(reference)}")

    if isinstance(output, torch.Tensor) and isinstance(reference, torch.Tensor):
        diff = output - reference
        if mask is not None: diff[mask] = 0
        if p == np.inf or root:
            return torch.norm(diff, p, op_dim) / torch.norm(reference, p, op_dim)
        else:
            return (torch.norm(diff, p, op_dim) / torch.norm(reference, p, op_dim)) ** p
    elif isinstance(output, np.ndarray) and isinstance(reference, np.ndarray):
        diff = output - reference
        if mask is not None: diff[mask] = 0
        if p == np.inf or root:
            return np.linalg.norm(diff, p, op_dim) / np.linalg.norm(reference, p, op_dim)
        else:
            return (np.linalg.norm(diff, p, op_dim) / np.linalg.norm(reference, p, op_dim)) ** p
    else:
        raise NotImplementedError(f"Unsupported Param Types: {type(output)} and {type(reference)}")