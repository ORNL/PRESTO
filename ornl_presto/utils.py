"""
Utility functions for PRESTO.
"""
import numpy as np
import torch

def flatten_and_shape(data):
    """
    Flatten input data and return its shape.
    Args:
        data: list, np.ndarray, or torch.Tensor
    Returns:
        tuple: (flattened list, original shape)
    """
    array = np.array(data)
    return array.ravel().tolist(), array.shape

def restore_type_and_shape(reference, flat_list, shape):
    """
    Restore flattened data to the original type and shape.
    Args:
        reference: original data (for type/shape)
        flat_list: flattened data
        shape: target shape
    Returns:
        np.ndarray, list, or torch.Tensor
    """
    arr = np.array(flat_list).reshape(shape)
    if isinstance(reference, torch.Tensor):
        return torch.from_numpy(arr).to(dtype=reference.dtype, device=reference.device)
    return arr.tolist()

def ensure_tensor(obj):
    """
    Convert input to torch.Tensor if not already.
    Args:
        obj: list, np.ndarray, or torch.Tensor
    Returns:
        torch.Tensor
    """
    if not torch.is_tensor(obj):
        return torch.as_tensor(obj, dtype=torch.float32)
    return obj

def fast_walsh_hadamard_transform(tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform the Fast Walsh–Hadamard Transform (FWHT) on a tensor.
    Args:
        tensor: torch.Tensor
    Returns:
        torch.Tensor (transformed)
    """
    h = 1
    y = tensor.clone()
    n = y.numel()
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = y[j]
                v = y[j + h]
                y[j] = u + v
                y[j + h] = u - v
        h *= 2
    return y
