"""
Utility functions for PRESTO differential privacy operations.
"""

from typing import Union, List
import numpy as np
import torch

ArrayLike = Union[List, np.ndarray, torch.Tensor]


def numpy_to_list(nd_array: torch.Tensor) -> List[float]:
    """
    Convert a PyTorch tensor to a flattened list.

    Args:
        nd_array: A PyTorch tensor (any shape, assumed to be on CPU or GPU).

    Returns:
        A flattened list of tensor values.
    """
    return nd_array.cpu().numpy().flatten().tolist()


def _to_tensor(data: ArrayLike) -> torch.Tensor:
    """
    Ensures input is a PyTorch tensor. If it's not, converts it using float32 precision.

    Args:
        data: The input data to convert (list, np.ndarray, or torch.Tensor).

    Returns:
        The input as a torch tensor with dtype=float32.
    """
    if not torch.is_tensor(data):
        return torch.as_tensor(data, dtype=torch.float32)
    return data


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
    elif isinstance(reference, np.ndarray):
        return arr
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
