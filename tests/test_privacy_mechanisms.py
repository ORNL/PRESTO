import numpy as np
import torch
import pytest
from ornl_presto import applyDPGaussian, applyDPExponential, applyDPLaplace


def test_applyDPGaussian_basic():
    data = [1.0, 2.0, 3.0]
    priv = applyDPGaussian(data, epsilon=1.0)
    assert len(priv) == len(data)
    assert not np.allclose(priv, data)


def test_applyDPExponential_basic():
    data = [1.0, 2.0, 3.0]
    priv = applyDPExponential(data, epsilon=1.0)
    assert len(priv) == len(data)
    assert not np.allclose(priv, data)


def test_applyDPLaplace_basic():
    data = [1.0, 2.0, 3.0]
    priv = applyDPLaplace(data, epsilon=1.0)
    assert len(priv) == len(data)
    assert not np.allclose(priv, data)
