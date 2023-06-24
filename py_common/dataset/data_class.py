"""
Regulate the form of the dataset
"""
from typing import TypedDict, Union,  Any
import torch
import numpy as np
from PIL.Image import Image


class ModelInput(TypedDict):
    """
    data: any data to feed into the model -- must be number, numpy.ndarray, torch.Tensor, or PIL.Image.Image
    original: default 0. A field to retain the original copy of data when necessary if a transformation is performed.
    filename: the uri for the data.
    meta: Any extra information that can be handled by default collate_fn of DataLoader
    ground_truth: labels or any target data. must be number, numpy.ndarray, torch.Tensor.
    """
    data: Union[float, int, np.ndarray, torch.Tensor, Image]
    original: Union[float, int, np.ndarray, torch.Tensor, Image]
    filename: str
    meta: Any
    ground_truth: Union[int, float, np.ndarray, torch.Tensor]


class ModelOutput(TypedDict):
    """
    Loss: output of loss function if available.
    logits: output layer responses.
    ground_truth: corresponding true value.
    """
    loss: torch.Tensor
    logits: torch.Tensor
    ground_truth: torch.Tensor
