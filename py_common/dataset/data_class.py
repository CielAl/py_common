from typing import TypedDict, Union,  Any
import torch
import numpy as np
from PIL.Image import Image


class ModelInput(TypedDict):
    data: Union[float, int, np.ndarray, torch.Tensor, Image]
    original: Union[float, int, np.ndarray, torch.Tensor, Image]
    filename: str
    meta: Any
    ground_truth: Union[int, np.ndarray, torch.Tensor]


class ModelOutput(TypedDict):
    loss: torch.Tensor
    logits: torch.Tensor
    ground_truth: torch.Tensor
