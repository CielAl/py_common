from typing import TypedDict, Any, Union
import torch
import numpy as np


class ModelInput(TypedDict):
    data: Any
    original: Any
    filename: str
    meta: Any
    ground_truth: Union[int, np.ndarray, torch.Tensor]


class ModelOutput(TypedDict):
    loss: torch.Tensor
    logits: torch.Tensor
    ground_truth: torch.Tensor
