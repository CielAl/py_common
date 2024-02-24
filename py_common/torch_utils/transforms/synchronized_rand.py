"""Perform identical random transformations two multiple inputs.

Refactor in future for transformsv2.
"""
from torchvision import transforms as tvf
import torch
import torchvision.transforms.functional as F
from PIL.Image import Image as PILImage
from typing import List

TYPE_TENSOR_IMG = torch.Tensor | PILImage


def is_simple_tensor_img(img: TYPE_TENSOR_IMG):
    return isinstance(img, (torch.Tensor, PILImage))

# class SynchronizedTransform(nn.Module):
#     transforms: nn.Module
#
#     def __init__(self, transforms: nn.Module):
#         super().__init__()
#         self.transforms = transforms
#
#     def forward(self, img: TYPE_TENSOR_IMG | List[TYPE_TENSOR_IMG]) -> TYPE_TENSOR_IMG | List[TYPE_TENSOR_IMG]:
#         if isinstance(img, (torch.Tensor, PILImage)):
#             return self.transforms(img)


class SynchronizedRotation90(tvf.RandomRotation):

    def fill_helper(self, img: TYPE_TENSOR_IMG):
        fill = self.fill
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        return fill

    def forward(self, img: TYPE_TENSOR_IMG | List[TYPE_TENSOR_IMG]):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """

        if is_simple_tensor_img(img):
            return super().forward(img)

        assert isinstance(img, List)
        angle = torch.randint(4, (1, )).item() * 90
        out_list = []
        for single_image in img:
            fill = self.fill_helper(single_image)
            rotated_image = F.rotate(single_image, angle, self.interpolation,
                                     self.expand, self.center, fill)
            out_list.append(rotated_image)
        return out_list


class SynchronizedHorizontalFlip(tvf.RandomHorizontalFlip):

    def forward(self, img: TYPE_TENSOR_IMG | List[TYPE_TENSOR_IMG]):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if is_simple_tensor_img(img):
            return super().forward(img)
        assert isinstance(img, List)
        if torch.rand(1) < self.p:
            return [F.hflip(x) for x in img]
        return img


class SynchronizedVerticalFlip(tvf.RandomVerticalFlip):

    def forward(self, img: TYPE_TENSOR_IMG | List[TYPE_TENSOR_IMG]):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if is_simple_tensor_img(img):
            return super().forward(img)
        assert isinstance(img, List)
        if torch.rand(1) < self.p:
            return [F.vflip(x) for x in img]
        return img
