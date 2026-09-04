"""High-quality image resampling, used to build and blend the pyramid.

Vendored from ResizeRight (https://github.com/assafshocher/ResizeRight, MIT license, Copyright
(c) 2020 Assaf Shocher), taken via GPNN's copy (https://github.com/WeizmannVision/DropTheGAN),
which flattens upstream's per-method support sizes into the kernel_width table below and differs
from them on two rows: the default is (cubic, 6.0) where upstream's cubic is 4, and lanczos2 is
4.0 where upstream declares 2.

This is third-party code adapted to fit, not ours to redesign. What *is* ours -- the pyramid
geometry and the Laplacian blend that call ``resize`` -- lives in ``efficient_sid/pyramid.py``.

``resize`` is the only name here anything outside this module uses; the interpolation kernels are
internal.
"""

import math

import numpy as np
import torch
from torch import nn


class _Resizer(nn.Module):
    def __init__(self, in_shape, scale_factor=None, output_shape=None,
                 kernel=None, antialiasing=True):
        super().__init__()
        scale_factor, output_shape = self._fix_scale_and_size(
            in_shape, output_shape, scale_factor)

        method, kernel_width = {
            "cubic": (_cubic, 4.0),
            "lanczos2": (_lanczos2, 4.0),
            "lanczos3": (_lanczos3, 6.0),
            "box": (_box, 1.0),
            "linear": (_linear, 2.0),
            None: (_cubic, 6.0),
        }.get(kernel)

        antialiasing *= (np.any(np.array(scale_factor) < 1))

        sorted_dims = np.argsort(np.array(scale_factor))
        self.sorted_dims = [int(dim) for dim in sorted_dims if scale_factor[dim] != 1]

        field_of_view_list, weights_list = [], []
        for dim in self.sorted_dims:
            weights, field_of_view = self._contributions(
                in_shape[dim], output_shape[dim], scale_factor[dim],
                method, kernel_width, antialiasing)
            weights = torch.tensor(weights.T, dtype=torch.float32)
            weights_list.append(
                nn.Parameter(
                    torch.reshape(weights, list(weights.shape) + (len(scale_factor) - 1) * [1]),
                    requires_grad=False))
            field_of_view_list.append(
                nn.Parameter(
                    torch.tensor(field_of_view.T.astype(np.int32), dtype=torch.long),
                    requires_grad=False))

        self.field_of_view = nn.ParameterList(field_of_view_list)
        self.weights = nn.ParameterList(weights_list)
        self.in_shape = in_shape

    def forward(self, in_tensor):
        x = in_tensor
        for dim, fov, w in zip(self.sorted_dims, self.field_of_view, self.weights):
            x = torch.transpose(x, dim, 0)
            x = torch.sum(x[fov] * w, dim=0)
            x = torch.transpose(x, dim, 0)
        return x

    def _fix_scale_and_size(self, input_shape, output_shape, scale_factor):
        if scale_factor is not None:
            if np.isscalar(scale_factor) and len(input_shape) > 1:
                scale_factor = [scale_factor, scale_factor]
            scale_factor = list(scale_factor)
            scale_factor = [1] * (len(input_shape) - len(scale_factor)) + scale_factor
        if output_shape is not None:
            output_shape = list(input_shape[len(output_shape):]) + list(np.uint(np.array(output_shape)))
        if scale_factor is None:
            scale_factor = np.array(output_shape) / np.array(input_shape)
        if output_shape is None:
            output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))
        return scale_factor, output_shape

    def _contributions(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
        fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing and scale < 1.0 else kernel
        kernel_width *= 1.0 / scale if antialiasing and scale < 1.0 else 1.0

        out_coordinates = np.arange(1, out_length + 1)
        shifted_out_coordinates = out_coordinates - (out_length - in_length * scale) / 2
        match_coordinates = shifted_out_coordinates / scale + 0.5 * (1 - 1 / scale)
        left_boundary = np.floor(match_coordinates - kernel_width / 2)
        expanded_kernel_width = np.ceil(kernel_width) + 2

        field_of_view = np.squeeze(
            np.int16(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

        weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)
        sum_weights = np.sum(weights, axis=1)
        sum_weights[sum_weights == 0] = 1.0
        weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

        mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
        field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

        non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
        weights = np.squeeze(weights[:, non_zero_out_pixels])
        field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

        return weights, field_of_view


def resize(image, scale_factor=None, output_shape=None, kernel=None, antialiasing=True):
    """Resize ``image`` with the vendored ResizeRight kernels -- the same resampler the pyramid
    uses, so a resized image stays consistent with pyramid levels built from it.

    ``image`` is expected to have ``[..., H, W]`` shape, where ``...`` means an arbitrary number
    of leading dimensions; only the trailing two are resampled.

    Pass ``output_shape`` (a full shape; the trailing dims are the spatial ones) or
    ``scale_factor``. ``antialiasing`` widens the kernel when downscaling. ``kernel`` selects the
    interpolation: "cubic", "lanczos2", "lanczos3", "box" or "linear". None -- what the pyramid
    uses -- is the cubic kernel.
    """
    resizer = _Resizer(image.shape, scale_factor, output_shape, kernel,
                       antialiasing).to(device=image.device)
    return resizer(image)


def _cubic(x):
    absx = np.abs(x); absx2 = absx ** 2; absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))


def _lanczos2(x):
    return (((np.sin(math.pi * x) * np.sin(math.pi * x / 2) + np.finfo(np.float32).eps) /
             ((math.pi ** 2 * x ** 2 / 2) + np.finfo(np.float32).eps)) * (abs(x) < 2))


def _lanczos3(x):
    return (((np.sin(math.pi * x) * np.sin(math.pi * x / 3) + np.finfo(np.float32).eps) /
             ((math.pi ** 2 * x ** 2 / 3) + np.finfo(np.float32).eps)) * (abs(x) < 3))


def _box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def _linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))
