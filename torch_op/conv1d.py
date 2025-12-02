# implementation of https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
from typing import Annotated

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Conv1d(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    k = 1. / (self.in_channels * self.kernel_size)

    self.weight = nn.Parameter(
      (2*t.rand(self.out_channels, self.in_channels, self.kernel_size) - 1)*np.sqrt(k)
    )

  def forward(self, x: Annotated[Tensor, "batch in_channels length"]) -> Annotated[Tensor, "batch out_channels out_length"]:
    batch, in_channels, length = x.shape

    x_padded = t.zeros(batch, in_channels, length + self.padding * 2)
    x_padded[:, :, self.padding:self.padding+length] = x
    x_stride = x_padded.stride()

    out_length = (length + 2*self.padding - self.kernel_size)//self.stride + 1

    x_s = x_padded.as_strided(size=(batch, in_channels, out_length, self.kernel_size), stride=(x_stride[0], x_stride[1], self.stride, x_stride[-1])) # should be (self.stride, stride[-1])
    x_s = einops.repeat(x_s, "batch c_in out k -> batch c_out c_in out k", c_out=self.out_channels)
    assert x_s.shape == (batch, self.out_channels, self.in_channels, out_length, self.kernel_size)

    w_s = einops.repeat(self.weight, "c_out c_in k -> batch c_out c_in out k", batch=batch, out=out_length)
    assert w_s.shape == (batch, self.out_channels, self.in_channels, out_length, self.kernel_size)

    out = x_s * w_s
    assert out.shape == (batch, self.out_channels, self.in_channels, out_length,  self.kernel_size)

    return einops.reduce(out, "b c_out c_in out k -> b c_out out", "sum")

if __name__ == "__main__":
  layer = Conv1d(2, 3, 2, stride=2)

  x = t.arange(0, 12, dtype=t.float32).reshape(2, 2, -1)
  output = layer(x)
  actual = F.conv1d(x, layer.weight, stride=2)

  print(f"input: {x}")
  print(f"output: {output}")
  print(f"actual: {actual}")

  assert t.isclose(output, actual).all()
