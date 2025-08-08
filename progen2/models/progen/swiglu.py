# SwiGLU activation function implementation, from ESM repo
import torch

class SwiGLU(torch.nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2
