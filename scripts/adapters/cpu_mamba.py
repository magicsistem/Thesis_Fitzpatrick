"""Small inference-only Mamba-1 reference implementation for CPU.

The module layout matches mamba-ssm 1.0.1 so official state dictionaries can
be loaded unchanged. The selective state-space scan is deliberately written in
plain PyTorch; it is slower than the CUDA kernel but reproducible on CPU.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Mamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        conv_bias: bool = True,
        bias: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("The CPU benchmark only supports full-sequence inference.")
        batch, length, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :length])

        projected = self.x_proj(x.transpose(1, 2))
        dt_low_rank, variable_b, variable_c = torch.split(
            projected, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = torch.einsum("dr,blr->bdl", self.dt_proj.weight, dt_low_rank)
        delta = F.softplus(delta + self.dt_proj.bias.view(1, -1, 1))
        transition = -torch.exp(self.A_log.float())
        state = torch.zeros(
            batch,
            self.d_inner,
            self.d_state,
            device=hidden_states.device,
            dtype=x.dtype,
        )
        outputs = []
        for index in range(length):
            step = delta[:, :, index]
            discrete_a = torch.exp(step.unsqueeze(-1) * transition.unsqueeze(0))
            discrete_b = step.unsqueeze(-1) * variable_b[:, index, :].unsqueeze(1)
            state = state * discrete_a + x[:, :, index].unsqueeze(-1) * discrete_b
            value = torch.sum(state * variable_c[:, index, :].unsqueeze(1), dim=-1)
            value = value + self.D.to(x.dtype).unsqueeze(0) * x[:, :, index]
            outputs.append(value * F.silu(z[:, :, index]))

        scanned = torch.stack(outputs, dim=2).transpose(1, 2)
        return self.out_proj(scanned)
