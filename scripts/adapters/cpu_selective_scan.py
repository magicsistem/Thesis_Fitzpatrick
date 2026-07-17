"""Plain-PyTorch selective scan compatible with VMamba/Mamba-SSM checkpoints."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """Inference-oriented CPU reference with the Mamba-SSM 1.0.1 signature."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)
    A = A.float()
    B = B.float()
    C = C.float()
    batch, dim, length = u.shape
    d_state = A.shape[1]

    if B.ndim == 4:
        groups = B.shape[1]
        if dim % groups:
            raise ValueError("La dimensión del selective scan no es divisible por sus grupos B.")
        B = B.repeat_interleave(dim // groups, dim=1)
    elif B.ndim == 3:
        B = B.unsqueeze(1).expand(-1, dim, -1, -1)
    else:
        B = B.view(1, dim, d_state, 1).expand(batch, -1, -1, length)

    if C.ndim == 4:
        groups = C.shape[1]
        if dim % groups:
            raise ValueError("La dimensión del selective scan no es divisible por sus grupos C.")
        C = C.repeat_interleave(dim // groups, dim=1)
    elif C.ndim == 3:
        C = C.unsqueeze(1).expand(-1, dim, -1, -1)
    else:
        C = C.view(1, dim, d_state, 1).expand(batch, -1, -1, length)

    state = torch.zeros(batch, dim, d_state, dtype=torch.float32, device=u.device)
    outputs = []
    for index in range(length):
        step = delta[:, :, index]
        transition = torch.exp(step.unsqueeze(-1) * A.unsqueeze(0))
        input_term = step.unsqueeze(-1) * B[:, :, :, index]
        state = transition * state + input_term * u[:, :, index].unsqueeze(-1)
        outputs.append(torch.sum(state * C[:, :, :, index], dim=-1))

    output = torch.stack(outputs, dim=-1)
    if D is not None:
        output = output + u * D.float().view(1, -1, 1)
    if z is not None:
        output = output * F.silu(z.float())
    output = output.to(dtype=dtype_in)
    if return_last_state:
        return output, state
    return output


selective_scan_fn = selective_scan_ref
