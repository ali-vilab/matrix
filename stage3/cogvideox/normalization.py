import numbers
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CogView3PlusAdaLayerNormZeroTextImage(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, dim: int):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 12 * dim, bias=True)
        self.norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)
        normed_x = self.norm_x(x)
        normed_context = self.norm_c(context)
        x = normed_x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        context = normed_context * (1 + c_scale_msa[:, None]) + c_shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, context, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,  # [B, FHW, C]
        timestep: Optional[torch.Tensor] = None,  # [B, F, C]
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        batch_size, fhw, dim = x.shape
        batch_size_t, f, dim2 = temb.shape

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == -1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = temb.chunk(2, dim=-1)
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x).view(batch_size, f, -1, dim) * (1 + scale).unsqueeze(2) + shift.unsqueeze(2)

        return x.view(batch_size, fhw, dim)


class CogVideoXLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, FHW, C]
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,  # [B, F, C]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if len(temb.shape) == 2:
            temb = temb[:, None, :]
        assert len(temb.shape) == 3

        batch_size, fhw, dim = hidden_states.shape
        batch_size_t, f, dim2 = temb.shape
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(
            self.silu(temb)
        ).chunk(6, dim=-1)

        hidden_states = self.norm(hidden_states).view(batch_size, f, -1, dim) * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)
        hidden_states = hidden_states.view(batch_size, fhw, dim)

        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale[:,-1:]) + enc_shift[:,-1:]
        return (
            hidden_states,
            encoder_hidden_states,
            gate,
            enc_gate,
        )

