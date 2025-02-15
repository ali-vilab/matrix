# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.embeddings import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, TimestepEmbedding, Timesteps

class CogVideoXPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

        if use_positional_embeddings or use_learned_positional_embeddings:
            persistent = use_learned_positional_embeddings
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)

    def _get_positional_embeddings(self, sample_height: int, sample_width: int, sample_frames: int) -> torch.Tensor:
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
        )
        pos_embedding = torch.from_numpy(pos_embedding).flatten(0, 1)
        joint_pos_embedding = torch.zeros(
            1, self.max_text_seq_length + num_patches, self.embed_dim, requires_grad=False
        )
        joint_pos_embedding.data[:, self.max_text_seq_length :].copy_(pos_embedding)

        return joint_pos_embedding

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames)
                pos_embedding = pos_embedding.to(embeds.device, dtype=embeds.dtype)
            else:
                pos_embedding = self.pos_embedding

            embeds = embeds + pos_embedding

        return embeds


class CogView3PlusPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
        text_hidden_size: int = 4096,
        pos_embed_max_size: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.text_hidden_size = text_hidden_size
        self.pos_embed_max_size = pos_embed_max_size
        # Linear projection for image patches
        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)

        # Linear projection for text embeddings
        self.text_proj = nn.Linear(text_hidden_size, hidden_size)

        pos_embed = get_2d_sincos_pos_embed(hidden_size, pos_embed_max_size, base_size=pos_embed_max_size)
        pos_embed = pos_embed.reshape(pos_embed_max_size, pos_embed_max_size, hidden_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float(), persistent=False)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = hidden_states.shape

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Height and width must be divisible by patch size")

        height = height // self.patch_size
        width = width // self.patch_size
        hidden_states = hidden_states.view(batch_size, channel, height, self.patch_size, width, self.patch_size)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).contiguous()
        hidden_states = hidden_states.view(batch_size, height * width, channel * self.patch_size * self.patch_size)

        # Project the patches
        hidden_states = self.proj(hidden_states)
        encoder_hidden_states = self.text_proj(encoder_hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Calculate text_length
        text_length = encoder_hidden_states.shape[1]

        image_pos_embed = self.pos_embed[:height, :width].reshape(height * width, -1)
        text_pos_embed = torch.zeros(
            (text_length, self.hidden_size), dtype=image_pos_embed.dtype, device=image_pos_embed.device
        )
        pos_embed = torch.cat([text_pos_embed, image_pos_embed], dim=0)[None, ...]

        return (hidden_states + pos_embed).to(hidden_states.dtype)

