from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from modules import get_sin_pos_embedding
from utils import zero_module
from einops import rearrange, repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.models.attention import FeedForward, AdaLayerNorm, AdaLayerNormZero
from diffusers.models.cross_attention import CrossAttention
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from models.mid_spatial_attention import SpatialAttention

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class BasicTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
            prepend_first_frame: bool = False,
            add_temp_embed: bool = False,
            add_spatial_attn: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        self.add_temp_embed = add_temp_embed
        self.add_spatial_attn = add_spatial_attn

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim
            if only_cross_attention
            else None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )

        # Spatial-Attn
        self.spatial_norm = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim)
        )
        self.spatial_attn = SpatialAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        zero_module(self.spatial_attn.to_out)

        # Temp-Attn
        self.temp_norm = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim)
        )

        self.temp_attn = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        zero_module(self.temp_attn.to_out)

        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

            # 2. Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
        else:
            self.norm2 = None

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            timestep=None,
            attention_mask=None,
            cross_attention_kwargs=None,
            video_length=None,
    ):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if self.temp_attn is not None and isinstance(self.attn1, SparseCausalAttention):
            cross_attention_kwargs.update({"video_length": video_length})
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            # 2. Cross-Attention
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # spatial_attn
        if self.spatial_attn is not None:
            identity = hidden_states
            d = hidden_states.shape[1]
            # normalization
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )
            norm_hidden_states = (
                self.spatial_norm(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.spatial_norm(hidden_states)
            )
            # apply temporal attention
            hidden_states = self.spatial_attn(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
            # ignore effects of temporal layers on image inputs
            if video_length <= 1:
                hidden_states = identity + 0.0 * hidden_states

        # temp_attn
        if self.temp_attn is not None:
            identity = hidden_states
            d = hidden_states.shape[1]
            # add temporal embedding
            if self.add_temp_embed:
                temp_emb = get_sin_pos_embedding(
                    hidden_states.shape[-1], video_length
                ).to(hidden_states)
                hidden_states = rearrange(
                    hidden_states, "(b f) d c -> b d f c", f=video_length
                )
                hidden_states += temp_emb
                hidden_states = rearrange(hidden_states, "b d f c -> (b f) d c")
            # normalization
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )
            norm_hidden_states = (
                self.temp_norm(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.temp_norm(hidden_states)
            )
            # apply temporal attention
            hidden_states = self.temp_attn(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
            # ignore effects of temporal layers on image inputs
            if video_length <= 1:
                hidden_states = identity + 0.0 * hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            patch_size: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            use_linear_projection: bool = False,
            only_cross_attention: bool = False,
            upcast_attention: bool = False,
            norm_type: str = "layer_norm",
            norm_elementwise_affine: bool = True,
            prepend_first_frame: bool = False,
            add_temp_embed: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    prepend_first_frame=prepend_first_frame,
                    add_temp_embed=add_temp_embed,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            timestep=None,
            class_labels=None,
            cross_attention_kwargs=None,
            return_dict: bool = True,
    ):
        # 1. Input
        assert (
                hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(
            encoder_hidden_states, "b n c -> (b f) n c", f=video_length
        )

        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                video_length=video_length,
            )

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class SparseCausalAttention(CrossAttention):
    def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            video_length=None,
            **cross_attention_kwargs,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        video_length = cross_attention_kwargs.get("video_length", 8)
        attention_mask = self.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = self.to_q(hidden_states)
        dim = query.shape[-1]

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.cross_attention_norm:
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        if video_length > 1:
            key = torch.cat(
                [key[:, [0] * video_length], key[:, former_frame_index]], dim=2
            )
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        if video_length > 1:
            value = torch.cat(
                [value[:, [0] * video_length], value[:, former_frame_index]], dim=2
            )
        value = rearrange(value, "b f d c -> (b f) d c")

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        # attention, what we cannot get enough of
        if hasattr(self.processor, "attention_op"):
            hidden_states = xformers.ops.memory_efficient_attention(
                query,
                key,
                value,
                attn_bias=attention_mask,
                op=self.processor.attention_op,
            )
            hidden_states = hidden_states.to(query.dtype)
        elif hasattr(self.processor, "slice_size"):
            batch_size_attention = query.shape[0]
            hidden_states = torch.zeros(
                (batch_size_attention, sequence_length, dim // self.heads),
                device=query.device,
                dtype=query.dtype,
            )
            for i in range(hidden_states.shape[0] // self.processor.slice_size):
                start_idx = i * self.slice_size
                end_idx = (i + 1) * self.slice_size
                query_slice = query[start_idx:end_idx]
                key_slice = key[start_idx:end_idx]
                attn_mask_slice = (
                    attention_mask[start_idx:end_idx]
                    if attention_mask is not None
                    else None
                )
                attn_slice = self.get_attention_scores(
                    query_slice, key_slice, attn_mask_slice
                )
                attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
                hidden_states[start_idx:end_idx] = attn_slice
        else:
            attention_probs = self.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
