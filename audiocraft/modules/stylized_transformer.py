# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
--- STYLIZED VERSION (3rd Correction) ---
This version fixes the TypeError for multiple keyword argument values.
"""

import typing as tp
import torch
from torch import nn

# Import original classes and functions to inherit from them
from .transformer import (
    StreamingTransformerLayer, StreamingTransformer, StreamingMultiheadAttention,
    LayerScale, RotaryEmbedding, create_sin_embedding, _is_custom, 
    _verify_xformers_internal_compat, _verify_xformers_memory_efficient_compat
)


class StylizedStreamingTransformerLayer(StreamingTransformerLayer):
    """
    --- STYLIZED VERSION of StreamingTransformerLayer ---
    Adds a parallel cross-attention block for an audio style vector.
    """
    def __init__(self, *args, style_cross_attention: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        
        d_model = kwargs.get('d_model')
        num_heads = kwargs.get('num_heads')
        dropout = kwargs.get('dropout', 0.1)
        bias_attn = kwargs.get('bias_attn', True)
        custom = kwargs.get('custom', False)
        memory_efficient = kwargs.get('memory_efficient', False)
        attention_as_float32 = kwargs.get('attention_as_float32', False)
        qk_layer_norm_cross = kwargs.get('qk_layer_norm_cross', False)
        layer_scale = kwargs.get('layer_scale')
        attention_dropout = kwargs.get('attention_dropout')
        device = kwargs.get('device')
        dtype = kwargs.get('dtype')

        factory_kwargs = {'device': device, 'dtype': dtype}
        attn_kwargs: tp.Dict[str, tp.Any] = {
            'embed_dim': d_model,
            'num_heads': num_heads,
            'dropout': dropout if attention_dropout is None else attention_dropout,
            'bias': bias_attn,
            'custom': custom,
            'memory_efficient': memory_efficient,
            'attention_as_float32': attention_as_float32,
        }

        self.style_cross_attention: tp.Optional[nn.Module] = None
        if style_cross_attention:
            self.style_cross_attention = StreamingMultiheadAttention(
                cross_attention=True, qk_layer_norm=qk_layer_norm_cross,
                **attn_kwargs, **factory_kwargs)
            self.dropout_cross_style = nn.Dropout(dropout)
            self.norm_cross_style = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
            self.layer_scale_cross_style: nn.Module = LayerScale(d_model, layer_scale, **factory_kwargs) if layer_scale else nn.Identity()

    def _style_cross_attention_block(self, src: torch.Tensor,
                                     style_cross_attention_src: torch.Tensor) -> torch.Tensor:
        assert self.style_cross_attention is not None
        x = self.style_cross_attention(src, style_cross_attention_src, style_cross_attention_src, need_weights=False)[0]
        return self.dropout_cross_style(x)

    def forward(self, src: torch.Tensor, src_mask: tp.Optional[torch.Tensor] = None,
                src_key_padding_mask: tp.Optional[torch.Tensor] = None,
                cross_attention_src: tp.Optional[torch.Tensor] = None,
                style_cross_attention_src: tp.Optional[torch.Tensor] = None):
        x = src
        if self.norm_first:
            x = x + self.layer_scale_1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask))
            if cross_attention_src is not None:
                x = x + self.layer_scale_cross(self._cross_attention_block(self.norm_cross(x), cross_attention_src))
            if style_cross_attention_src is not None:
                x = x + self.layer_scale_cross_style(self._style_cross_attention_block(self.norm_cross_style(x), style_cross_attention_src))
            x = x + self.layer_scale_2(self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(x + self.layer_scale_1(self._sa_block(x, src_mask, src_key_padding_mask)))
            if cross_attention_src is not None:
                x = self.norm_cross(x + self.layer_scale_cross(self._cross_attention_block(src, cross_attention_src)))
            if style_cross_attention_src is not None:
                x = self.norm_cross_style(x + self.layer_scale_cross_style(self._style_cross_attention_block(src, style_cross_attention_src)))
            x = self.norm2(x + self.layer_scale_2(self._ff_block(x)))
        return x


class StylizedStreamingTransformer(StreamingTransformer):
    """
    --- STYLIZED VERSION of StreamingTransformer (3rd Correction) ---
    Simplifies argument passing to avoid TypeError.
    """
    def __init__(self, d_model: int, num_heads: int, num_layers: int, **kwargs):
        super(StreamingTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.positional_embedding = kwargs.get('positional_embedding', 'sin')
        self.max_period = kwargs.get('max_period', 10_000)
        self.positional_scale = kwargs.get('positional_scale', 1.0)
        self.weight_decay = kwargs.get('weight_decay')
        self.lr = kwargs.get('lr')
        self.checkpointing = kwargs.get('checkpointing', 'none')

        self.rope: tp.Optional[RotaryEmbedding] = None
        if self.positional_embedding in ['rope', 'sin_rope']:
            assert _is_custom(kwargs.get('custom', False), kwargs.get('memory_efficient', False))
            self.rope = RotaryEmbedding(d_model // num_heads, max_period=self.max_period,
                                        xpos=kwargs.get('xpos', False), scale=self.positional_scale, device=kwargs.get('device'))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # --- MODIFIED: Simplified and corrected layer instantiation ---
            # We pass all kwargs through. StylizedStreamingTransformerLayer's __init__ will
            # use the ones it needs and pass the rest to its parent.
            self.layers.append(
                StylizedStreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    **kwargs  # Pass all other kwargs directly
                )
            )

        if self.checkpointing != 'none':
            for layer in self.layers:
                layer._magma_checkpointed = True