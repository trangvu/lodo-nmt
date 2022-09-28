"""
This file includes modifications to fairseq distributed through GitHub at https://github.com/pytorch/fairseq/blob/83e615d66905b8ca7483122a37da1a85f13f4b8e/fairseq/models/transformer.py under this license https://github.com/pytorch/fairseq/blob/83e615d66905b8ca7483122a37da1a85f13f4b8e/LICENSE.

Copyright with respect to the modifications: Copyright 2021 Naver Corporation

ORIGINAL COPYRIGHT NOTICE AND PERMISSION NOTICE:

Copyright (c) Facebook, Inc. and its affiliates.

MIT License. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import logging
import re
from fairseq.modules.transformer_layer import TransformerEncoderLayer
from fairseq.modules.transformer_layer import TransformerDecoderLayer

from fairseq.models.transformer import TransformerEncoder
from fairseq.models.transformer import TransformerDecoder
from .modular_transformer import ModularTransformerModel

from fairseq.models.transformer import base_architecture, transformer_iwslt_de_en, transformer_vaswani_wmt_en_de_big
from fairseq.models import register_model, register_model_architecture
from torch import nn
import torch.nn.functional as F
from fairseq.modules.layer_norm import LayerNorm


logger = logging.getLogger(__name__)


class AdapterLayer(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super().__init__()

        self.down = nn.Linear(input_dim, projection_dim)
        self.up = nn.Linear(projection_dim, input_dim)
        self.layer_norm = LayerNorm(input_dim)

        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.layer_norm(x)
        y = self.down(y)
        y = F.relu(y)
        y = self.up(y)
        y = x + y
        return y


class AdapterTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)
        adapter_uids = getattr(args, 'adapter_uids', None) or []
        adapters = []
        for uid in adapter_uids:
            uid, *dim = uid.rsplit(';', maxsplit=1)
            dim = int(dim[0]) if dim else args.adapter_dim
            adapters.append((uid, dim))
        self.adapters = nn.ModuleDict({
            uid: AdapterLayer(args.encoder_embed_dim, dim)
            for uid, dim in adapters if dim > 0
        })

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        for adapter in self.adapters.values():
            x = adapter(x)
        return x


class AdapterTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)
        adapter_uids = (
            getattr(args, 'decoder_adapter_uids', None) or
            getattr(args, 'adapter_uids', None) or
            []
        )
        adapters = []
        for uid in adapter_uids:
            uid, *dim = uid.rsplit(';', maxsplit=1)
            dim = (
                int(dim[0]) if dim else
                getattr(args, 'decoder_adapter_dim', None) or
                args.adapter_dim
            )
            adapters.append((uid, dim))
        self.adapters = nn.ModuleDict({
            uid: AdapterLayer(args.encoder_embed_dim, dim)
            for uid, dim in adapters if dim > 0
        })

    def forward(self, x, *args, **kwargs):
        x, *extra = super().forward(x, *args, **kwargs)
        for adapter in self.adapters.values():
            x = adapter(x)
        return (x, *extra)


class AdapterTransformerEncoder(TransformerEncoder):
    def build_encoder_layer(self, *args, **kwargs):
        return AdapterTransformerEncoderLayer(*args, **kwargs)


class AdapterTransformerDecoder(TransformerDecoder):
    def build_decoder_layer(self, *args, **kwargs):
        return AdapterTransformerDecoderLayer(*args, **kwargs)


@register_model('adapter_transformer')
class AdapterTransformerModel(ModularTransformerModel):
    @classmethod
    def build_encoder(cls, args, *args_, **kwargs):
        return AdapterTransformerEncoder(args, *args_, **kwargs)

    @classmethod
    def build_decoder(cls, args, *args_, **kwargs):
        return AdapterTransformerDecoder(args, *args_, **kwargs)

    def load_state_dict(self, state_dict, strict=False, args=None):
        self.upgrade_state_dict(state_dict)
        status = super().load_state_dict(state_dict, strict=False, args=args)

        if status.missing_keys:
            missing_adapters = {}
            for key in status.missing_keys:
                m = re.match(r'(encoder|decoder).layers.\d+.adapters.[^\.]+', key)
                if m:
                    missing_adapters[m.group(0)] = None
                else:
                    raise ValueError(f'missing parameters in checkpoint: {status.missing_keys}')
            logger.warn(f"missing adapters: {' '.join(missing_adapters.keys())}")


@register_model_architecture('adapter_transformer', 'adapter_transformer')
def adapter_transformer(args):
    base_architecture(args)

@register_model_architecture('adapter_transformer', 'adapter_transformer_iwslt_de_en')
def adapter_transformer_iwslt_de_en(args):
    transformer_iwslt_de_en(args)
    adapter_transformer(args)

@register_model_architecture('adapter_transformer', 'adapter_transformer_vaswani_wmt_en_de_big')
def adapter_transformer_vaswani_wmt_en_de_big(args):
    transformer_vaswani_wmt_en_de_big(args)
    adapter_transformer(args)

@register_model_architecture("transformer", "transformer_mbart_large")
def transformer_mbart_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    transformer_vaswani_wmt_en_de_big(args)
