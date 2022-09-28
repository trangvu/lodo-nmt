import gc
import logging
import os
import sys
from typing import Optional, Dict, List, Any

from fairseq.distributed import fsdp_wrap
from fairseq.file_io import PathManager
from fairseq.models import  register_model
import torch
from fairseq.models.transformer import TransformerModel, TransformerEncoder, DEFAULT_MIN_PARAMS_TO_WRAP, \
  TransformerDecoder
from fairseq.modules import LayerNorm, TransformerEncoderLayer, TransformerDecoderLayer, FairseqDropout
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from torch import nn, Tensor
import torch.nn.functional as F
from fairseq import utils
from torch.serialization import default_restore_location

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_state_dict_from_pretrained_model(
  pretrained_model_path: str,
) -> Dict[str, Any]:
  if not os.path.exists(pretrained_model_path):
    raise IOError("Model file not found: {}".format(pretrained_model_path))

  with PathManager.open(pretrained_model_path, "rb") as f:
    state_dict = torch.load(
      f, map_location=lambda s, l: default_restore_location(s, "cpu")
    )["model"]

    keys = list(state_dict.keys())
    # Standardize weight name
    for key in keys:
      if 'encoder' in key or 'decoder' in key:
        encoder_offset = key.find('encoder')
        decoder_offset = key.find('decoder')
        if encoder_offset < 0:
          offset = decoder_offset
        elif decoder_offset < 0:
          offset = encoder_offset
        else:
          offset = min(encoder_offset, decoder_offset)
        if offset > 0:
          state_dict[key[offset:]] = state_dict[key]
          del state_dict[key]

  return state_dict

def load_pretrained_model(model, pretrained_path):
  bexists = PathManager.isfile(pretrained_path)
  if bexists:
    with PathManager.open(pretrained_path, "rb") as f:
      state = torch.load(
        f, map_location=lambda s, l: default_restore_location(s, "cpu")
      )
    model.load_state_dict(
      state["model"], strict=True
    )
  else:
    logger.warning("Cannot file checkpoint {}".format(pretrained_path))
    return model
  return model

@register_model("domain_adapter_transformer")
class EncoderDecoderWithDomainAdapter(TransformerModel):
  def __init__(self, args, encoder, decoder):
    super().__init__(args, encoder, decoder)

  @staticmethod
  def add_args(parser):
    """Add model-specific arguments to the parser."""
    TransformerModel.add_args(parser)
    # fmt: off
    parser.add_argument('--pretrained-nmt', type=str, metavar='N', default=None,
                        help='Load model from pretrained NMT')
    parser.add_argument('--add-layer-norm-before', default=False, action='store_true', help="Add layer norm before "
                                                                                            "linear layer")
    parser.add_argument('--residual-before-ln', default=False, action='store_true',
                        help="Residual connection before layer norm")
    parser.add_argument('--down-sample', type=int, default=256,
                        help="Down sample size in the adapter")
    parser.add_argument('--encoder-adapter', action='store_true',
                        help="Add adapter to encoder layer")
    parser.add_argument('--decoder-adapter', action='store_true',
                        help="Add adapter to decoder layer")
    parser.add_argument('--freeze-embedding', action='store_true', default=False,
                        help='Freeze encoder')
    parser.add_argument('--freeze-encoder', action='store_true', default=False,
                        help='Freeze encoder')
    parser.add_argument('--freeze-decoder', action='store_true', default=False,
                        help='Freeze decoder')
    parser.add_argument('--fusion', action='store_true', default=False,
                        help='Add fusion layer')
    parser.add_argument('--fusion-dropout', type=float, default=0.0,
                        help='Fusion dropout')
    parser.add_argument(
      "--adapter-dir",
      type=str,
      metavar="STR",
      help="dir to adaptive sublayers (only applicable to fusion case). Each adative sublayer is stored separately in "
           "subfolder with same name of the domain",
    )
  @classmethod
  def build_model(cls, args, task):
    """Build a new model instance."""
    model = super().build_model(args, task)
    
    if task.training:
      if hasattr(args, 'pretrained_nmt') and args.pretrained_nmt:
        model_checkpoint = args.pretrained_nmt
        logger.info("Load NMT models from {}".format(model_checkpoint))
        # model = load_pretrained_model(model, model_checkpoint)
        pretrained_model = load_state_dict_from_pretrained_model(model_checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_model, strict=False)

        logger.info(
          "Missing parameters: {}".format('\n * '.join(missing_keys))
        )

        logger.info(
          "Unexpected parameters: {}".format('\n *'.join(unexpected_keys))
        )

    def freeze_module_params(m):
      if m is not None:
        for p in m.parameters():
          p.requires_grad = False

    if args.freeze_encoder:
      logger.info("Freeze encoder")
      freeze_module_params(model.encoder)

    if args.freeze_decoder:
      logger.info("Freeze decoder")
      freeze_module_params(model.decoder)

    if args.fusion:
      domains = getattr(args, "domains", "")
      domains = domains.split(',')
      if args.encoder_adapter:
        for domain in domains:
          model.encoder.register_adapter(domain)
        if args.freeze_encoder:
          logger.info("Freeze encoder")
          freeze_module_params(model.encoder)
        model.encoder.add_fusion_layer()

      if args.decoder_adapter:
        for domain in domains:
          model.decoder.register_adapter(domain)
        if args.freeze_decoder:
          logger.info("Freeze decoder")
          freeze_module_params(model.decoder)
        model.decoder.add_fusion_layer()

      # load adapter layers from adapter dir
      if task.training:
        for domain in domains:
          assert hasattr(args, "adapter_dir"), (
            "You must specify a path for --adapter-dir to train fusion layer"
          )
          adapt_layer_path = os.path.join(args.adapter_dir, domain, "adapter_sublayers.pt")
          save_adapt_layer = torch.load(adapt_layer_path, map_location=torch.device('cpu'))
          logger.info(f"Load adapter for domain {domain} from {adapt_layer_path}")
          encoder_adapt_layer = {}
          decoder_adapt_layer = {}
          for key in save_adapt_layer:
            if key.startswith('encoder'):
              new_key = key.replace('encoder.layers.','')
              encoder_adapt_layer[new_key] = save_adapt_layer[key]
            elif key.startswith('decoder'):
              new_key = key.replace('decoder.layers.', '')
              decoder_adapt_layer[new_key] = save_adapt_layer[key]
          if args.encoder_adapter:
            assert len(encoder_adapt_layer) > 0, (
              f"encoder adapter sublayer not found for domain {domain}"
            )
            model.encoder.load_adapter_layer(domain, encoder_adapt_layer)
          if args.decoder_adapter:
            assert len(decoder_adapt_layer) > 0, (
              f"decoder adapter sublayer not found for domain {domain}"
            )
            model.decoder.load_adapter_layer(domain, decoder_adapt_layer)

    else:
      if args.encoder_adapter:
          model.encoder.register_adapter('default')

      if args.decoder_adapter:
          model.decoder.register_adapter('default')
    return model

  @classmethod
  def build_encoder(cls, args, src_dict, embed_tokens):
    return TransformerEncoderWithAdapter(args, src_dict, embed_tokens)

  @classmethod
  def build_decoder(cls, args, tgt_dict, embed_tokens):
    return TransformerDecoderWithAdapter(
      args,
      tgt_dict,
      embed_tokens,
      no_encoder_attn=getattr(args, "no_cross_attention", False),
    )

  def forward(self, src_tokens, src_lengths, prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        adapter_mask=None):

    # anything in original adapter_mask = 1, becomes -1e8
    # anything in original adapter_mask = 0, becomes 0
    # Note that we cannot use -inf here, because at some edge cases,
    # the attention weight (before softmax) for some padded element in query
    # will become -inf, which results in NaN in model parameters
    if adapter_mask is not None:
      adapter_mask = adapter_mask.masked_fill(adapter_mask.to(torch.bool), -1e8)

    encoder_out = self.encoder(
      src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, adapter_mask=adapter_mask
    )
    fusion_scores = encoder_out['fusion_scores']
    decoder_out = self.decoder(
      prev_output_tokens,
      encoder_out=encoder_out,
      features_only=features_only,
      alignment_layer=alignment_layer,
      alignment_heads=alignment_heads,
      src_lengths=src_lengths,
      return_all_hiddens=return_all_hiddens,
      adapter_mask=adapter_mask
    )
    fusion_scores.extend(decoder_out[1]['fusion_scores'])
    decoder_out[1].update(fusion_scores=fusion_scores)
    return decoder_out

  def get_fusion_regularization_loss(self):
    fusion = getattr(self.args, 'fusion', False)
    if not fusion:
      return 0
    reg_loss = 0.0
    if self.args.encoder_adapter:
      reg_loss += self.encoder.get_fusion_regularization_loss()
    if self.args.decoder_adapter:
      reg_loss += self.decoder.get_fusion_regularization_loss()
    return reg_loss
  
class TransformerEncoderWithAdapter(TransformerEncoder):
  def __init__(self, args, dictionary, embed_tokens):
    super(TransformerEncoderWithAdapter, self).__init__(args, dictionary, embed_tokens)

  def build_encoder_layer(self, args):
    layer = TransformerEncoderLayerWithAdapter(args)
    checkpoint = getattr(args, "checkpoint_activations", False)
    if checkpoint:
      offload_to_cpu = getattr(args, "offload_activations", False)
      layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
    # if we are checkpointing, enforce that FSDP always wraps the
    # checkpointed layer, regardless of layer size
    min_params_to_wrap = (
      getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
      if not checkpoint
      else 0
    )
    layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
    return layer

  def register_adapter(self, name):
    for encoder_layer in self.layers:
      encoder_layer.register_adapter(name)

  def load_adapter_layer(self, name, state_dict):
    layer_state_dicts = {}
    for idx, encoder_layer in enumerate(self.layers):
      layer_state_dicts[str(idx)] = {}

    for key in state_dict:
      layer_idx = key.split('.')[0]
      new_key = key.replace(f"{layer_idx}.adapters.default.", "")
      layer_state_dicts[layer_idx][new_key] = state_dict[key]

    for idx, encoder_layer in enumerate(self.layers):
      encoder_layer.load_adapter_layer(name, layer_state_dicts[str(idx)])

  def add_fusion_layer(self):
    for encoder_layer in self.layers:
      encoder_layer.add_fusion_layer()

  def get_fusion_regularization_loss(self):
    reg_loss = 0.0
    for layer in self.layers:
        reg_loss += layer.get_fusion_regularization_loss()
    return reg_loss
  
  def forward_scriptable(
    self,
    src_tokens,
    src_lengths: Optional[torch.Tensor] = None,
    return_all_hiddens: bool = False,
    token_embeddings: Optional[torch.Tensor] = None,
    adapter_mask : Optional[torch.Tensor] = None,
  ):
    """
    Args:
        src_tokens (LongTensor): tokens in the source language of shape
            `(batch, src_len)`
        src_lengths (torch.LongTensor): lengths of each source sentence of
            shape `(batch)`
        return_all_hiddens (bool, optional): also return all of the
            intermediate hidden states (default: False).
        token_embeddings (torch.Tensor, optional): precomputed embeddings
            default `None` will recompute embeddings

    Returns:
        dict:
            - **encoder_out** (Tensor): the last encoder layer's output of
              shape `(src_len, batch, embed_dim)`
            - **encoder_padding_mask** (ByteTensor): the positions of
              padding elements of shape `(batch, src_len)`
            - **encoder_embedding** (Tensor): the (scaled) embedding lookup
              of shape `(batch, src_len, embed_dim)`
            - **encoder_states** (List[Tensor]): all intermediate
              hidden states of shape `(src_len, batch, embed_dim)`.
              Only populated if *return_all_hiddens* is True.
    """
    # compute padding mask
    encoder_padding_mask = src_tokens.eq(self.padding_idx)
    has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

    x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

    # account for padding while computing the representation
    if has_pads:
      x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    encoder_states = []

    if return_all_hiddens:
      encoder_states.append(x)

    # encoder layers
    fusion_scores = []
    for layer in self.layers:
      x, fusion_score = layer(
        x, encoder_padding_mask=encoder_padding_mask if has_pads else None,
        adapter_mask = adapter_mask
      )
      if fusion_score is not None:
        fusion_scores.append(fusion_score.transpose(1,0))
      if return_all_hiddens:
        assert encoder_states is not None
        encoder_states.append(x)

    if self.layer_norm is not None:
      x = self.layer_norm(x)

    # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
    # `forward` so we use a dictionary instead.
    # TorchScript does not support mixed values so the values are all lists.
    # The empty list is equivalent to None.
    return {
      "encoder_out": [x],  # T x B x C
      "encoder_padding_mask": [encoder_padding_mask],  # B x T
      "encoder_embedding": [encoder_embedding],  # B x T x C
      "encoder_states": encoder_states,  # List[T x B x C]
      "src_tokens": [],
      "src_lengths": [],
      "fusion_scores": fusion_scores,
    }

class Adapter(nn.Module):
  def __init__(self, input_size, down_sample=None, activation_fn="relu", add_layer_norm_before=True,
        residual_before_ln=True, q_noise=0.0, qn_block_size=8):
    
    super(Adapter, self).__init__()
    self.input_size = input_size
    self.add_layer_norm_before = add_layer_norm_before
    self.residual_before_ln = residual_before_ln

    self.down_sample = down_sample
    if down_sample is None:
      self.down_sample = self.input_size // 2

    self.layer_norm = LayerNorm(self.input_size)

    # Linear down projection
    self.fc1 = self.build_fc(self.input_size, self.down_sample, q_noise, qn_block_size)
    self.activation_fn = utils.get_activation_fn(activation=activation_fn)

    # Linear up projection
    self.fc2 = self.build_fc(self.down_sample, self.input_size, q_noise, qn_block_size)
    self.apply(init_bert_params)

  def build_fc(self,input_dim, output_dim, q_noise, qn_block_size):
    return quant_noise(
        nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
    )

  def forward(self, x):
    residual_input = x
    if self.add_layer_norm_before:
      x = self.layer_norm(x)
    x = self.fc1(x)
    x = self.activation_fn(x)

    x = self.fc2(x)
    # x = self.activation_fn(x)
    if self.residual_before_ln:
      x = x + residual_input
    if not self.add_layer_norm_before:
      x = self.layer_norm(x)

    if not self.residual_before_ln:
      x = x + residual_input
    return x

class Fusion(nn.Module):
  def __init__(self, embed_dim,
        kdim=None, vdim=None, dropout=0.1,
        q_noise=0.0,
        qn_block_size=8, residual_before=False, temperature=None):

    super(Fusion, self).__init__()
    self.embed_dim = embed_dim
    self.kdim = kdim if kdim is not None else embed_dim
    self.vdim = vdim if vdim is not None else embed_dim
    self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
    self.dropout_module = FairseqDropout(
      dropout, module_name=self.__class__.__name__
    )

    self.residual_before = residual_before

    self.k_proj = quant_noise(
      nn.Linear(self.kdim, embed_dim), q_noise, qn_block_size
    )
    self.v_proj = quant_noise(
      nn.Linear(self.vdim, embed_dim), q_noise, qn_block_size
    )
    self.q_proj = quant_noise(
      nn.Linear(embed_dim, embed_dim, bias=False), q_noise, qn_block_size
    )

    if temperature:
      self.T = 50.0
    else:
      self.T = 1.0
    self.reduction = self.T / 1000.0

    self.apply(init_bert_params)
    self.v_proj.weight.data = (torch.zeros(self.vdim, embed_dim) + 0.000001).fill_diagonal_(1.0)
    
  def get_fusion_regularization_loss(self):
    target = torch.zeros((self.vdim, self.embed_dim)).fill_diagonal_(1.0).to(self.v_proj.weight.device)
    reg_loss = (target - self.v_proj.weight).pow(2).sum()
    return reg_loss

  def forward(self, query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        adapter_mask: Optional[Tensor] = None):
    q = self.q_proj(query)
    k = self.k_proj(key)
    v = self.v_proj(value)
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.squeeze(torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)), dim=2)

    if adapter_mask is not None:
      adapter_mask = adapter_mask.unsqueeze(0)
      adapter_mask = adapter_mask.repeat(attention_scores.size(0), 1, 1)
      attention_scores += adapter_mask

    attention_scores = self.dropout_module(attention_scores)

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)
    self.T = max(self.T - self.reduction, 1.0)
    if not self.training:
      self.recent_attention = attention_probs.detach().cpu().numpy()

    context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), v), dim=2)
    return context_layer, attention_scores

class TransformerEncoderLayerWithAdapter(TransformerEncoderLayer):
  def __init__(self, args):
    super().__init__(args)
    self.adapters = nn.ModuleDict()
    self.fusion_layer = None
    self.adapter_list = []
    self.adapter_list_reverse = {}

  def register_adapter(self, name):
    if name in self.adapters:
        logger.warning(f"Adapter {name} is already registered")
    self.adapters[name] = self.build_adapter(self.args)
    self.adapter_list.append(name)
    self.adapter_list_reverse[len(self.adapter_list)] = name

  def load_adapter_layer(self, name, state_dict):
    self.adapters[name].load_state_dict(state_dict)

  def add_fusion_layer(self):
    self.fusion_layer = Fusion(self.args.decoder_embed_dim, dropout=self.args.fusion_dropout,
                               q_noise=self.args.quant_noise_pq, qn_block_size=self.args.quant_noise_pq_block_size
                               )
    
  def get_fusion_regularization_loss(self):
    reg_loss = 0.0
    if self.fusion_layer:
        reg_loss = self.fusion_layer.get_fusion_regularization_loss()
    return reg_loss

  def build_adapter(self, args):
    return Adapter(args.decoder_embed_dim, args.down_sample, args.activation_fn,
                   add_layer_norm_before=args.add_layer_norm_before,
                   residual_before_ln=args.residual_before_ln,
                   q_noise=args.quant_noise_pq, qn_block_size=args.quant_noise_pq_block_size)

  def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None,
              adapter_mask: Optional[Tensor] = None):
    """
    Args:
        x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
        encoder_padding_mask (ByteTensor): binary ByteTensor of shape
            `(batch, seq_len)` where padding elements are indicated by ``1``.
        attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
            where `tgt_len` is the length of output and `src_len` is the
            length of input, though here both are equal to `seq_len`.
            `attn_mask[tgt_i, src_j] = 1` means that when calculating the
            embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
            useful for strided self-attention.

    Returns:
        encoded output of shape `(seq_len, batch, embed_dim)`
    """
    # anything in original attn_mask = 1, becomes -1e8
    # anything in original attn_mask = 0, becomes 0
    # Note that we cannot use -inf here, because at some edge cases,
    # the attention weight (before softmax) for some padded element in query
    # will become -inf, which results in NaN in model parameters
    if attn_mask is not None:
      attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

    residual = x
    if self.normalize_before:
      x = self.self_attn_layer_norm(x)
    x, _ = self.self_attn(
      query=x,
      key=x,
      value=x,
      key_padding_mask=encoder_padding_mask,
      need_weights=False,
      attn_mask=attn_mask,
    )
    x = self.dropout_module(x)
    x = self.residual_connection(x, residual)
    if not self.normalize_before:
      x = self.self_attn_layer_norm(x)

    residual = x
    if self.normalize_before:
      x = self.final_layer_norm(x)
    x = self.activation_fn(self.fc1(x))
    x = self.activation_dropout_module(x)
    x = self.fc2(x)
    x = self.dropout_module(x)
    x = self.residual_connection(x, residual)
    if not self.normalize_before:
      x = self.final_layer_norm(x)
    fusion_score = None
    if self.args.encoder_adapter:
      if self.fusion_layer is not None:
        # logger.info("Perform fusion")
        output = []
        for adapter_name in self.adapter_list:
          output.append(self.adapters[adapter_name](x))
        output = torch.stack(output, dim=2)
        x, fusion_score = self.fusion_layer(x, output, output, adapter_mask)
      else:
        assert len(self.adapter_list) == 1
        x = self.adapters[self.adapter_list[0]](x)
    return x, fusion_score

class TransformerDecoderWithAdapter(TransformerDecoder):
  def build_decoder_layer(self, args, no_encoder_attn=False):
    layer = TransformerDecoderLayerWithAdapter(args, no_encoder_attn)
    checkpoint = getattr(args, "checkpoint_activations", False)
    if checkpoint:
      offload_to_cpu = getattr(args, "offload_activations", False)
      layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
    # if we are checkpointing, enforce that FSDP always wraps the
    # checkpointed layer, regardless of layer size
    min_params_to_wrap = (
      getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
      if not checkpoint
      else 0
    )
    layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
    return layer

  def register_adapter(self, name):
    for decoder_layer in self.layers:
      decoder_layer.register_adapter(name)

  def load_adapter_layer(self, name, state_dict):
    layer_state_dicts = {}
    for idx, encoder_layer in enumerate(self.layers):
      layer_state_dicts[str(idx)] = {}

    for key in state_dict:
      layer_idx = key.split('.')[0]
      new_key = key.replace(f"{layer_idx}.adapters.default.", "")
      layer_state_dicts[layer_idx][new_key] = state_dict[key]

    for idx, decoder_layer in enumerate(self.layers):
      decoder_layer.load_adapter_layer(name, layer_state_dicts[str(idx)])

  def add_fusion_layer(self):
    for decoder_layer in self.layers:
      decoder_layer.add_fusion_layer()
      
  def get_fusion_regularization_loss(self):
    reg_loss = 0.0
    for layer in self.layers:
        reg_loss += layer.get_fusion_regularization_loss()
    return reg_loss

  def extract_features_scriptable(
    self,
    prev_output_tokens,
    encoder_out: Optional[Dict[str, List[Tensor]]],
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    full_context_alignment: bool = False,
    alignment_layer: Optional[int] = None,
    alignment_heads: Optional[int] = None,
    adapter_mask: Optional[torch.Tensor] = None,
  ):
    """
    Similar to *forward* but only return features.

    Includes several features from "Jointly Learning to Align and
    Translate with Transformer Models" (Garg et al., EMNLP 2019).

    Args:
        full_context_alignment (bool, optional): don't apply
            auto-regressive mask to self-attention (default: False).
        alignment_layer (int, optional): return mean alignment over
            heads at this layer (default: last layer).
        alignment_heads (int, optional): only average alignment over
            this many heads (default: all heads).

    Returns:
        tuple:
            - the decoder's features of shape `(batch, tgt_len, embed_dim)`
            - a dictionary with any model-specific outputs
    """
    bs, slen = prev_output_tokens.size()
    if alignment_layer is None:
      alignment_layer = self.num_layers - 1

    enc: Optional[Tensor] = None
    padding_mask: Optional[Tensor] = None
    if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
      enc = encoder_out["encoder_out"][0]
      assert (
        enc.size()[1] == bs
      ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
    if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
      padding_mask = encoder_out["encoder_padding_mask"][0]

    # embed positions
    positions = None
    if self.embed_positions is not None:
      positions = self.embed_positions(
        prev_output_tokens, incremental_state=incremental_state
      )

    if incremental_state is not None:
      prev_output_tokens = prev_output_tokens[:, -1:]
      if positions is not None:
        positions = positions[:, -1:]

    # embed tokens and positions
    x = self.embed_scale * self.embed_tokens(prev_output_tokens)

    if self.quant_noise is not None:
      x = self.quant_noise(x)

    if self.project_in_dim is not None:
      x = self.project_in_dim(x)

    if positions is not None:
      x += positions

    if self.layernorm_embedding is not None:
      x = self.layernorm_embedding(x)

    x = self.dropout_module(x)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    self_attn_padding_mask: Optional[Tensor] = None
    if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
      self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

    # decoder layers
    attn: Optional[Tensor] = None
    inner_states: List[Optional[Tensor]] = [x]
    fusion_scores = []
    for idx, layer in enumerate(self.layers):
      if incremental_state is None and not full_context_alignment:
        self_attn_mask = self.buffered_future_mask(x)
      else:
        self_attn_mask = None

      x, layer_attn, self_attn_state, fusion_score = layer(
        x,
        enc,
        padding_mask,
        incremental_state,
        self_attn_mask=self_attn_mask,
        self_attn_padding_mask=self_attn_padding_mask,
        need_attn=bool((idx == alignment_layer)),
        need_head_weights=bool((idx == alignment_layer)),
        adapter_mask = adapter_mask
      )
      if fusion_score is not None:
        fusion_scores.append(fusion_score.transpose(1,0))
      inner_states.append(x)
      if layer_attn is not None and idx == alignment_layer:
        attn = layer_attn.float().to(x)

    if attn is not None:
      if alignment_heads is not None:
        attn = attn[:alignment_heads]

      # average probabilities over heads
      attn = attn.mean(dim=0)

    if self.layer_norm is not None:
      x = self.layer_norm(x)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if self.project_out_dim is not None:
      x = self.project_out_dim(x)

    return x, {"attn": [attn], "inner_states": inner_states, "fusion_scores": fusion_scores}


class TransformerDecoderLayerWithAdapter(TransformerDecoderLayer):
  def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
    super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
    self.args = args
    self.adapters = nn.ModuleDict()
    self.adapter_list = []
    self.adapter_list_reverse = {}
    self.fusion_layer = None

  def register_adapter(self, name):
    if name in self.adapters:
        logger.warning(f"Adapter {name} is already registered")
    self.adapters[name] = self.build_adapter(self.args)
    self.adapter_list.append(name)
    self.adapter_list_reverse[len(self.adapter_list)] = name

  def load_adapter_layer(self, name, state_dict):
    self.adapters[name].load_state_dict(state_dict)

  def add_fusion_layer(self):
    self.fusion_layer = Fusion(self.args.decoder_embed_dim, dropout=self.args.fusion_dropout,
                        q_noise = self.args.quant_noise_pq, qn_block_size = self.args.quant_noise_pq_block_size
    )

  def get_fusion_regularization_loss(self):
    reg_loss = 0.0
    if self.fusion_layer:
        reg_loss = self.fusion_layer.get_fusion_regularization_loss()
    return reg_loss
  
  def build_adapter(self, args):
    return Adapter(args.decoder_embed_dim, args.down_sample, args.activation_fn,
                   add_layer_norm_before=args.add_layer_norm_before,
                   residual_before_ln=args.residual_before_ln,
                   q_noise=args.quant_noise_pq, qn_block_size=args.quant_noise_pq_block_size)

  def forward(
    self,
    x,
    encoder_out: Optional[torch.Tensor] = None,
    encoder_padding_mask: Optional[torch.Tensor] = None,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    prev_self_attn_state: Optional[List[torch.Tensor]] = None,
    prev_attn_state: Optional[List[torch.Tensor]] = None,
    self_attn_mask: Optional[torch.Tensor] = None,
    self_attn_padding_mask: Optional[torch.Tensor] = None,
    need_attn: bool = False,
    need_head_weights: bool = False,
    adapter_mask: Optional[torch.Tensor] = None,
  ):
    """
    Args:
        x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
        encoder_padding_mask (ByteTensor, optional): binary
            ByteTensor of shape `(batch, src_len)` where padding
            elements are indicated by ``1``.
        need_attn (bool, optional): return attention weights
        need_head_weights (bool, optional): return attention weights
            for each head (default: return average over heads).

    Returns:
        encoded output of shape `(seq_len, batch, embed_dim)`
    """
    if need_head_weights:
      need_attn = True

    residual = x
    if self.normalize_before:
      x = self.self_attn_layer_norm(x)
    if prev_self_attn_state is not None:
      prev_key, prev_value = prev_self_attn_state[:2]
      saved_state: Dict[str, Optional[Tensor]] = {
        "prev_key": prev_key,
        "prev_value": prev_value,
      }
      if len(prev_self_attn_state) >= 3:
        saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
      assert incremental_state is not None
      self.self_attn._set_input_buffer(incremental_state, saved_state)
    _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
    if self.cross_self_attention and not (
      incremental_state is not None
      and _self_attn_input_buffer is not None
      and "prev_key" in _self_attn_input_buffer
    ):
      if self_attn_mask is not None:
        assert encoder_out is not None
        self_attn_mask = torch.cat(
          (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
        )
      if self_attn_padding_mask is not None:
        if encoder_padding_mask is None:
          assert encoder_out is not None
          encoder_padding_mask = self_attn_padding_mask.new_zeros(
            encoder_out.size(1), encoder_out.size(0)
          )
        self_attn_padding_mask = torch.cat(
          (encoder_padding_mask, self_attn_padding_mask), dim=1
        )
      assert encoder_out is not None
      y = torch.cat((encoder_out, x), dim=0)
    else:
      y = x

    x, attn = self.self_attn(
      query=x,
      key=y,
      value=y,
      key_padding_mask=self_attn_padding_mask,
      incremental_state=incremental_state,
      need_weights=False,
      attn_mask=self_attn_mask,
    )
    x = self.dropout_module(x)
    x = self.residual_connection(x, residual)
    if not self.normalize_before:
      x = self.self_attn_layer_norm(x)

    if self.encoder_attn is not None and encoder_out is not None:
      residual = x
      if self.normalize_before:
        x = self.encoder_attn_layer_norm(x)
      if prev_attn_state is not None:
        prev_key, prev_value = prev_attn_state[:2]
        saved_state: Dict[str, Optional[Tensor]] = {
          "prev_key": prev_key,
          "prev_value": prev_value,
        }
        if len(prev_attn_state) >= 3:
          saved_state["prev_key_padding_mask"] = prev_attn_state[2]
        assert incremental_state is not None
        self.encoder_attn._set_input_buffer(incremental_state, saved_state)

      x, attn = self.encoder_attn(
        query=x,
        key=encoder_out,
        value=encoder_out,
        key_padding_mask=encoder_padding_mask,
        incremental_state=incremental_state,
        static_kv=True,
        need_weights=need_attn or (not self.training and self.need_attn),
        need_head_weights=need_head_weights,
      )
      x = self.dropout_module(x)
      x = self.residual_connection(x, residual)
      if not self.normalize_before:
        x = self.encoder_attn_layer_norm(x)

    residual = x
    if self.normalize_before:
      x = self.final_layer_norm(x)

    x = self.activation_fn(self.fc1(x))
    x = self.activation_dropout_module(x)
    x = self.fc2(x)
    x = self.dropout_module(x)
    x = self.residual_connection(x, residual)
    if not self.normalize_before:
      x = self.final_layer_norm(x)
    fusion_score = None
    if self.args.decoder_adapter:
      if self.fusion_layer is not None:
        # logger.info("Perform fusion")
        output = []
        for adapter_name in self.adapter_list:
          output.append(self.adapters[adapter_name](x))
        output = torch.stack(output, dim=2)
        x, fusion_score = self.fusion_layer(x, output, output, adapter_mask)
      else:
        assert len(self.adapter_list) == 1
        x = self.adapters[self.adapter_list[0]](x)

    if self.onnx_trace and incremental_state is not None:
      saved_state = self.self_attn._get_input_buffer(incremental_state)
      assert saved_state is not None
      if self_attn_padding_mask is not None:
        self_attn_state = [
          saved_state["prev_key"],
          saved_state["prev_value"],
          saved_state["prev_key_padding_mask"],
        ]
      else:
        self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
      return x, attn, self_attn_state, fusion_score
    return x, attn, None, fusion_score