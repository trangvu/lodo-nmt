import logging
import sys
from typing import Optional

from fairseq.file_io import PathManager
from fairseq.models import BaseFairseqModel, register_model
import torch
from fairseq.models.transformer import TransformerModel
from torch import nn
from torch.autograd import Function

from fairseq import utils
from torch.serialization import default_restore_location

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


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

@register_model("domain_aware_transformer")
class EncoderDecoderWithDomainDisciriminator(BaseFairseqModel):
  def __init__(self, args, nmt):
    super().__init__()
    self.args = args
    self.nmt = nmt
    # Init domain classifier
    src_inner_dim = self.args.encoder_embed_dim
    self.gradrev = None
    gradient_reverse = getattr(self.args, "gradient_reverse", False)
    if gradient_reverse:
      self.gradrev = GradientReversal()

    self.src_domain_discriminator = FeedForwadLayer(src_inner_dim, args.num_domain,
                                                    utils.get_activation_fn(self.args.discriminator_activation_fn),
                                                    self.args.discriminator_dropout)

  @property
  def encoder(self):
    return self.nmt.encoder

  @property
  def decoder(self):
    return self.nmt.decoder

  @staticmethod
  def add_args(parser):
    """Add model-specific arguments to the parser."""
    TransformerModel.add_args(parser)
    # fmt: off
    parser.add_argument( "--num-domain", type=int, help="Num domains")
    parser.add_argument('--discriminator-activation-fn', default='gelu',
                        choices=utils.get_available_activation_fns(),
                        help='activation function to use')
    parser.add_argument('--discriminator-dropout', type=float, metavar='D', default=0.1,
                        help='dropout probability in the masked_lm discriminator layers')
    parser.add_argument('--discriminator-hidden-size', type=int, metavar='N',
                        help='hidden size of discriminator layers')
    parser.add_argument('--discriminator-layers', type=int, metavar='N', default=2,
                        help='num discriminator layers')
    parser.add_argument('--pretrained-nmt', type=str, metavar='N', default=None,
                        help='Load model from pretrained NMT')
    parser.add_argument('--gradient-reverse', default=False, action='store_true',
                        help='gradient reversal')

  @classmethod
  def build_model(cls, args, task):
    """Build a new model instance."""
    nmt = TransformerModel.build_model(args, task)
    if hasattr(args, 'pretrained_nmt') and args.pretrained_nmt:
      model_checkpoint = args.pretrained_nmt
      logger.info("Load NMT models from {}".format(model_checkpoint))
      nmt = load_pretrained_model(nmt, model_checkpoint)
    return cls(args, nmt)


  def forward(self, src_tokens, src_lengths, prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        adapter_mask = None):

    x = self.encoder(
      src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
    )

    nmt_output, _ = self.decoder(
      prev_output_tokens,
      encoder_out=x,
      features_only=features_only,
      alignment_layer=alignment_layer,
      alignment_heads=alignment_heads,
      src_lengths=src_lengths,
      return_all_hiddens=return_all_hiddens,
    )

    x = torch.transpose(x['encoder_out'][0], 0, 1)
    x = torch.sum(x, dim=1).squeeze()
    if self.gradrev:
      x = self.gradrev(x)
    x = self.src_domain_discriminator(x)

    return nmt_output, x



class FeedForwadLayer(nn.Module):
    def __init__(self, input_size, output_size, activation_fn,  dropout_prob=None):
        super(FeedForwadLayer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.activate_fn = activation_fn
        if dropout_prob is not None:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.layer(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.activate_fn(x)
        return x


class GradientReversalFunction(Function):
  """
  Gradient Reversal Layer from:
  Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
  Forward pass is the identity function. In the backward pass,
  the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
  """
  
  @staticmethod
  def forward(ctx, x, lambda_):
    ctx.lambda_ = lambda_
    return x.clone()
  
  @staticmethod
  def backward(ctx, grads):
    lambda_ = ctx.lambda_
    lambda_ = grads.new_tensor(lambda_)
    dx = -lambda_ * grads
    return dx, None


class GradientReversal(torch.nn.Module):
  def __init__(self, lambda_=1):
    super(GradientReversal, self).__init__()
    self.lambda_ = lambda_
  
  def forward(self, x):
    return GradientReversalFunction.apply(x, self.lambda_)
