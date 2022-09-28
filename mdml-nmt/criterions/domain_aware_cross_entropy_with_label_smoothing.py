import gc
import logging
import math

import torch

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging import metrics

import torch.nn.functional as F

logger = logging.getLogger(__name__)

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, weights=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('domain_aware_xent_with_smoothing')
class DomainAwareXentCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, lambda_1, lambda_2, label_smoothing=0.1,
                 disable_domain_disc=False):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.eps = label_smoothing
        self.disable_domain_disc = disable_domain_disc

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--lambda_1', default=1, type=float, metavar='D',
                            help='weight for the NMT loss')
        parser.add_argument('--lambda_2', default=1, type=float, metavar='D',
                            help='weight for the discriminative src loss')
        parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                            help='weight for the discriminative loss')
        parser.add_argument('--disable-domain-disc', action="store_true", default=False,
                            help='Disable domain discriminative loss')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        net_output= model(**sample['net_input'])
        loss, nmt_nll_loss = self.compute_loss_with_label_smoothed(model.nmt, net_output,
                                                                   sample['target'], reduce=reduce)
        nmt_nll_loss = nmt_nll_loss.data
        nmt_loss = loss.item()
        
        if self.disable_domain_disc:
            disc_loss = 0
        else:
            disc_loss = self.compute_classification_loss(net_output[1],
                                                            sample['labels'])
            loss += self.lambda_2 * disc_loss
            disc_loss = disc_loss.data
        sample_size = sample['target'].size(0)
        logging_output = {
            'loss': loss.data,
            'nll_loss': nmt_nll_loss,
            'nmt_loss': nmt_loss,
            'disc_loss': disc_loss,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'ndisc_sentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss_with_label_smoothed(self, model, net_output, target, reduce=True, weights=None):
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce, weights=weights
        )
        return loss, nll_loss

    def compute_classification_loss(self, logits, targets, entropy_minimization=False):
        targets = targets.view(-1)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

        entropy = 0
        if entropy_minimization:
            entropy = probs * lprobs
            entropy = -1.0 * entropy.sum()

        lprobs = lprobs.view(-1, lprobs.size(-1))
        targets = targets.view(-1)
        # if targets.dim() == lprobs.dim() - 1:
        #     lprobs = lprobs.squeeze(0)
        loss = F.nll_loss(lprobs, targets, reduction='sum')
        loss = loss - 0.1 * entropy
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        nmt_sample_size = sum(log.get('nsentences', 0) for log in logging_outputs)
        disc_sample_size = sum(log.get('ndisc_sentences', nmt_sample_size) for log in logging_outputs)
        nmt_loss_sum = sum(log.get('nmt_loss', 0) for log in logging_outputs)
        disc_loss_sum = sum(log.get('disc_loss', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nmt_loss', nmt_loss_sum / nmt_sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('disc_loss', disc_loss_sum / disc_sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_scalar('nll_nmt_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_scalar('nll_disc_loss', disc_loss_sum / disc_sample_size / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
