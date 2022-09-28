import random

import torch
from fairseq.data import FairseqDataset, TransformEosLangPairDataset

import torch.nn.functional as F

class DomainAwareLanguagePairDataset(FairseqDataset):
  def __init__(self, lang_pair, domain_label, label_dict, add_domain_tag=False,
               tag_at_decoder=False, training=True, random_domain_mask_ratio=0.,
               fusion=False, leave_one_out=True, language_adapter=False, src_lang=None, tgt_lang=None):
    self.lang_pair = lang_pair
    self.sizes = lang_pair.sizes
    self.domain_label = domain_label
    if isinstance(lang_pair, TransformEosLangPairDataset):
      self.src_dict = lang_pair.dataset.src_dict
      self.tgt_dict = lang_pair.dataset.tgt_dict
    else:
      self.src_dict = lang_pair.src_dict
      self.tgt_dict = lang_pair.tgt_dict
    self.domain_index = self.src_dict.index(domain_label)
    self.label_dict = label_dict
    self.add_domain_tag = add_domain_tag
    self.tag_at_decoder = tag_at_decoder
    self.num_domains = len(self.label_dict) - self.label_dict.nspecial
    self.training = training
    self.random_domain_mask_ratio = random_domain_mask_ratio
    self.fusion = fusion
    self.leave_one_out = leave_one_out
    self.language_adapter = language_adapter
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang

  def __len__(self):
    return len(self.lang_pair)

  def __getitem__(self, index):
    item = self.lang_pair[index]
    domain_index = self.domain_index
    domain_label = self.domain_label
    if self.training and self.random_domain_mask_ratio > 0:
      if random.random() < self.random_domain_mask_ratio:
        domain_index = self.src_dict.unk_index
        item.update(domain_mask=1)
      else:
        item.update(domain_mask=0)
    if self.add_domain_tag:
      if self.tag_at_decoder:
        tgt = torch.cat([torch.LongTensor([domain_index]), item['target']])
        item.update(target=tgt)
      else:
        src = torch.cat([torch.LongTensor([domain_index]), item['source']])
        item.update(source=src)
    if domain_label is not None:
      item.update(label=self.label_dict.index(domain_label) - self.label_dict.nspecial)
    else:
      item.update(label=-1)
    return item

  def collater(self, samples, **extra_args):
    batch = self.lang_pair.collater(samples)
    labels = torch.LongTensor([s['label'] for s in samples])
    tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples])
    batch.update(labels=labels)
    batch.update(tgt_lengths=tgt_lengths)
    if self.training and self.random_domain_mask_ratio > 0:
      domain_masks = torch.LongTensor([s['domain_mask'] for s in samples])
      batch.update(domain_masks=domain_masks)
    if self.training and self.fusion and self.leave_one_out:
      if 'net_input' in batch:
        batch['net_input'].update(adapter_mask=F.one_hot(labels, num_classes=self.num_domains))
        
    if self.language_adapter:
      batch['net_input'].update(active_adapters={
        'encoder_adapter': f"lang-{self.src_lang}",
        'decoder_adapter': f"lang-{self.tgt_lang}",
      })
    else:
      batch['net_input'].update(active_adapters=None)
    return batch

  def num_tokens(self, index):
    return self.lang_pair.num_tokens(index)

  def size(self, index):
    return self.lang_pair.size(index)

  def ordered_indices(self):
    return self.lang_pair.ordered_indices()

  @property
  def supports_prefetch(self):
    return self.lang_pair.supports_prefetch

  def prefetch(self, indices):
    self.lang_pair.prefetch(indices)
    self.domain_label.prefetch(indices)
