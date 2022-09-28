import logging
import sys
import time

import torch
from fairseq.data import iterators, FairseqDataset, data_utils, ListDataset, LanguagePairDataset
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask, get_time_gap

from ..data.multidomain_multilingual_data_manager import MultidomainMultilingualDatasetManager

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

@register_task("translation_multilingual_multidomain")
class TranslationMultiDomainMultilingualTask(TranslationMultiSimpleEpochTask):
  """
  Translate from one language to another language, supported multiple domain
  """

  @staticmethod
  def add_args(parser):
    """Add task-specific arguments to the parser."""
    # fmt: off
    TranslationMultiSimpleEpochTask.add_args(parser)
    parser.add_argument("--domain-dict",
      default=None, type=str, help="list of allowed domains for provided language pairs in the form "
                                   "{en-xx:domain1,domain2;xx-en:domain1;...}"
                                   "'. If not listed, "
                                   "all domains "
                                   "will be used if available",
    )
    parser.add_argument('--domains', default=None,
                        help='comma-separated list of domains')
    parser.add_argument('--random-domain-mask-ratio', default=0., type=float,
                        help='Ratio to random mask domain as out-of-domain')
    parser.add_argument('--add-domain-tags', default=False,
                        action='store_true', help="Add domain tag to the beginning of the sentence")
    parser.add_argument('--tag-at-decoder', default=False,
                        action='store_true', help="Add domain tag to the beginning of decoder side rather than "
                                                  "encoder side")
    parser.add_argument('--domain-tok', default=None, type=str, help="Domain tok to be prepend during inference")
    parser.add_argument('--disable-leave-one-out', default=False,
                        action='store_true', help="Disable leave-one-out in fusion training")
    parser.add_argument('--mixed-batch', default=False,
                        action='store_true', help="Disable leave-one-out in fusion training")
    parser.add_argument('--homo-batch', default=False,
                        action='store_true', help="Train with homo batch")

    # SamplingMethod.add_arguments(parser)
    # MultidomainMultilingualDatasetManager.add_args(parser)
    # fmt: on

  def __init__(self, args, langs, domains, dicts, domain_dict, training):
    self.args = args
    self.datasets = {}
    self.dataset_to_epoch_iter = {}
    self.langs = langs
    self.dicts = dicts
    self.domains = domains
    self.domain_dict = domain_dict
    self.training = training
    if training:
      self.lang_pairs = args.lang_pairs
    else:
      self.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]

    self.eval_lang_pairs = self.lang_pairs

    self.model_lang_pairs = self.lang_pairs
    self.source_langs = [d.split("-")[0] for d in self.lang_pairs]
    self.target_langs = [d.split("-")[1] for d in self.lang_pairs]
    self.check_dicts(self.dicts, self.source_langs, self.target_langs)

    self.sampling_method = SamplingMethod.build_sampler(args, self)
    self.data_manager = MultidomainMultilingualDatasetManager.setup_data_manager(
      args, self.lang_pairs, langs, dicts, self.sampling_method
    )
    self.valid_flag = False

  @classmethod
  def setup_task(cls, args, **kwargs):
    langs, domains, dicts, domain_dict, training = MultidomainMultilingualDatasetManager.prepare(
      cls.load_dictionary, args, **kwargs
    )
    return cls(args, langs, domains, dicts, domain_dict, training)


  def check_dicts(self, dicts, source_langs, target_langs):
    if self.args.source_dict is not None or self.args.target_dict is not None:
      # no need to check whether the source side and target side are sharing dictionaries
      return
    src_dict = dicts[source_langs[0]]
    tgt_dict = dicts[target_langs[0]]
    for src_lang in source_langs:
      assert (
          src_dict == dicts[src_lang]
      ), "Diffrent dictionary are specified for different source languages; "
      "TranslationMultiDomainMultilingualTask only supports one shared dictionary across all source languages"
    for tgt_lang in target_langs:
      assert (
          tgt_dict == dicts[tgt_lang]
      ), "Diffrent dictionary are specified for different target languages; "
      "TranslationMultiDomainMultilingualTask only supports one shared dictionary across all target languages"

  def load_dataset(self, split, epoch=1, combine=False, **kwargs):
    """Load a given dataset split.

    Args:
        split (str): name of the split (e.g., train, valid, test)
    """
    if not self.training:
      self.datasets[split] = self.data_manager.load_dataset_for_inference(split, combine, **kwargs)
      return
    if split in self.datasets:
      dataset = self.datasets[split]
      if self.has_sharded_data(split):
        if self.args.virtual_epoch_size is not None:
          if dataset.load_next_shard:
            shard_epoch = dataset.shard_epoch
          else:
            # no need to load next shard so skip loading
            # also this avoid always loading from beginning of the data
            return
        else:
          shard_epoch = epoch
    else:
      # estimate the shard epoch from virtual data size and virtual epoch size
      shard_epoch = self.data_manager.estimate_global_pass_epoch(epoch)
    logger.info(f"loading data for {split} epoch={epoch}/{shard_epoch}")
    logger.info(f"mem usage: {data_utils.get_mem_usage()}")
    if split in self.datasets:
      del self.datasets[split]
      logger.info("old dataset deleted manually")
      logger.info(f"mem usage: {data_utils.get_mem_usage()}")
    self.datasets[split] = self.data_manager.load_dataset(
      split,
      self.training,
      epoch=epoch,
      combine=combine,
      shard_epoch=shard_epoch,
      **kwargs,
    )

  def build_generator(
    self,
    models,
    args,
    seq_gen_cls=None,
    extra_gen_cls_kwargs=None,
  ):
    if not getattr(self.args, "keep_inference_langtok", False):
      _, tgt_langtok_spec = self.args.langtoks["main"]
      symbols_to_strip = set()
      if tgt_langtok_spec:
        tgt_lang_tok = self.data_manager.get_decoder_langtok(
          self.args.target_lang, tgt_langtok_spec
        )
        symbols_to_strip.add(tgt_lang_tok)

      add_domain_tags = getattr(self.args, 'add_domain_tags', False)
      tag_at_decoder = getattr(self.args, 'tag_at_decoder', False)
      if add_domain_tags and tag_at_decoder:
        domain_tok = self.data_manager._get_domain_tok(self.args.domain_tok)
        domain_tok = self.data_manager.get_target_dictionary(self.args.target_lang).index(domain_tok)
        symbols_to_strip.add(domain_tok)

      if len(symbols_to_strip) > 0:
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = symbols_to_strip

    return super().build_generator(
      models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

  def inference_step(
    self, generator, models, sample, prefix_tokens=None, constraints=None
  ):
    with torch.no_grad():
      add_domain_tags = getattr(self.args, 'add_domain_tags', False)
      tag_at_decoder = getattr(self.args, 'tag_at_decoder', False)
      to_add_prefix_tokens = []
      if add_domain_tags and tag_at_decoder:
        domain_tok = self.data_manager._get_domain_tok(self.args.domain_tok)
        domain_tok = self.data_manager.get_target_dictionary(self.args.target_lang).index(domain_tok)
        to_add_prefix_tokens.append(domain_tok)
      _, tgt_langtok_spec = self.args.langtoks["main"]
      if not self.args.lang_tok_replacing_bos_eos:
        if prefix_tokens is None and tgt_langtok_spec:
          tgt_lang_tok = self.data_manager.get_decoder_langtok(
            self.args.target_lang, tgt_langtok_spec
          )
          src_tokens = sample["net_input"]["src_tokens"]
          bsz = src_tokens.size(0)
          to_add_prefix_tokens.insert(0, tgt_lang_tok)
          prefix_tokens = (
            torch.LongTensor([to_add_prefix_tokens]).expand(bsz, -1).to(src_tokens)
          )
        return generator.generate(
          models,
          sample,
          prefix_tokens=prefix_tokens,
          constraints=constraints,
        )
      else:
        if prefix_tokens is None and len(to_add_prefix_tokens) > 0:
          src_tokens = sample["net_input"]["src_tokens"]
          bsz = src_tokens.size(0)
          prefix_tokens = (
            torch.LongTensor([to_add_prefix_tokens]).expand(bsz, -1).to(src_tokens)
          )
        return generator.generate(
          models,
          sample,
          prefix_tokens=prefix_tokens,
          bos_token=self.data_manager.get_decoder_langtok(
            self.args.target_lang, tgt_langtok_spec
          )
          if tgt_langtok_spec
          else self.target_dictionary.eos(),
        )

  def create_batch_sampler_single_domain_func(
    self,
    max_positions,
    ignore_invalid_inputs,
    max_tokens,
    max_sentences,
    required_batch_size_multiple=1,
    seed=1,
  ):
    def construct_batch_sampler(dataset, epoch):
      splits = [
        s for s, _ in self.datasets.items() if self.datasets[s] == dataset
      ]
      split = splits[0] if len(splits) > 0 else None
      # NEW implementation
      if epoch is not None:
        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

      # get indices ordered by example size
      start_time = time.time()
      logger.info(f"start batch sampler: mem usage: {data_utils.get_mem_usage()}")
      
      mixed_batch = getattr(self.args, "mixed_batch", True)
      if split == 'train' and not mixed_batch:
        logger.info("Create single domain minibatch for train set")
        num_datasets = len(dataset.cumulated_sizes)
        batch_sampler = []
        for i in range(num_datasets):
          with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices_of_single_dataset(i)
          # filter examples that are too large
          if max_positions is not None:
            indices = self.filter_indices_by_size(
              indices, dataset, max_positions, ignore_invalid_inputs
            )
          batches = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
          )
          batch_sampler.extend(batches)
      else:
        logger.info(f"Create mixed domain minibatch for {split} set")
        with data_utils.numpy_seed(seed):
          indices = dataset.ordered_indices()
        logger.info(
          f"[{split}] @batch_sampler order indices time: {get_time_gap(start_time, time.time())}"
        )
        logger.info(f"mem usage: {data_utils.get_mem_usage()}")

        # filter examples that are too large
        if max_positions is not None:
          my_time = time.time()
          indices = self.filter_indices_by_size(
            indices, dataset, max_positions, ignore_invalid_inputs
          )
          logger.info(
            f"[{split}] @batch_sampler filter_by_size time: {get_time_gap(my_time, time.time())}"
          )
          logger.info(f"mem usage: {data_utils.get_mem_usage()}")

        # create mini-batches with given size constraints
        my_time = time.time()
        batch_sampler = dataset.batch_by_size(
          indices,
          max_tokens=max_tokens,
          max_sentences=max_sentences,
          required_batch_size_multiple=required_batch_size_multiple,
        )

        logger.info(
          f"[{split}] @batch_sampler batch_by_size time: {get_time_gap(my_time, time.time())}"
        )
      logger.info(
        f"[{split}] per epoch batch_sampler set-up time: {get_time_gap(start_time, time.time())}"
      )
      logger.info(f"mem usage: {data_utils.get_mem_usage()}")

      return batch_sampler

    return construct_batch_sampler

  def get_batch_iterator(
    self,
    dataset,
    max_tokens=None,
    max_sentences=None,
    max_positions=None,
    ignore_invalid_inputs=False,
    required_batch_size_multiple=1,
    seed=1,
    num_shards=1,
    shard_id=0,
    num_workers=0,
    epoch=1,
    data_buffer_size=0,
    disable_iterator_cache=False,
  ):
    """
    Get an iterator that yields batches of data from the given dataset.

    Args:
        dataset (~fairseq.data.FairseqDataset): dataset to batch
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        max_positions (optional): max sentence length supported by the
            model (default: None).
        ignore_invalid_inputs (bool, optional): don't raise Exception for
            sentences that are too long (default: False).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 0).
        data_buffer_size (int, optional): number of batches to
            preload (default: 0).
        disable_iterator_cache (bool, optional): don't cache the
            EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
            (default: False).
    Returns:
        ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
            given dataset split
    """
    # initialize the dataset with the correct starting epoch
    assert isinstance(dataset, FairseqDataset)
    if dataset in self.dataset_to_epoch_iter:
      return self.dataset_to_epoch_iter[dataset]
    if self.args.sampling_method == "RoundRobin":
      batch_iter = super().get_batch_iterator(
        dataset,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=ignore_invalid_inputs,
        required_batch_size_multiple=required_batch_size_multiple,
        seed=seed,
        num_shards=num_shards,
        shard_id=shard_id,
        num_workers=num_workers,
        epoch=epoch,
        data_buffer_size=data_buffer_size,
        disable_iterator_cache=disable_iterator_cache,
      )
      self.dataset_to_epoch_iter[dataset] = batch_iter
      return batch_iter

    fusion = getattr(self.args, 'fusion', False)
    homo_batch = getattr(self.args, 'homo_batch', False)
    if fusion or homo_batch:
      construct_batch_sampler = self.create_batch_sampler_single_domain_func(
        max_positions,
        ignore_invalid_inputs,
        max_tokens,
        max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
        seed=seed,
      )
    else:
      construct_batch_sampler = self.create_batch_sampler_func(
        max_positions,
        ignore_invalid_inputs,
        max_tokens,
        max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
        seed=seed,
      )

    epoch_iter = iterators.EpochBatchIterator(
      dataset=dataset,
      collate_fn=dataset.collater,
      batch_sampler=construct_batch_sampler,
      seed=seed,
      num_shards=num_shards,
      shard_id=shard_id,
      num_workers=num_workers,
      epoch=epoch,
    )
    return epoch_iter