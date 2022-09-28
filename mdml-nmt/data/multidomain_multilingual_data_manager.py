import logging
import sys
from collections import defaultdict

from fairseq import utils
from fairseq.data import Dictionary, PrependTokenDataset
from fairseq.data.multilingual.multilingual_data_manager import MultilingualDatasetManager, _lang_id
from fairseq.file_io import PathManager
from .domain_aware_language_pair_dataset import DomainAwareLanguagePairDataset

logging.basicConfig(
  format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class MultidomainMultilingualDatasetManager(MultilingualDatasetManager):
  def __init__(self, args, lang_pairs, langs, domains, dicts, sampling_method):
    super().__init__(args, lang_pairs, langs, dicts, sampling_method)
    self.domains = domains
    self.domain_dict = {}
    domain_dict = getattr(args, 'domain_dict', None)
    if domain_dict:
      domain_dicts = args.domain_dict.split(';')
      for domain_dict in domain_dicts:
        lang_pair = domain_dict.split(':')[0]
        domain = domain_dict.split(':')[1]
        self.domain_dict[lang_pair] = set(domain.split(','))


  @classmethod
  def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
    domains = args.domains
    return MultidomainMultilingualDatasetManager(
      args, lang_pairs, langs, domains, dicts, sampling_method
    )

  @classmethod
  def load_domains(cls, args, **kwargs):
    if args.domains is None:
      raise ValueError("--domains is not specified")
    domains = args.domains.split(',')
    return domains

  @classmethod
  def prepare(cls, load_dictionary, args, **kargs):
    language_list, dicts, training = MultilingualDatasetManager.prepare(load_dictionary, args, **kargs)
    # Check domains
    domain_lists = cls.load_domains(args, **kargs)
    cls.domain_tags = [f"__domain-{d}__" for d in domain_lists]

    cls.domain_dictionary = Dictionary()
    for l in cls.domain_tags:
      cls.domain_dictionary.add_symbol(l)
    # Add domain tags to dictionary
    for lang, dictionary in dicts.items():
      for tag in cls.domain_tags:
        dictionary.add_symbol(tag)
    return language_list, domain_lists, dicts, cls.domain_dictionary, training

  def get_split_data_param_list(self, split, epoch, shard_epoch=None):
    param_list = []
    data_paths, domains, lang_pairs = self.get_data_paths_and_domain_lang_pairs(split)
    logger.info(f"langtoks settings: {self.args.langtoks}")
    split_num_shards_dict = self.get_split_num_data_shards(split)
    for data_category, paths in data_paths.items():
      if data_category not in lang_pairs:
        continue
      paths = utils.split_paths(paths)
      assert len(paths) > 0
      if len(paths) > 1:
        self._has_sharded_data = True
      if split != getattr(self.args, "train_subset", None):
        # if not training data set, use the first shard for valid and test
        paths = paths[:1]
    
      if data_category in self.args.langtoks:
        lang_tok_spec = self.args.langtoks[data_category]
      else:
        # default to None
        lang_tok_spec = (None, None)
    
      # infer langcode
      lang_dirs = [
        lang_pair.split("-") for lang_pair in lang_pairs[data_category]
      ]
      lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
      for domain in domains[data_category]:
        for src, tgt in lang_dirs:
          assert src is not None or data_category == "mono_dae", (
            f"error: src={src}, " "tgt={tgt} for data_category={data_category}"
          )
          logger.info(f"preparing param for {data_category} : {domain} : {src} - {tgt}")
          key = self.get_dataset_key(data_category, domain, src, tgt)
          if key not in split_num_shards_dict:
            continue
          data_path = self.get_split_data_path(
            paths, epoch, shard_epoch, split_num_shards_dict[key]
          )
          param_list.append(
            {
              "key": key,
              "data_path": f"{data_path}/{domain}",
              "split": split,
              "src": src,
              "src_dict": self.get_source_dictionary(src)
              if src and data_category != "mono_dae"
              else None,
              "tgt": tgt,
              "tgt_dict": self.get_target_dictionary(tgt),
              "data_category": data_category,
              "langtok_spec": lang_tok_spec,
            }
          )
    return param_list

  def _get_domain_tok(self, domain):
    return f"__domain-{domain}__"

  def get_data_paths_and_domain_lang_pairs(self, split):
    datapaths = {"main": self.args.data}
    domains = {"main": self.domains.split(',')}
    lang_pairs = {"main": self.lang_pairs}
    if split == getattr(self.args, "train_subset", None):
      # only training data can have extra data and extra language pairs
      if self.args.extra_data:
        extra_datapaths = self.args.extra_data
        datapaths.update(extra_datapaths)
      if self.args.extra_lang_pairs:
        extra_lang_pairs = {
          k: v.split(",") for k, v in self.args.extra_lang_pairs.items()
        }
        lang_pairs.update(extra_lang_pairs)
    return datapaths, domains, lang_pairs

  @classmethod
  def _get_shard_num_dict(cls, domains, split, paths):
    shards = {}
    for path in paths:
      for domain in domains:
        files = PathManager.ls(f"{path}/{domain}")
        directions = set()
        for f in files:
          if f.startswith(split) and f.endswith(".idx"):
            # idx files of the form "{split}.{src}-{tgt}.{lang}.idx"
            direction = f.split(".")[-3]
            directions.add(direction)
        if len(directions) > 0:
          shards[domain] = defaultdict(int)
          for direction in directions:
            shards[domain][direction] += 1
    return shards

  def get_split_num_data_shards(self, split):
    if split in self._num_shards_dict:
      return self._num_shards_dict[split]
    num_shards_dict = {}
    data_paths, domains, lang_pairs = self.get_data_paths_and_domain_lang_pairs(split)
  
    for data_category, paths in data_paths.items():
      if data_category not in domains:
        continue
      if data_category not in lang_pairs:
        continue
      paths = utils.split_paths(paths)
      shards_dict = self._get_shard_num_dict(domains[data_category], split, paths)
      lang_dirs = [
        lang_pair.split("-") for lang_pair in lang_pairs[data_category]
      ]
      lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
      for domain in domains[data_category]:
        for src, tgt in lang_dirs:
          key = self.get_dataset_key(data_category, domain, src, tgt)
          if "mono_" in data_category:
            # monolingual data requires tgt only
            assert src is None or src == tgt, (
              f"error: src={src}, "
              "tgt={tgt} for data_category={data_category}"
            )
            num_shards_dict[key] = shards_dict[tgt]
          else:
            direction = f"{src}-{tgt}"
            if ( direction in self.domain_dict) and (not domain in self.domain_dict[direction]):
              continue
            if f"{src}-{tgt}" in shards_dict[domain]:
              num_shards_dict[key] = shards_dict[domain][f"{src}-{tgt}"]
            elif f"{tgt}-{src}" in shards_dict[domain]:
              # follow the fairseq tradition to use reversed direction data if it is not available
              num_shards_dict[key] = shards_dict[domain][f"{tgt}-{src}"]
    self._num_shards_dict[split] = num_shards_dict
    logger.info(f"[{split}] num of shards: {num_shards_dict}")
    return num_shards_dict

  @classmethod
  def get_dataset_key(cls, data_category, domain, src, tgt):
    return f"{data_category}:{domain}:{src}-{tgt}"

  # def src_dataset_tranform_func(self, src_lang, tgt_lang, dataset, spec=None, domain_label=None):
  #   if self.args.lang_tok_replacing_bos_eos:
  #     # it is handled by self.alter_dataset_langtok
  #     # TODO: Unifiy with alter_dataset_langtok
  #     return dataset
  #   if spec is None:
  #     if domain_label:
  #       domain_id = self.get_source_dictionary(src_lang).index(domain_label)
  #       dataset = PrependTokenDataset(dataset, domain_id)
  #     return dataset
  #   tok = self.get_encoder_langtok(src_lang, tgt_lang, spec)
  #   if tok:
  #     if domain_label:
  #       domain_id = self.get_source_dictionary(src_lang).index(domain_label)
  #       dataset = PrependTokenDataset(dataset, domain_id)
  #     return PrependTokenDataset(dataset, tok)
  #   return dataset
  #
  # def tgt_dataset_tranform_func(self, source_lang, target_lang, dataset, spec=None, domain_label=None):
  #   if dataset is None:
  #     # note that target dataset can be None during inference time
  #     return None
  #   if self.args.lang_tok_replacing_bos_eos:
  #     # TODO: Unifiy with alter_dataset_langtok
  #     # It is handled by self.alter_dataset_langtok.
  #     # The complication in self.alter_dataset_langtok
  #     # makes a unified framework difficult.
  #     return dataset
  #   # if not self.args.decoder_langtok:
  #   if not spec:
  #     if domain_label:
  #       domain_id = self.get_target_dictionary(target_lang).index(domain_label)
  #       dataset = PrependTokenDataset(dataset, domain_id)
  #     return dataset
  #   tok = self.get_decoder_langtok(target_lang, spec)
  #   if tok:
  #     if domain_label:
  #       domain_id = self.get_target_dictionary(target_lang).index(domain_label)
  #       dataset = PrependTokenDataset(dataset, domain_id)
  #     return PrependTokenDataset(dataset, tok)
  #   return dataset

  def load_a_dataset(
    self,
    split,
    data_path,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    prepend_bos=False,
    langpairs_sharing_datasets=None,
    data_category=None,
    **extra_kwargs,
  ):
    logger.info(f"Load a dataset {split}")
    langpair_ds = super(MultidomainMultilingualDatasetManager, self).load_a_dataset(split, data_path, src,
                                                                        src_dict, tgt, tgt_dict, combine,
                                                                        prepend_bos, langpairs_sharing_datasets,
                                                                        data_category, **extra_kwargs)
    # src_domain_label = None
    # tgt_domain_label = None
    # if self.args.add_domain_tags:
    #   if self.args.tag_at_decoder:
    #     tgt_domain_label = domain_label
    #   else:
    #     src_domain_label = domain_label
    data_key = extra_kwargs['key']
    domain_label = self._get_domain_tok(data_key.split(':')[1])
    fusion = getattr(self.args, 'fusion', False)
    disable_l1o = getattr(self.args, 'disable_leave_one_out', False)
    language_adapter = getattr(self.args, 'language_adapter', None)
    src_lang = src
    tgt_lang = tgt
    dataset = DomainAwareLanguagePairDataset(langpair_ds, domain_label,
                                             self.domain_dictionary, self.args.add_domain_tags,
                                             self.args.tag_at_decoder,
                                             random_domain_mask_ratio=self.args.random_domain_mask_ratio,
                                             fusion=fusion, leave_one_out=not(disable_l1o),
                                             language_adapter=language_adapter, src_lang=src_lang, tgt_lang=tgt_lang)
    return dataset

  def load_dataset_for_inference(self, split, combine, **extra_kwargs):
    args = self.args
    src = args.source_lang
    tgt = args.target_lang
    langpair_ds = super(MultidomainMultilingualDatasetManager, self).\
                  load_a_dataset(split, self.args.data, src,
                                self.dicts[src], tgt, self.dicts[tgt], combine=combine, langtok_spec=args.langtoks[
        'main'])
    language_adapter = getattr(self.args, 'language_adapter', None)
    src_lang = src
    tgt_lang = tgt
    domain_label = self._get_domain_tok(args.domain_tok)
    dataset = DomainAwareLanguagePairDataset(langpair_ds, domain_label,
                                             self.domain_dictionary, self.args.add_domain_tags,
                                             self.args.tag_at_decoder,
                                             training=False, leave_one_out=False,
                                             language_adapter=language_adapter, src_lang=src_lang, tgt_lang=tgt_lang)
    return dataset