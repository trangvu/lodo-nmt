"""
This file includes modifications to fairseq distributed through GitHub at https://github.com/pytorch/fairseq/blob/83e615d66905b8ca7483122a37da1a85f13f4b8e/fairseq/tasks/translation.py under this license https://github.com/pytorch/fairseq/blob/83e615d66905b8ca7483122a37da1a85f13f4b8e/LICENSE.

Copyright with respect to the modifications: Copyright 2021 Naver Corporation

ORIGINAL COPYRIGHT NOTICE AND PERMISSION NOTICE:

Copyright (c) Facebook, Inc. and its affiliates.

MIT License. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import logging
import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq import utils


logger = logging.getLogger(__name__)


@register_task("nle_translation")
class NLETranslationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--adapter-dim', type=int,
                            help="bottleneck dimension of the adapters")
        parser.add_argument('--decoder-adapter-dim', type=int,
                            help="bottleneck dimension of the decoder adapters")
        parser.add_argument('--adapter-uids', '--adapters', nargs='+', default=[],
                            help="name of the encoder adapters")
        parser.add_argument('--decoder-adapter-uids', '--decoder-adapters', '--dec-adapters', nargs='+', default=[],
                            help="name of the decoder adapters")

        parser.add_argument('--lang-adapters', action='store_true')
        parser.add_argument('--target-lang-code', action='store_true')
        parser.add_argument('--lang-code', action='store_true')
        parser.add_argument('--append-codes', action='store_true')
        parser.add_argument('--lang-as-bos', action='store_true')

        parser.add_argument('--domain-adapters')
        parser.add_argument('--domain-tag')
        # fmt: on

    def __init__(self, args, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)
        self.init_tags(args)

    def init_tags(self, args):
        self.bos_token = None
        self.append_codes = args.append_codes

        source_lang_code_idx = target_lang_code_idx = domain_tag_idx = None
        if args.lang_code:
            code = f'<lang:{args.source_lang}>'
            assert code in self.source_dictionary, f'source lang code {code} is out of vocabulary'
            source_lang_code_idx = self.source_dictionary.index(code)
        if args.target_lang_code:
            code = f'<lang:{args.target_lang}>'
            assert code in self.source_dictionary, f'target lang code {code} is out of vocabulary'
            target_lang_code_idx = self.source_dictionary.index(code)
        if args.domain_tag:
            code = f'<corpus:{args.domain_tag}>'
            assert code in self.source_dictionary, f'domain tag {code} is out of vocabulary'
            domain_tag_idx = self.source_dictionary.index(code)

        self.tags = [source_lang_code_idx, target_lang_code_idx, domain_tag_idx]
        self.tags = torch.tensor([code for code in self.tags if code is not None])

        if args.lang_as_bos:
            code = f'<lang:{args.target_lang}>'
            assert code in self.target_dictionary, f'target lang code {code} is out of vocabulary'
            self.bos_token = self.target_dictionary.index(code)

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.source_lang is not None and args.target_lang is not None, '--source-lang and --target-lang are required'

        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        assert len(utils.split_paths(args.data)) == 1

        def find_file(*filenames):
            for filename in filenames:
                file_path = os.path.join(args.data, filename)
                if os.path.exists(file_path):
                    return file_path

        # automatically infer checkpoint path
        if args.path is None:
            args.path = find_file('checkpoint_best.pt', 'checkpoint_last.pt')
            if args.path is None:
                raise ValueError('no checkpoint found, please specify --path')

        src_dict = find_file('dict.src.txt', f'dict.{args.source_lang}.txt', 'dict.txt')
        tgt_dict = find_file('dict.tgt.txt', f'dict.{args.target_lang}.txt', 'dict.txt')
        assert src_dict is not None, 'no source dictionary found'
        assert tgt_dict is not None, 'no target dictionary found'

        # load dictionaries
        src_dict = cls.load_dictionary(src_dict)
        tgt_dict = cls.load_dictionary(tgt_dict)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info(f"[{args.source_lang}] dictionary: {len(src_dict)} types")
        logger.info(f"[{args.target_lang}] dictionary: {len(tgt_dict)} types")

        if args.lang_adapters:
            args.adapter_uids.insert(0, f'lang:{args.source_lang}')
            args.decoder_adapter_uids.insert(0, f'lang:{args.target_lang}')
        if args.domain_adapters:
            args.adapter_uids.append(args.domain_adapters)
            args.decoder_adapter_uids.append(args.domain_adapters)
        
        # remove duplicates
        args.adapter_uids = list(dict.fromkeys(args.adapter_uids))
        args.decoder_adapter_uids = list(dict.fromkeys(args.decoder_adapter_uids))

        # dirty hack to force the model to use these adapter uids
        model_overrides = eval(args.model_overrides)
        model_overrides['adapter_uids'] = args.adapter_uids
        model_overrides['decoder_adapter_uids'] = args.decoder_adapter_uids
        model_overrides['source_lang'] = args.source_lang
        model_overrides['target_lang'] = args.target_lang
        if args.adapter_dim:
            model_overrides['adapter_dim'] = args.adapter_dim
        if args.decoder_adapter_dim:
            model_overrides['decoder_adapter_dim'] = args.decoder_adapter_dim
        args.model_overrides = repr(model_overrides)

        return cls(args, src_dict, tgt_dict)

    def get_batch_iterator(self, *args, max_positions=None, **kwargs):
        # ignore max_positions
        return super().get_batch_iterator(*args, max_positions=None, **kwargs)

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        max_len = self.args.max_source_positions - len(self.tags) - 1
        src_tokens = [torch.cat([tokens[:-1][:max_len], tokens[-1:]]) for tokens in src_tokens]

        # preprend or append language codes to the source sequence
        if len(self.tags) > 0:
            src_lengths = [length + len(self.tags) for length in src_lengths]
            if self.append_codes:
                src_tokens = [torch.cat([tokens[:-1], self.tags, tokens[-1:]]) for tokens in src_tokens]
            else:
                src_tokens = [torch.cat([self.tags, tokens]) for tokens in src_tokens]
        return super().build_dataset_for_inference(src_tokens, src_lengths, constraints)

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        # if --lang-as-bos is specified, use language code instead of bos as first target embedding
        return generator.generate(
            models, sample, prefix_tokens=prefix_tokens, constraints=constraints, bos_token=self.bos_token
        )

    def build_generator(self, models, args, *args_, **kwargs):
        # dirty hack to automatically infer BPE tokenizer from model checkpoint        
        if hasattr(args, 'bpe') and args.bpe is None:
            args.bpe = models[0].args.bpe

            if args.bpe == 'sentencepiece' and not getattr(args, 'sentencepiece_model', None):
                args.sentencepiece_model = os.path.join(args.data, 'spm.model')

        # dirty hack to automatically infer lang code strategy from model checkpoint
        if hasattr(args, 'target_lang_code'):
            for opt in 'lang_code', 'target_lang_code', 'lang_as_bos', 'append_codes':
                if getattr(models[0].args, opt, False):
                    setattr(args, opt, True)
            self.init_tags(args)

        # logger.info(args)
        # logger.info(models[0])

        return super().build_generator(models, args, *args_, **kwargs)

    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError('this fairseq task only supports inference with fairseq-interactive')
    def train_step(self, *args, **kwargs):
        raise NotImplementedError('this fairseq task only supports inference with fairseq-interactive')
    def valid_step(self, *args, **kwargs):
        raise NotImplementedError('this fairseq task only supports inference with fairseq-interactive')
