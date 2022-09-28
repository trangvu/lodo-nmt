#!/bin/bash

SCRIPTS=/home/trangvu/tools/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

CUR_DIR=`pwd`
DATA_DICT=/home/trangvu/damnmt/models/m2m/data_dict.128k.txt
SPM_MODEL=/home/trangvu/damnmt/models/m2m/spm.128k.model
SPM_ENCODE=$CUR_DIR/../../fairseq/scripts/spm_encode.py
data_dir='./test'
DATA_BIN=/home/trangvu/damnmt/data-bin/medical

# khresmoi-summary
in_dir='./test/khresmoi2.0'
langs="en cs de es fr hu pl sv"
prefix="khresmoi-summary"
for l in $langs; do
  for f in dev test; do
    cat $in_dir/$prefix-$f.$l | \
      perl $NORM_PUNC $l | \
      perl $REM_NON_PRINT_CHAR | \
      perl $TOKENIZER -threads 8 -a -l $l > $in_dir/$f.tok.$l

    echo "Apply SPM ${src}-${tgt} ${f} data"
      python $SPM_ENCODE \
        --model $SPM_MODEL \
        --output_format=piece \
        --inputs $in_dir/$f.tok.$l \
        --outputs $in_dir/sp.$f.$l
  done
done
for src in $langs; do
  for tgt in $langs; do
    if [ $src != $tgt ]; then
      echo "* language direction: ${src} -> ${tgt}"
      fairseq-preprocess \
        --source-lang $src --target-lang $tgt \
        --validpref $in_dir/sp.dev \
        --testpref $in_dir/sp.test \
        --thresholdsrc 0 --thresholdtgt 0 \
        --destdir $DATA_BIN \
        --srcdict $DATA_DICT --tgtdict $DATA_DICT
    fi
  done
done