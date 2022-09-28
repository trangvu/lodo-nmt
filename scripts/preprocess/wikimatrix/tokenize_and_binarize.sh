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
data_dir='./WikiMatrix/processed'
DATA_BIN=/home/trangvu/damnmt/data-bin/medical

# train
cd $data_dir
for lang_pair in *; do
  if [ -d "$lang_pair" ]; then
    echo "========================================"
    echo "Prosscess $lang_pair"
    src=$(echo "$lang_pair" | cut -d- -f1)
    tgt=$(echo "$lang_pair" | cut -d- -f2)
    echo "========================================"
    echo "Tokenize ${src}-${tgt} training data"
    in_dir="${src}-${tgt}"

    for f in train dev; do
      for l in $src $tgt; do
        cat $in_dir/$f.$l | \
              perl $NORM_PUNC $l | \
              perl $REM_NON_PRINT_CHAR | \
              perl $TOKENIZER -threads 8 -a -l $l > $in_dir/$f.tok.$l
      done
      echo "Apply SPM ${src}-${tgt} ${f} data"
      python $SPM_ENCODE \
        --model $SPM_MODEL \
        --output_format=piece \
        --inputs $in_dir/$f.tok.$src $in_dir/$f.tok.$tgt \
        --outputs $in_dir/sp.$f.$src $in_dir/sp.$f.$tgt
      perl $CLEAN -ratio 3 $in_dir/sp.$f $src $tgt $in_dir/sp.$f.$src-$tgt 1 250
    done
    echo "Binarize ${src}-${tgt} training data"
    fairseq-preprocess \
      --source-lang $src --target-lang $tgt \
      --trainpref $in_dir/sp.train.$src-$tgt \
      --validpref $in_dir/sp.dev.$src-$tgt \
      --thresholdsrc 0 --thresholdtgt 0 \
      --destdir $DATA_BIN \
      --srcdict $DATA_DICT --tgtdict $DATA_DICT --workers 4
  fi
done