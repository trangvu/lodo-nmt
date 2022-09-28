#!/bin/bash

NMT_DIR=${KRYLOV_DATA_DIR}/thitvu/damnmt
CUR_DIR=`pwd`
TOKENIZER=$NMT_DIR/mdml/m2m_100/tok.sh
DATA_DICT=$NMT_DIR/models/mbart.cc25/dict.txt
SPM_MODEL=$NMT_DIR/models/mbart.cc25/sentence.bpe.model
SPM_ENCODE=$CUR_DIR/../../fairseq/scripts/spm_encode.py
DATA_BIN=$NMT_DIR/data-bin/wmt

MOSES=$NMT_DIR/mdml/m2m_100/tokenizers/thirdparty/mosesdecoder
SCRIPTS=$MOSES/scripts
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
CLEAN=$MOSES/scripts/training/clean-corpus-n.perl

URLS=(
    "http://www.statmt.org/europarl/v10/training/europarl-v10.pl-en.tsv.gz"
    "https://s3.amazonaws.com/web-language-models/paracrawl/release5.1/en-pl.txt.gz"
    "http://data.statmt.org/wikititles/v2/wikititles-v2.pl-en.tsv.gz"
    "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.en-pl.langid.tsv.gz"
    "http://data.statmt.org/wmt20/translation-task/dev.tgz"
    "http://data.statmt.org/wmt20/translation-task/test.tgz"
)
FILES=(
    "europarl-v10.pl-en.tsv.gz"
    "en-pl.txt.gz"
    "wikititles-v2.pl-en.tsv.gz"
    "WikiMatrix.v1.en-pl.langid.tsv.gz"
    "dev.tgz"
    "test.tgz"
)

PL_EN_TSV_CORPORA=(
    "europarl-v10.pl-en.tsv"
    "wikititles-v2.pl-en.tsv"
)

EN_PL_TSV_CORPORA=(
    "en-pl.txt"
    "WikiMatrix.v1.en-pl.langid.tsv"
)

CORPORA=(
  "europarl-v10.pl-en"
  "en-pl"
  "wikititles-v2.pl-en"
  "WikiMatrix.v1.en-pl.langid"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=pl
lang=en-pl
prep=wmt20_en_pl
tmp=$prep/tmp
orig=orig_wmt20
dev=dev/newstest2019

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -3} == ".gz" ]; then
            gunzip $file
        fi
    fi
done
cd ..

echo "pre-process pl-en tsv parallel dataset"
for f in "${PL_EN_TSV_CORPORA[@]}"; do
  f_org="${f%.*}"
  cat $orig/$f | \
      cut -d$'\t' -f1 > $orig/$f_org.pl
  cat $orig/$f | \
      cut -d$'\t' -f2 > $orig/$f_org.en
done

echo "pre-process en-pl tsv parallel dataset"
for f in "${EN_PL_TSV_CORPORA[@]}"; do
  f_org="${f%.*}"
  cat $orig/$f | \
      cut -d$'\t' -f1 > $orig/$f_org.en
  cat $orig/$f | \
      cut -d$'\t' -f2 > $orig/$f_org.pl
done

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/sgm/newstest2020-enpl-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175

echo "splitting train and valid..."
for l in $src $tgt; do
    "en-de.txt.gz"
    awk '{if (NR%1333 == 0)  print $0; }' $tmp/train.tags.$lang.clean.$l > $tmp/valid.$l
    awk '{if (NR%1333 != 0)  print $0; }' $tmp/train.tags.$lang.clean.$l > $tmp/train.$l
done


TRAIN=$tmp/train.pl-en
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done


for f in train valid; do
    echo "encode sentencepiece to ${f}..."
    python $SPM_ENCODE --model=$SPM_MODEL \
        --output_format=piece \
        --inputs $tmp/$f.$src $tmp/$f.$tgt \
        --outputs $tmp/sp.$f.$src $tmp/sp.$f.$tgt --min-len 5 --max-len 150
done

f=test
python $SPM_ENCODE --model=$SPM_MODEL \
        --output_format=piece \
        --inputs $tmp/$f.$src $tmp/$f.$tgt \
        --outputs $tmp/sp.$f.$src $tmp/sp.$f.$tgt

for L in $src $tgt; do
    cp $tmp/sp.test.$L $prep/test.$L
    cp $tmp/sp.train.$L $prep/train.$L
    cp $tmp/sp.valid.$L $prep/valid.$L
done

echo "Binarize ${src}-${tgt} training data"
fairseq-preprocess \
  --source-lang $src --target-lang $tgt \
  --trainpref $prep/train \
  --validpref $prep/valid \
  --testpref $prep/test \
  --thresholdsrc 0 --thresholdtgt 0 \
  --destdir $DATA_BIN \
  --srcdict $DATA_DICT --tgtdict $DATA_DICT --workers 4