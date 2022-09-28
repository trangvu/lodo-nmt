#!/bin/bash

NMT_DIR=<path to working folder>
ROOT_DIR=$NMT_DIR
src=$1
tgt=$2

exp_name=backbone/generic/$src-$tgt
CUR_DIR=$NMT_DIR/$exp_name
path_2_data=$NMT_DIR/data-bin
SAVE_DIR=$CUR_DIR
rm -r -f $SAVE_DIR
mkdir -p $SAVE_DIR
LANGS='cs,de,en,es_XX,pl,fi_FI,fr,bg,hi_IN,it_IT,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si,tr_TR,vi,ar,ja'
ARCH=mbart_large
TASK=translation_multilingual_multidomain
CRITERION=label_smoothed_cross_entropy
DOMAINS="wmt"
TOK_SCRIPT="${NMT_DIR}/mdml/m2m_100/tok.sh"
MODEL_DICT=$NMT_DIR/models/mbart.cc25/dict.txt
pretrained_model=$NMT_DIR/models/mbart.cc25/model.pt
tool=$NMT_DIR/mdml/fairseq/train.py
decode_tool=$NMT_DIR/mdml/fairseq/fairseq_cli/generate.py

python $tool $path_2_data \
  --save-dir $SAVE_DIR \
  --restore-file $pretrained_model \
  --user-dir $NMT_DIR/mdml/mdml-nmt \
  --encoder-normalize-before --decoder-normalize-before \
  --arch $ARCH \
  --task $TASK \
  --lang-pairs "${src}-${tgt}" \
  --encoder-langtok "tgt" \
  --langs $LANGS \
  --domains $DOMAINS \
  --criterion $CRITERION --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-5 --warmup-updates 5000 --max-update 200000 --patience 5 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 2048 --update-freq 2  \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 1 --no-epoch-checkpoints \
  --seed 123123 --log-format simple  \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --share-decoder-input-output-embed \
  --share-all-embeddings --ddp-backend no_c10d \
  --tensorboard-logdir $SAVE_DIR/log --amp --num-workers 8


  #### Decode
model=$SAVE_DIR/checkpoint_best.pt
DATA_BIN=$path_2_data/../$domain
echo "* Eval ${domain} ${src}-${tgt}"
SAVE_DIR=.
outfile=$SAVE_DIR/test.$domain.log

python $decode_tool $DATA_BIN \
--user-dir $NMT_DIR/mdml/mdml-nmt \
--batch-size 32 \
--path $model \
--fixed-dictionary $MODEL_DICT \
--domains $domain --domain-tok $domain \
-s $src -t $tgt \
--remove-bpe 'sentencepiece' \
--beam 5 \
--task $TASK \
--langs $LANGS --lang-pairs "${src}-${tgt}" \
--gen-subset test --encoder-langtok "tgt" \
--amp | tee $outfile

pref=test.$domain
grep ^H $outfile | sort -V | cut -f 3- | sh $TOK_SCRIPT $tgt > $SAVE_DIR/$pref.detok.sys
grep ^T $outfile | sort -V | cut -f 2- | sh $TOK_SCRIPT $tgt > $SAVE_DIR/$pref.detok.ref

cat $SAVE_DIR/$pref.detok.sys | \
  sacrebleu -tok 'none' -w 2 $SAVE_DIR/$pref.detok.ref | \
  tee $SAVE_DIR/sacrebleu.$pref.log
cat $SAVE_DIR/sacrebleu.$pref.log