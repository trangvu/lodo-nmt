#!/bin/bash
src=$1
tgt=$2
exp_name=${3:-mdml-disc-adv}
hparams=${@:4}

NMT_DIR=<path to working folder>
SAVE_DIR=$NMT_DIR/models/$exp_name
data_dir=$NMT_DIR/data-bin
mkdir -p $SAVE_DIR
chmod  a+w $SAVE_DIR

LANGS='cs,de,en,es,et,fi,fr,pl,hi,it,kk,ko,lt,lv,my,ne,nl,ro,ru,si,tr'
ARCH=domain_agnostic_mbart_large
TASK=translation_multilingual_multidomain
CRITERION=domain_aware_xent_with_smoothing
DOMAINS="it,law,koran,medical,subtitles"
TOK_SCRIPT="${NMT_DIR}/lodo-nmt/m2m_100/tok.sh"
MODEL_DICT=$NMT_DIR/models/mbart.cc25/dict.txt
pretrained_model=$NMT_DIR/models/mbart.cc25/model.pt
tool=$NMT_DIR/lodo-nmt/fairseq/train.py
decode_tool=$NMT_DIR/lodo-nmt/fairseq/fairseq_cli/generate.py

python $tool $data_dir \
  --save-dir $SAVE_DIR \
  --pretrained-nmt $pretrained_model \
  --user-dir $NMT_DIR/lodo-nmt/mdml-nmt \
  --encoder-normalize-before --decoder-normalize-before \
  --arch $ARCH \
  --task $TASK \
  --sampling-method "temperature" \
  --sampling-temperature 5 \
  --langs $LANGS  --lang-pairs "${src}-${tgt}" \
  --domains $DOMAINS \
  --criterion $CRITERION --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-5 --warmup-updates 5000 --max-update 200000 --patience 10 --max-epoch 30 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 8  \
  --save-interval 1 --save-interval-updates 2000 --keep-interval-updates 1 --no-epoch-checkpoints \
  --log-format simple  \
  --share-decoder-input-output-embed \
  --share-all-embeddings --ddp-backend no_c10d \
  --tensorboard-logdir $SAVE_DIR/log --amp --num-workers 0 --num-domain 5 \
  $hparams | tee $SAVE_DIR/log.txt