#!/bin/bash
src=
tgt=
path_2_data=data-bin
SAVE_DIR=
LANGS='cs,de,en,es_XX,pl,fi_FI,fr,bg,hi_IN,it_IT,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si,tr_TR'
ARCH=domain_adapter_mbart_large
TASK=translation_multilingual_multidomain
CRITERION=label_smoothed_cross_entropy
DOMAINS="it,law,koran,medical,subtitles"
TOK_SCRIPT="m2m_100/tok.sh"
MODEL_DICT=mbart.cc25/dict.txt
DOMAIN_DICT="${src}-${tgt}:${domain}"
tool=fairseq/train.py
decode_tool=fairseq/fairseq_cli/generate.py

model=$SAVE_DIR/checkpoint_best.pt
DATA_BIN=$path_2_data/$domain
echo "* Eval ${domain} ${src}-${tgt}"
SAVE_DIR=./$backbone/$src-$tgt/$domain
outfile=$SAVE_DIR/test.$domain.log

python $decode_tool $DATA_BIN \
--user-dir $NMT_DIR/mdml/mdml-nmt \
--batch-size 32 \
--path $model \
--fixed-dictionary $MODEL_DICT \
-s $src -t $tgt \
--domains $DOMAINS --domain-tok $domain \
--remove-bpe 'sentencepiece' \
--beam 5 \
--task $TASK \
--lang-pairs "${src}-${tgt}" --langs $LANGS \
--encoder-langtok "tgt" --keep-inference-langtok \
--gen-subset test \
--amp | tee $outfile

pref=test.$domain
grep ^H $outfile | sort -V | cut -f 3- | sh $TOK_SCRIPT $tgt > $SAVE_DIR/$pref.detok.sys
grep ^T $outfile | sort -V | cut -f 2- | sh $TOK_SCRIPT $tgt > $SAVE_DIR/$pref.detok.ref

cat $SAVE_DIR/$pref.detok.sys | \
  sacrebleu -tok 'none' -w 2 $SAVE_DIR/$pref.detok.ref | \
  tee $SAVE_DIR/sacrebleu.$pref.log
cat $SAVE_DIR/sacrebleu.$pref.log