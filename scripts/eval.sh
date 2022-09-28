model=$SAVE_DIR/checkpoint_best.pt
DATA_BIN=$path_2_data/$domain
echo "* Eval ${domain} ${src}-${tgt}"
SAVE_DIR=.
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
--encoder-langtok "tgt" --tag-at-decoder --add-domain-tags \
--gen-subset test \
--amp | tee $outfile

pref=test.$domain
grep ^H $outfile | sort -V | cut -f 3- | sh $TOK_SCRIPT $tgt > $SAVE_DIR/$pref.detok.sys
grep ^T $outfile | sort -V | cut -f 2- | sh $TOK_SCRIPT $tgt > $SAVE_DIR/$pref.detok.ref

cat $SAVE_DIR/$pref.detok.sys | \
  sacrebleu -tok 'none' -w 2 $SAVE_DIR/$pref.detok.ref | \
  tee $SAVE_DIR/sacrebleu.$pref.log
cat $SAVE_DIR/sacrebleu.$pref.log



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
--decoder-langtok --encoder-langtok src --gen-subset test \
--amp | tee $outfile