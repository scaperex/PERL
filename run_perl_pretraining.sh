#!/bin/bash
# Run this before if needed : sed -i -e 's/\r$//' run_perl_pretraining.sh
# Finally, run this command : sh run_perl_pretraining.sh

PIVOT_PROB=0.5
NON_PIVOT_PROB=0.1
NUM_PRE_TRAIN_EPOCHS=1
SAVE_FREQ=1
UNFROZEN_BERT_LAYERS=3
NUM_PIVOTS=100

for MODEL in feminist_to_abortion
do
  SRC_DOMAIN="${MODEL%_to_*}" # split model name according to '_to_' and take the prefix
  TRG_DOMAIN="${MODEL#*_to_}" # split model name according to '_to_' and take the suffix

  mkdir -p models/${MODEL}

  OUTPUT_DIR_NAME=models/${MODEL}
  PIVOTS_PATH=data/pivots/${MODEL}/${NUM_PIVOTS}_bi

  python perl_pretrain.py \
   --src_domain=${SRC_DOMAIN} \
   --trg_domain=${TRG_DOMAIN} \
   --pivot_path=${PIVOTS_PATH} \
   --output_dir=${OUTPUT_DIR_NAME} \
   --num_train_epochs=${NUM_PRE_TRAIN_EPOCHS} \
   --save_every_num_epochs=${SAVE_FREQ} \
   --pivot_prob=${PIVOT_PROB} \
   --non_pivot_prob=${NON_PIVOT_PROB} \
   --num_of_unfrozen_bert_layers=${UNFROZEN_BERT_LAYERS} \
   --init_output_embeds \
   --train_output_embeds



done
