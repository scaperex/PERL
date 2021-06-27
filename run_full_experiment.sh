#!/bin/bash
# Run this before if needed : sed -i -e 's/\r$//' run_full_experiment.sh
# Finally, run this command : sh run_full_experiment.sh\

# UNLABELED-SIZE (6000-35,000) ---> 20000

# NUM_PIVOTS (100, 200,300, 400, 500) ---> 100

# NUM_PRE_TRAIN_EPOCHS (20, 40, 60) ---> 20
# UNFROZEN_BERT_LAYERS (1, 2, 3, 5, 8, 12) ---> [2, 8]
# PIVOT_PROB (0.1, 0.3, 0.5,0.8) ---> 0.5
# NON_PIVOT_PROB (0.1, 0.3, 0.5,0.8) ---> 0.1

# PRE_TRAINED_EPOCH
# CNN_OUT_CHANNELS (16, 32, 64) ---> 32
# CNN_FILTER_SIZE (7, 9, 11) --->  9
# BATCH_SIZE (32, 64) ---> 32
# NUM_TRAIN_EPOCHS ---> 20

DATA_DIR=stancedata
MODEL_DIR=stancemodels

#DATA_DIR=data
#MODEL_DIR=models

# before running - delete the right model dir

for MODEL in feminist_to_abortion  abortion_to_feminist feminist_to_atheism atheism_to_feminist
do
  SRC_DOMAIN="${MODEL%_to_*}" # split model name according to '_to_' and take the prefix
  TRG_DOMAIN="${MODEL#*_to_}" # split model name according to '_to_' and take the suffix
  MODELS_DIR=${MODEL_DIR}/${MODEL}

  # Step 1 - Select pivot features
  # Pivot selection params
  NUM_PIVOTS=100
  PIV_MN_ST=20
  LOG_NAME="log/pivot.log"

  python utils/pivot_selection.py \
  --pivot_num=${NUM_PIVOTS} \
  --pivot_min_st=${PIV_MN_ST} \
  --src=${DATA_DIR}/${SRC_DOMAIN} \
  --dest=${DATA_DIR}/${TRG_DOMAIN} \
  --log_name=${LOG_NAME} \
  --tokenizer_name=None

  # Step 2 - Run pivot-based finetuning on a pre-trained BERT
  # Finetuning params
  PIVOT_PROB=0.5
  NON_PIVOT_PROB=0.1
  NUM_PRE_TRAIN_EPOCHS=10
  SAVE_FREQ=${NUM_PRE_TRAIN_EPOCHS}
  UNFROZEN_BERT_LAYERS=2

  mkdir -p ${MODELS_DIR}

  OUTPUT_DIR_NAME=${MODELS_DIR}
  PIVOTS_PATH=${DATA_DIR}/pivots/${MODEL}/${NUM_PIVOTS}_bi

  python perl_pretrain.py \
   --src_domain=${DATA_DIR}/${SRC_DOMAIN} \
   --trg_domain=${DATA_DIR}/${TRG_DOMAIN} \
   --pivot_path=${PIVOTS_PATH} \
   --output_dir=${OUTPUT_DIR_NAME} \
   --num_train_epochs=${NUM_PRE_TRAIN_EPOCHS} \
   --save_every_num_epochs=${SAVE_FREQ} \
   --pivot_prob=${PIVOT_PROB} \
   --non_pivot_prob=${NON_PIVOT_PROB} \
   --num_of_unfrozen_bert_layers=${UNFROZEN_BERT_LAYERS} \
   --init_output_embeds \
   --train_output_embeds

   # TODO init_output_embeds is "R"-PERL?

  # Step 3 - Train a classifier on source domain labeled data then predict and evaluate on target domain.
  # Supervised task params
  PRE_TRAINED_EPOCH=${NUM_PRE_TRAIN_EPOCHS}
  CNN_OUT_CHANNELS=32
  BATCH_SIZE=32
  CNN_FILTER_SIZE=9
  FOLD_NUM=1
  NUM_TRAIN_EPOCHS=5

  mkdir -p 5-fold-hyper-tune
  mkdir 5-fold-hyper-tune/${MODEL}/

  TEMP_DIR=${MODELS_DIR}/temp
  mkdir -p ${TEMP_DIR}/
  mkdir -p 5-fold-hyper-tune/${MODEL}/

  cp ${MODELS_DIR}/pytorch_model${PRE_TRAINED_EPOCH}.bin ${TEMP_DIR}

  python supervised_task_learning.py \
  --in_domain_data_dir=${DATA_DIR}/${SRC_DOMAIN}/ \
  --cross_domain_data_dir=${DATA_DIR}/${TRG_DOMAIN}/ \
  --do_train \
  --output_dir=${TEMP_DIR}/ \
  --load_model \
  --model_name=pytorch_model${PRE_TRAINED_EPOCH}.bin \
  --cnn_window_size=${CNN_FILTER_SIZE} \
  --cnn_out_channels=${CNN_OUT_CHANNELS} \
  --learning_rate=5e-5 \
  --train_batch_size=${BATCH_SIZE} \
  --num_train_epochs=${NUM_TRAIN_EPOCHS} \
  --save_according_to=acc \
  --write_log_for_each_epoch

  COPY_FROM_PATH=${TEMP_DIR}/pytorch_model${PRE_TRAINED_EPOCH}.bin-final_eval_results.txt

  COPY_TO_PATH=5-fold-hyper-tune/${MODEL}/ep-${PRE_TRAINED_EPOCH}_ch-${CNN_OUT_CHANNELS}_batch-${BATCH_SIZE}_filt-${CNN_FILTER_SIZE}_fold-${FOLD_NUM}.txt
  cp ${COPY_FROM_PATH} ${COPY_TO_PATH}
  rm ${TEMP_DIR}/*

done