#!/bin/bash
# Run this before if needed : sed -i -e 's/\r$//' run_full_experiment.sh
# Finally, run this command : sh run_full_experiment.sh\

MODEL=feminist_to_abortion # books_to_dvd
SRC_DOMAIN="${MODEL%_to_*}" # split model name according to '_to_' and take the prefix
TRG_DOMAIN="${MODEL#*_to_}" # split model name according to '_to_' and take the suffix

# Step 1 - Select pivot features
# Pivot selection params
NUM_PIVOTS=100
PIV_MN_ST=20
LOG_NAME = "log/pivot.log"

python utils/pivot_selection.py \
--pivot_num=${NUM_PIVOTS} \
--pivot_min_st=${PIV_MN_ST} \
--src=${SRC_DOMAIN} \
--dest=${TRG_DOMAIN} \
--log_name=${LOG_NAME}

# Step 2 - Run pivot-based finetuning on a pre-trained BERT
# Finetuning params
PIVOT_PROB=0.5
NON_PIVOT_PROB=0.1
NUM_PRE_TRAIN_EPOCHS=1
SAVE_FREQ=1
UNFROZEN_BERT_LAYERS=3

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
 --train_output_embeds \


 # TODO init_output_embeds is "R"-PERL?

# Step 3 - Train a classifier on source domain labeled data then predict and evaluate on target domain.
# Supervised task params
PRE_TRAINED_EPOCH=1 # TODO Was 20
CNN_OUT_CHANNELS=32
BATCH_SIZE=32
CNN_FILTER_SIZE=9
FOLD_NUM=1
NUM_TRAIN_EPOCHS=1

mkdir -p 5-fold-hyper-tune
mkdir 5-fold-hyper-tune/${MODEL}/

TEMP_DIR=models/${MODEL}/temp
mkdir -p ${TEMP_DIR}/
mkdir -p 5-fold-hyper-tune/${MODEL}/

MODELS_DIR=models/${MODEL}/

cp ${MODELS_DIR}pytorch_model${PRE_TRAINED_EPOCH}.bin ${TEMP_DIR}

python supervised_task_learning.py \
--in_domain_data_dir=data/${SRC_DOMAIN}/ \
--cross_domain_data_dir=data/${TRG_DOMAIN}/ \
--do_train \
--output_dir=${TEMP_DIR}/ \
--load_model \
--model_name=pytorch_model${PRE_TRAINED_EPOCH}.bin \
--cnn_window_size=${CNN_FILTER_SIZE} \
--cnn_out_channels=${CNN_OUT_CHANNELS} \
--learning_rate=5e-5 \
--train_batch_size=${BATCH_SIZE} \
--num_train_epochs=${NUM_TRAIN_EPOCHS} \
--save_according_to=loss \

COPY_FROM_PATH=${TEMP_DIR}/pytorch_model${PRE_TRAINED_EPOCH}.bin-final_eval_results.txt

COPY_TO_PATH=5-fold-hyper-tune/${MODEL}/ep-${PRE_TRAINED_EPOCH}_ch-${CNN_OUT_CHANNELS}_batch-${BATCH_SIZE}_filt-${CNN_FILTER_SIZE}_fold-${FOLD_NUM}.txt
cp ${COPY_FROM_PATH} ${COPY_TO_PATH}
rm ${TEMP_DIR}/*