#!/usr/bin/env bash

#
# IMPORTANT: this script MUST be ran from the training folder to work !!!!
#

# Data parameters
BASE_DIR="$PWD"
DATA_DIR="$BASE_DIR/data" # where are data.csv and environ.txt located
OUTPUT_DIR="$BASE_DIR/out" # where to export the model after fine-tuning
mkdir -p "$OUTPUT_DIR" # create the output dir if not exist

# Model parameters
BASE_MODEL='bert-base-german-cased'
MAX_SEQ_LENGTH=64

# copy and source environ.txt
cp "$DATA_DIR/environ.txt" "$OUTPUT_DIR" 
. "$DATA_DIR/environ.txt"         

echo "Using LABELS  $BERT_LABELS"
echo "Using WEIGHTS $BERT_LABELS_WEIGHTS"

# Train parameters
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=1.0

export PYTHONUNBUFFERED=1

# launch
command="bert_run_classifier \
  --task_name "pandas" \
  --do_train \
  --data_dir $DATA_DIR \
  --bert_model $BASE_MODEL \
  --max_seq_length $MAX_SEQ_LENGTH \
  --train_batch_size $BATCH_SIZE \
  --eval_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --output_dir $OUTPUT_DIR/bert"
  
  echo -e "USING COMMAND:\n$command\n" | tee "$OUTPUT_DIR/train_output.log"
  $command |& tee -a "$OUTPUT_DIR/train_output.log"