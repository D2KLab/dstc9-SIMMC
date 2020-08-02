#/usr/bin/bash

export DATASET_PATH=../simmc/data/simmc_fashion/fashion_train_dials.json
export METADATA_PATH=../simmc/data/simmc_fashion/fashion_metadata.json
export EVAL_PATH=../simmc/data/simmc_fashion/fashion_dev_dials.json
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export ACTIONS_PATH=annotations/fashion_train_dials_api_calls.json
export EVAL_ACTIONS_PATH=annotations/fashion_dev_dials_api_calls.json


python train.py \
        --data  $DATASET_PATH\
        --metadata  $METADATA_PATH\
        --eval $EVAL_PATH\
        --embeddings $GLOVE_PATH\
        --actions $ACTIONS_PATH\
        --eval_actions $EVAL_ACTIONS_PATH\
        --batch_size 4\
        --cuda 0