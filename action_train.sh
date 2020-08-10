#/usr/bin/bash

export DATASET_PATH=data/simmc_fashion/fashion_train_dials.json
export METADATA_PATH=data/simmc_fashion/fashion_metadata.json
export EVAL_PATH=data/simmc_fashion/fashion_dev_dials.json
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export ACTIONS_PATH=action_prediction/action_annotations/fashion_train_dials_api_calls.json
export EVAL_ACTIONS_PATH=action_prediction/action_annotations/fashion_dev_dials_api_calls.json


python action_prediction/train.py \
        --data $DATASET_PATH\
        --metadata $METADATA_PATH\
        --eval $EVAL_PATH\
        --embeddings $GLOVE_PATH\
        --actions $ACTIONS_PATH\
        --eval_actions $EVAL_ACTIONS_PATH\
        --batch_size 32\
        --epochs 20\
        --cuda 0