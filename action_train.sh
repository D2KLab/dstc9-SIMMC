#/usr/bin/bash

#export MODEL=blindstateless
#export MODEL=blindstateful
export MODEL=mmstateful

export DATASET_PATH=data/simmc_fashion/train/fashion_train_dials.json
export METADATA_PATH=data/simmc_fashion/fashion_metadata.json
export EVAL_PATH=data/simmc_fashion/dev/fashion_dev_dials.json
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export METADATA_EMBEDDINGS=data/simmc_fashion/fashion_metadata_embeddings.npy
export ACTIONS_PATH=data/simmc_fashion/train/fashion_train_dials_api_calls.json
export EVAL_ACTIONS_PATH=data/simmc_fashion/dev/fashion_dev_dials_api_calls.json


python mm_action_prediction/train.py \
        --model $MODEL\
        --data $DATASET_PATH\
        --metadata $METADATA_PATH\
        --eval $EVAL_PATH\
        --embeddings $GLOVE_PATH\
        --metadata_embeddings $METADATA_EMBEDDINGS\
        --actions $ACTIONS_PATH\
        --eval_actions $EVAL_ACTIONS_PATH\
        --batch_size 256\
        --epochs 100\
        --cuda 0