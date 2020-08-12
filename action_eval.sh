#/usr/bin/bash

export MODEL=mm_action_prediction/checkpoints/2020-08-07T16:12:45/state_dict.pt
export VOCABULARY=mm_action_prediction/checkpoints/2020-08-07T16:12:45/vocabulary.pkl
export DATASET_PATH=data/simmc_fashion/devtest/fashion_devtest_dials.json
export METADATA_PATH=data/simmc_fashion/fashion_metadata.json
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export METADATA_EMBEDDINGS=data/simmc_fashion/fashion_metadata_embeddings.npy
export ACTIONS_PATH=data/simmc_fashion/devtest/fashion_devtest_dials_api_calls.json


python mm_action_prediction/eval.py \
        --model $MODEL\
        --vocabulary $VOCABULARY\
        --data  $DATASET_PATH\
        --metadata  $METADATA_PATH\
        --embeddings $GLOVE_PATH\
        --metadata_embeddings $METADATA_EMBEDDINGS\
        --actions $ACTIONS_PATH\
        --cuda 0