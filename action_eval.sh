#/usr/bin/bash

export MODEL=action_prediction/checkpoints/2020-08-07T16:12:45/state_dict.pt
export VOCABULARY=action_prediction/checkpoints/2020-08-07T16:12:45/vocabulary.pkl
export DATASET_PATH=../simmc/data/simmc_fashion/fashion_devtest_dials.json
export METADATA_PATH=../simmc/data/simmc_fashion/fashion_metadata.json
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export ACTIONS_PATH=action_prediction/action_annotations/fashion_devtest_dials_api_calls.json


python action_prediction/eval.py \
        --model $MODEL\
        --vocabulary $VOCABULARY\
        --data  $DATASET_PATH\
        --metadata  $METADATA_PATH\
        --embeddings $GLOVE_PATH\
        --actions $ACTIONS_PATH\
        --cuda 0