#/usr/bin/bash

export MODEL=checkpoints/2020-08-03T15:38:52/state_dict.pt
export VOCABULARY=checkpoints/2020-08-03T15:38:52/vocabulary.pkl
export DATASET_PATH=../simmc/data/simmc_fashion/fashion_devtest_dials.json
export METADATA_PATH=../simmc/data/simmc_fashion/fashion_metadata.json
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export ACTIONS_PATH=annotations/fashion_devtest_dials_api_calls.json


python eval.py \
        --model $MODEL\
        --vocabulary $VOCABULARY\
        --data  $DATASET_PATH\
        --metadata  $METADATA_PATH\
        --embeddings $GLOVE_PATH\
        --actions $ACTIONS_PATH\
        --cuda 0