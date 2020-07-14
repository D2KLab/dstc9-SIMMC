#/usr/bin/bash

export DATASET_PATH=../simmc/data/simmc_fashion/fashion_train_dials.json
export METADATA_PATH=../simmc/data/simmc_fashion/fashion_metadata.json
export GLOVE_PATH=embeddings/glove.6B.50d.txt
export ACTIONS_PATH=action_annotations/fashion_train_dials_api_calls.json

pipenv shell

python main.py \
        --data  $DATASET_PATH\
        --metadata  $METADATA_PATH\
        --embeddings $GLOVE_PATH\
        --actions $ACTIONS_PATH