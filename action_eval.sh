#/usr/bin/bash

#export MODEL=blindstateless
export MODEL=blindstateful
#export MODEL=mmstateful

export CHECKPOINT_FOLDER=mm_action_prediction/checkpoints/2020-08-14T16:00:08

export MODEL_WEIGHTS_PATH=${CHECKPOINT_FOLDER}/state_dict.pt
export VOCABULARY=${CHECKPOINT_FOLDER}/vocabulary.pkl
export DATASET_PATH=data/simmc_fashion/devtest/fashion_devtest_dials.json
export METADATA_PATH=data/simmc_fashion/fashion_metadata.json
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export METADATA_EMBEDDINGS=data/simmc_fashion/fashion_metadata_embeddings.npy
export ACTIONS_PATH=data/simmc_fashion/devtest/fashion_devtest_dials_api_calls.json


python mm_action_prediction/eval.py \
        --model $MODEL\
        --model_path $MODEL_WEIGHTS_PATH\
        --vocabulary $VOCABULARY\
        --data  $DATASET_PATH\
        --metadata  $METADATA_PATH\
        --embeddings $GLOVE_PATH\
        --metadata_embeddings $METADATA_EMBEDDINGS\
        --actions $ACTIONS_PATH\
        --cuda 0

python mm_action_prediction/scripts/action_evaluation.py \
        --action_json_path data/simmc_fashion/devtest/fashion_devtest_dials_api_calls.json\
        --model_output_path ${CHECKPOINT_FOLDER}/eval_out.json
