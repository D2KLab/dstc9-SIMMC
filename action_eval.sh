#/usr/bin/bash

export MODEL=blindstateless
#export MODEL=blindstateful
#export MODEL=mmstateful

export CHECKPOINT_FOLDER=mm_action_prediction/checkpoints/2020-08-23T17:33:26

export MODEL_WEIGHTS_PATH=${CHECKPOINT_FOLDER}/state_dict.pt
export VOCABULARY=${CHECKPOINT_FOLDER}/vocabulary.pkl
export DATASET_PATH=data/simmc_fashion/devtest/action_prediction_data.dat
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export METADATA_EMBEDDINGS=data/simmc_fashion/fashion_metadata_embeddings.npy


python mm_action_prediction/eval.py \
        --model $MODEL\
        --model_path $MODEL_WEIGHTS_PATH\
        --vocabulary $VOCABULARY\
        --data $DATASET_PATH\
        --embeddings $GLOVE_PATH\
        --metadata_embeddings $METADATA_EMBEDDINGS\
        --cuda 0

python mm_action_prediction/scripts/action_evaluation.py \
        --action_json_path data/simmc_fashion/devtest/fashion_devtest_dials_api_calls.json\
        --model_output_path ${CHECKPOINT_FOLDER}/eval_out.json
