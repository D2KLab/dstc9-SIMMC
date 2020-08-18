#/usr/bin/bash

export MODEL=blindstateless
#export MODEL=blindstateful
#export MODEL=mmstateful

export CHECKPOINT_FOLDER=mm_response_generation/checkpoints/2020-08-17T12:31:17

export MODEL_WEIGHTS_PATH=${CHECKPOINT_FOLDER}/state_dict.pt
export VOCABULARY=${CHECKPOINT_FOLDER}/vocabulary.pkl
export DATASET_PATH=data/simmc_fashion/devtest/fashion_devtest_dials.json
export METADATA_PATH=data/simmc_fashion/fashion_metadata.json
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export METADATA_EMBEDDINGS=data/simmc_fashion/fashion_metadata_embeddings.npy
export CANDIDATES_PATH=data/simmc_fashion/devtest/fashion_devtest_dials_retrieval_candidates.json
export ACTIONS_PATH=data/simmc_fashion/devtest/fashion_devtest_dials_api_calls.json


python mm_response_generation/eval.py \
        --model $MODEL\
        --model_path $MODEL_WEIGHTS_PATH\
        --vocabulary $VOCABULARY\
        --data  $DATASET_PATH\
        --metadata  $METADATA_PATH\
        --embeddings $GLOVE_PATH\
        --metadata_embeddings $METADATA_EMBEDDINGS\
        --candidates $CANDIDATES_PATH\
        --actions $ACTIONS_PATH\
        --cuda 0

python mm_response_generation/evaluate/retrieval_evaluation.py \
        --retrieval_json_path $CANDIDATES_PATH\
        --model_score_path ${CHECKPOINT_FOLDER}/eval_out.json