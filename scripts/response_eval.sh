#/usr/bin/bash

#export MODEL=blindstateless
#export MODEL=blindstateful
export MODEL=mmstateful

export CHECKPOINT_FOLDER=mm_response_generation/checkpoints/2020-09-03T16:51:56

export MODEL_WEIGHTS_PATH=${CHECKPOINT_FOLDER}/state_dict.pt
export VOCABULARY=${CHECKPOINT_FOLDER}/vocabulary.pkl
export DATASET_PATH=data/simmc_fashion/devtest/response_retrieval_data.dat
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export METADATA_IDS_PATH=data/simmc_fashion/metadata_ids.dat


python mm_response_generation/eval.py\
        --model $MODEL\
        --model_path $MODEL_WEIGHTS_PATH\
        --vocabulary $VOCABULARY\
        --data  $DATASET_PATH\
        --embeddings $GLOVE_PATH\
        --metadata_ids $METADATA_IDS_PATH\
        --cuda 0

python mm_response_generation/utilities/retrieval_evaluation.py \
        --retrieval_json_path data/simmc_fashion/devtest/fashion_devtest_dials_retrieval_candidates.json\
        --model_score_path ${CHECKPOINT_FOLDER}/eval_out.json