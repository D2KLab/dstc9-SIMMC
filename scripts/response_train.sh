#/usr/bin/bash

#export MODEL=blindstateless
#export MODEL=blindstateful
export MODEL=mmstateful

export DATASET_PATH=data/simmc_fashion/train/response_retrieval_data.dat #todo change to train folder
export EVAL_PATH=data/simmc_fashion/dev/response_retrieval_data.dat
export METADATA_IDS_PATH=data/simmc_fashion/metadata_ids.dat
export GEN_VOCAB=data/simmc_fashion/generative_vocab.dat
export CHECKPOINT_FOLDER=mm_response_generation/checkpoints/2020-09-21T15:59:06


python mm_response_generation/train.py \
        --model $MODEL\
        --data $DATASET_PATH\
        --eval $EVAL_PATH\
        --metadata_ids $METADATA_IDS_PATH\
        --generative_vocab $GEN_VOCAB\
        --batch_size 512\
        --epochs 200\
        --checkpoints\
        --cuda