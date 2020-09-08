#/usr/bin/bash

#export MODEL=blindstateless
#export MODEL=blindstateful
export MODEL=mmstateful

export DATASET_PATH=data/simmc_fashion/dev/response_retrieval_data.dat #todo change to train folder
export EVAL_PATH=data/simmc_fashion/dev/response_retrieval_data.dat
export VOCABULARY=data/simmc_fashion/vocabulary.npy
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export METADATA_IDS_PATH=data/simmc_fashion/metadata_ids.dat


python mm_response_generation/train.py \
        --model $MODEL\
        --data $DATASET_PATH\
        --eval $EVAL_PATH\
        --vocabulary $VOCABULARY\
        --embeddings $GLOVE_PATH\
        --metadata_ids $METADATA_IDS_PATH\
        --batch_size 2\
        --epochs 15\
        --mode generation\
        --cuda