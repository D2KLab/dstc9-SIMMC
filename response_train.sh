#/usr/bin/bash

export MODEL=blindstateless
#export MODEL=blindstateful
#export MODEL=mmstateful

export DATASET_PATH=data/simmc_fashion/train/response_retrieval_data.dat
export EVAL_PATH=data/simmc_fashion/dev/response_retrieval_data.dat
export VOCABULARY=data/simmc_fashion/vocabulary.npy
export GLOVE_PATH=embeddings/glove.6B.300d.txt
export METADATA_EMBEDDINGS=data/simmc_fashion/fashion_metadata_embeddings.npy


python mm_response_generation/train.py \
        --model $MODEL\
        --data $DATASET_PATH\
        --eval $EVAL_PATH\
        --vocabulary $VOCABULARY\
        --embeddings $GLOVE_PATH\
        --metadata_embeddings $METADATA_EMBEDDINGS\
        --batch_size 128\
        --epochs 50\
        --cuda 0