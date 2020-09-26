#/usr/bin/bash

#export MODEL=blindstateless
#export MODEL=blindstateful
export MODEL=mmstateful

export PROCESSED_FOLDER=processed_data/mm_response_generation
export TRAIN_DATA=$PROCESSED_FOLDER/train_response_retrieval_data.dat
export EVAL_DATA=$PROCESSED_FOLDER/dev_response_retrieval_data.dat

export METADATA_IDS_PATH=$PROCESSED_FOLDER/metadata_ids.dat
export GEN_VOCAB=$PROCESSED_FOLDER/generative_vocab.dat
export CHECKPOINT_FOLDER=mm_response_generation/checkpoints/2020-09-21T15:59:06


python mm_response_generation/train.py \
        --model $MODEL\
        --data $TRAIN_DATA\
        --eval $EVAL_DATA\
        --metadata_ids $METADATA_IDS_PATH\
        --generative_vocab $GEN_VOCAB\
        --batch_size 512\
        --epochs 200\
        --checkpoints\
        --cuda