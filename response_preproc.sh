#/usr/bin/bash


export TRAIN_FOLDER=data/simmc_fashion/train
export DEV_FOLDER=data/simmc_fashion/dev
export DEVTEST_FOLDER=data/simmc_fashion/devtest
export EMBEDDINGS_PATH=embeddings/glove.6B.300d.txt
export METADATA_EMBEDDINGS=data/simmc_fashion/fashion_metadata_embeddings.npy
export METADATA_PATH=data/simmc_fashion/fashion_metadata.json


python mm_response_generation/preprocessing.py\
            --train_folder $TRAIN_FOLDER\
            --dev_folder $DEV_FOLDER\
            --test_folder $DEVTEST_FOLDER\
            --metadata $METADATA_PATH\
            --embeddings $EMBEDDINGS_PATH\
            --metadata_embeddings $METADATA_EMBEDDINGS
