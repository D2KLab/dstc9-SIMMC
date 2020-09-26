#/usr/bin/bash

export SIMMC_FASHION_FOLDER=simmc/data/simmc_fashion
export PROCESSED_FOLDER=processed_data
export EMBEDDINGS_PATH=embeddings/glove.6B.300d.txt
export METADATA_EMBEDDINGS=data/simmc_fashion/fashion_metadata_embeddings.npy

mkdir $PROCESSED_FOLDER
mkdir $PROCESSED_FOLDER/mm_action_prediction

python simmc/mm_action_prediction/tools/extract_actions_fashion.py\
            --json_path $SIMMC_FASHION_FOLDER/fashion_train_dials.json\
            --save_root $PROCESSED_FOLDER/mm_action_prediction\
            --metadata_path $SIMMC_FASHION_FOLDER/fashion_metadata.json

python simmc/mm_action_prediction/tools/extract_actions_fashion.py\
            --json_path $SIMMC_FASHION_FOLDER/fashion_dev_dials.json\
            --save_root $PROCESSED_FOLDER/mm_action_prediction\
            --metadata_path $SIMMC_FASHION_FOLDER/fashion_metadata.json

python simmc/mm_action_prediction/tools/extract_actions_fashion.py\
            --json_path $SIMMC_FASHION_FOLDER/fashion_devtest_dials.json\
            --save_root $PROCESSED_FOLDER/mm_action_prediction\
            --metadata_path $SIMMC_FASHION_FOLDER/fashion_metadata.json

python mm_action_prediction/preprocessing.py\
            --simmc_folder $SIMMC_FASHION_FOLDER\
            --actions_folder $PROCESSED_FOLDER/mm_action_prediction\
            --embeddings $EMBEDDINGS_PATH\
            --metadata_embeddings $METADATA_EMBEDDINGS\
            --metadata $SIMMC_FASHION_FOLDER/fashion_metadata.json