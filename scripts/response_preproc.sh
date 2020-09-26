#/usr/bin/bash



export SIMMC_FASHION_FOLDER=simmc/data/simmc_fashion
export PROCESSED_FOLDER=processed_data

mkdir $PROCESSED_FOLDER
mkdir $PROCESSED_FOLDER/mm_response_generation

python simmc/mm_action_prediction/tools/extract_actions_fashion.py\
            --json_path $SIMMC_FASHION_FOLDER/fashion_train_dials.json\
            --save_root $PROCESSED_FOLDER/mm_response_generation\
            --metadata_path $SIMMC_FASHION_FOLDER/fashion_metadata.json

python simmc/mm_action_prediction/tools/extract_actions_fashion.py\
            --json_path $SIMMC_FASHION_FOLDER/fashion_dev_dials.json\
            --save_root $PROCESSED_FOLDER/mm_response_generation\
            --metadata_path $SIMMC_FASHION_FOLDER/fashion_metadata.json

python simmc/mm_action_prediction/tools/extract_actions_fashion.py\
            --json_path $SIMMC_FASHION_FOLDER/fashion_devtest_dials.json\
            --save_root $PROCESSED_FOLDER/mm_response_generation\
            --metadata_path $SIMMC_FASHION_FOLDER/fashion_metadata.json

python mm_response_generation/preprocessing.py\
            --simmc_folder $SIMMC_FASHION_FOLDER\
            --actions_folder $PROCESSED_FOLDER/mm_response_generation\
            --metadata $SIMMC_FASHION_FOLDER/fashion_metadata.json\
            --save_path $PROCESSED_FOLDER/mm_response_generation
