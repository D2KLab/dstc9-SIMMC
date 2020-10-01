#/usr/bin/bash

export SIMMC_FOLDER=simmc/data/simmc_fashion
export PROCESSED_FOLDER=processed_data

mkdir $PROCESSED_FOLDER
mkdir $PROCESSED_FOLDER/mm_response_generation


python mm_response_generation/preprocessing_copy.py\
           --simmc_folder $SIMMC_FOLDER\
            --actions_folder $PROCESSED_FOLDER/mm_response_generation\
            --metadata $SIMMC_FOLDER/fashion_metadata.json\
            --save_path $PROCESSED_FOLDER/mm_response_generation


export MODEL=mmstateful
export GEN_EVAL_SCRIPT=simmc/mm_action_prediction/tools/response_evaluation.py
export RETR_EVAL_SCRIPT=simmc/mm_action_prediction/tools/retrieval_evaluation.py

export PROCESSED_FOLDER=processed_data/mm_response_generation
export CHECKPOINT_FOLDER=mm_response_generation/model_params
export MODEL_WEIGHTS_PATH=${CHECKPOINT_FOLDER}/state_dict.pt
export VOCABULARY=${CHECKPOINT_FOLDER}/bert2genid.pkl
export MODEL_CONF=${CHECKPOINT_FOLDER}/model_conf.json

export DATASET_PATH=$PROCESSED_FOLDER/teststd_response_retrieval_data.dat
export METADATA_IDS_PATH=$PROCESSED_FOLDER/metadata_ids.dat


python mm_response_generation/eval.py\
        --model $MODEL\
        --model_path $MODEL_WEIGHTS_PATH\
        --vocabulary $VOCABULARY\
        --model_conf $MODEL_CONF\
        --data  $DATASET_PATH\
        --metadata_ids $METADATA_IDS_PATH\
        --beam_size 5\
        --retrieval_eval\
        --cuda 0