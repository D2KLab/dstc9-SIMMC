#/usr/bin/bash

#export MODEL=blindstateless
#export MODEL=blindstateful
export MODEL=mmstateful

export SIMMC_FOLDER=simmc/data/simmc_fashion
export GEN_EVAL_SCRIPT=simmc/mm_action_prediction/tools/response_evaluation.py
export RETR_EVAL_SCRIPT=simmc/mm_action_prediction/tools/retrieval_evaluation.py

export PROCESSED_FOLDER=processed_data/mm_response_generation
export CHECKPOINT_FOLDER=mm_response_generation/checkpoints/archive/2020-09-21T15:59:06
export MODEL_WEIGHTS_PATH=${CHECKPOINT_FOLDER}/state_dict.pt
export VOCABULARY=${CHECKPOINT_FOLDER}/bert2genid.pkl
export MODEL_CONF=${CHECKPOINT_FOLDER}/model_conf.json

export DATASET_PATH=$PROCESSED_FOLDER/devtest_response_retrieval_data.dat
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
        --gen_eval\
        --cuda 0

#python $GEN_EVAL_SCRIPT\
#        --data_json_path data/simmc_fashion/devtest/fashion_devtest_dials.json \
#        --model_response_path  ${CHECKPOINT_FOLDER}/eval_gen.json > ${CHECKPOINT_FOLDER}/gen_scores.txt

python $RETR_EVAL_SCRIPT\
        --retrieval_json_path $SIMMC_FOLDER/fashion_devtest_dials_retrieval_candidates.json\
        --model_score_path ${CHECKPOINT_FOLDER}/eval_retr.json > ${CHECKPOINT_FOLDER}/retr_scores.txt

cat ${CHECKPOINT_FOLDER}/gen_scores.txt
cat ${CHECKPOINT_FOLDER}/retr_scores.txt