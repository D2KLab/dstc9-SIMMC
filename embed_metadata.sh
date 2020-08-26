

export METADATA_PATH=data/simmc_fashion/fashion_metadata.json
export TYPE=list
export EMBEDDINGS_PATH=embeddings/glove.6B.300d.txt
export SAVE_PATH=data/simmc_fashion/fashion_metadata_${TYPE}_embeddings.npy

python tools/embed_metadata.py\
        --type $TYPE\
        --metadata $METADATA_PATH\
        --embeddings $EMBEDDINGS_PATH\
        --save_path $SAVE_PATH