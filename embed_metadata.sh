

export METADATA_PATH=data/simmc_fashion/fashion_metadata.json
export EMBEDDINGS_PATH=embeddings/glove.6B.300d.txt
export SAVE_PATH=data/simmc_fashion/fashion_metadata_embeddings.npy


python preprocessing/embed_metadata.py\
        --metadata $METADATA_PATH\
        --embeddings $EMBEDDINGS_PATH\
        --save_path $SAVE_PATH