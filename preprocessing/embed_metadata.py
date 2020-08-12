import argparse
import json
import pdb
import re

import numpy as np
from nltk.tokenize import WordPunctTokenizer

FIELDS_TO_EMBED = ['type', 'color', 'embellishments', 'pattern', 'brand']



def load_embeddings_from_file(embeddings_path):
        glove = {}
        with open(embeddings_path) as fp:
            for l in fp:
                line_tokens = l.split()
                word = line_tokens[0]
                if word in glove:
                    raise Exception('Repeated words in {} embeddings file'.format(embeddings_path))
                vector = np.asarray(line_tokens[1:], "float32")
                glove[word] = vector
        embedding_size = vector.size
        return glove, embedding_size


def clean_value(value, tokenizer):
    results = []
    tokenized_val = tokenizer.tokenize(value.lower())
    for v in tokenized_val:
        results.extend(re.split('_|-', v))
    return results


def extract_metadata_embeddings(metadata_path, embeddings_path, save_path):
    
    with open(metadata_path) as fp:
        metadata_dict = json.load(fp)

    glove, embedding_size = load_embeddings_from_file(embeddings_path=embeddings_path)

    item_ids = []
    item_embeddings = []
    tokenizer = WordPunctTokenizer() 
    for item_id, item in metadata_dict.items():
        fields_embeddings = []
        for field in FIELDS_TO_EMBED:
            assert field in item['metadata'], '{} field not in item {}'.format(field, item_id)

            if isinstance(item['metadata'][field], list,):
                for value in item['metadata'][field]:
                    cleaned_values.extend(clean_value(value, tokenizer))
            else:
                cleaned_values = clean_value(item['metadata'][field], tokenizer)
            emb = []
            for v in cleaned_values:
                if v in glove:
                    emb.append(np.array(glove[v]))
                else:
                    emb.append(np.random.rand(300,))
            emb = np.stack(emb)
            fields_embeddings.append(emb.mean(0))
            assert fields_embeddings[-1].size == embedding_size, 'Wrong embedding dimension'

        assert len(fields_embeddings) == len(FIELDS_TO_EMBED), 'Wrong number of embeddings'
        item_ids.append(item_id)
        item_embeddings.append(np.concatenate(fields_embeddings))
    
    assert len(item_ids) == len(item_embeddings), 'Item ids list does not match item embeddings'
    np.save(
        save_path,
        {
            'embedding_size': embedding_size*len(FIELDS_TO_EMBED),
            'item_ids': item_ids,
            'embeddings': np.stack(item_embeddings)
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata JSON file")
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to embeddings file"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path where to save the embeddings"
    )

    args = parser.parse_args()
    extract_metadata_embeddings(args.metadata, args.embeddings, args.save_path)
