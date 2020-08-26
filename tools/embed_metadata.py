import argparse
import json
import pdb
import re
import string

import numpy as np
from nltk.tokenize import WordPunctTokenizer

from simmc_dataset import SIMMCDatasetForResponseGeneration

# for single embedding
FIELDS_TO_EMBED = ['type', 'color', 'embellishments', 'pattern', 'brand']
FIELD2STR = SIMMCDatasetForResponseGeneration._ATTR2STR



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


def extract_single_metadata_embeddings(metadata_path, embeddings_path, save_path):
    
    with open(metadata_path) as fp:
        metadata_dict = json.load(fp)

    glove, embedding_size = load_embeddings_from_file(embeddings_path=embeddings_path)

    item_embeddings = {}
    tokenizer = WordPunctTokenizer() 
    for item_id, item in metadata_dict.items():
        fields_embeddings = []
        for field in FIELDS_TO_EMBED:
            assert field in item['metadata'], '{} field not in item {}'.format(field, item_id)
            cleaned_values = []
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
                    print('Unknown word \'{}\' initiated with a random embedding'.format(v))
            emb = np.stack(emb)
            fields_embeddings.append(emb.mean(0))
            assert fields_embeddings[-1].size == embedding_size, 'Wrong embedding dimension'

        assert len(fields_embeddings) == len(FIELDS_TO_EMBED), 'Wrong number of embeddings'
        item_embeddings[item_id] = np.concatenate(fields_embeddings)

    np.save(
        save_path,
        {
            'embedding_size': embedding_size*len(FIELDS_TO_EMBED),
            'embeddings': item_embeddings
        }
    )


def extract_list_metadata_embeddings(metadata_path, embeddings_path, save_path):
    with open(metadata_path) as fp:
        metadata_dict = json.load(fp)

    glove, embedding_size = load_embeddings_from_file(embeddings_path=embeddings_path)

    unknown_words = set()
    item_ids = []
    item_embeddings = []
    tokenizer = WordPunctTokenizer()
    for item_id, item in metadata_dict.items():
        for key in item['metadata']:
            # availability field is always an empty list
            if key == 'availability':
                continue
            field_name = FIELD2STR[key.lower()] if key.lower() in FIELD2STR else key.lower()
            field_tokens = clean_value(field_name, tokenizer)
            cleaned_values = []
            if isinstance(item['metadata'][key], list,):
                if not len(item['metadata'][key]):
                    cleaned_values.extend('none') #for empty lists
                for value in item['metadata'][key]:
                    cleaned_values.extend(clean_value(value, tokenizer))
            else:
                cleaned_values = clean_value(item['metadata'][key], tokenizer)

            fields_emb = []
            for t in field_tokens:
                if t in glove:
                    fields_emb.append(np.array(glove[t]))
                else:
                    if t in string.punctuation:
                        continue
                    fields_emb.append(np.random.rand(300,))
                    unknown_words.add(t)
            values_emb = []
            for v in cleaned_values:
                if v in glove:
                    values_emb.append(np.array(glove[v]))
                else:
                    if v in string.punctuation:
                        continue
                    values_emb.append(np.random.rand(300,))
                    unknown_words.add(v)
            item_ids.append(item_id)
            item_embeddings.append((np.stack(fields_emb).mean(0), np.stack(values_emb).mean(0)))
    print('UNKNOWN WORDS: {}'.format(unknown_words))

    np.save(
        save_path,
        {
            'embedding_size': embedding_size,
            'item_ids': item_ids,
            'embeddings': item_embeddings
        }
    )
    print('embeddings saved in {}'.format(save_path))



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
    parser.add_argument(
        "--type",
        type=str,
        choices=['single', 'list'],
        required=True,
        help="Type of embedding for each item (options: 'single', 'list')"
    )

    args = parser.parse_args()
    if args.type == 'single':
        extract_single_metadata_embeddings(args.metadata, args.embeddings, args.save_path)
    else:
        extract_list_metadata_embeddings(args.metadata, args.embeddings, args.save_path)
