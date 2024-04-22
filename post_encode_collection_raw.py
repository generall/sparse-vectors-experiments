from itertools import count
import math
from tqdm import tqdm
import json
from typing import Counter, Iterable
import os
import time

from remap_tokens import aggregate_weights, calc_tf, filter_list_tokens, filter_pair_tokens, reconstruct_bpe, snowball_tokenize, stem_list_tokens, stem_pair_tokens


DATASET = os.getenv("DATASET", "quora")


def read_corpus_file(file_name: str) -> Iterable[str]:
    with open(file_name, "r") as file:
        for line in file:
            row = json.loads(line)
            yield row["_id"], row["text"]


def read_raw_encoded_file(file_name: str) -> Iterable[dict]:
    with open(file_name, "r") as file:
        for line in file:
            yield json.loads(line)


def rescore_vector(vector: dict) -> dict:

    sorted_vector = sorted(vector.items(), key=lambda x: x[1], reverse=True)

    new_vector = {}

    for num, (token, value) in enumerate(sorted_vector):
        new_vector[token] = math.log(4. / (num + 1.) + 1.) # value

    return new_vector

def main():
    corpus_file_name = f"data/{DATASET}/corpus.jsonl"  # MS MARCO collection
    # output file
    raw_vectors_file_name = f"data/{DATASET}/collection_vectors_raw.jsonl"

    # output file
    file_out = f"data/{DATASET}/collection_vectors.jsonl"

    total_tokens_overall = 0
    num_docs = 0

    with open(file_out, "w") as out_file:
        for (raw_vectors, (idx, text)) in tqdm(zip(read_raw_encoded_file(raw_vectors_file_name), read_corpus_file(corpus_file_name))):

            # print(doc_id)
            # print(text)
            
            # print()
            # print(snowball_tokens)
            # print()
            # print("-------------------")
            # print()

            max_token_weight = {}
            num_tokens = {}

            total_tokens = 0

            for sentence in raw_vectors:

                # print("tokens:\t", sentence['tokens'])

                reconstructed = reconstruct_bpe(enumerate(sentence['tokens']))

                # print("reconstructed:\t", reconstructed)

                filtered_reconstructed = filter_pair_tokens(reconstructed)

                # print("filtered:\t", filtered_reconstructed)

                stemmed_reconstructed = stem_pair_tokens(filtered_reconstructed)

                # print("stemmed:\t", stemmed_reconstructed)

                weighed_reconstructed = aggregate_weights(stemmed_reconstructed, sentence['weights'])
                
                # print("weighed:\t", weighed_reconstructed)

                total_tokens += len(weighed_reconstructed)


                for reconstructed_token, score in weighed_reconstructed:
                    max_token_weight[reconstructed_token] = max(max_token_weight.get(reconstructed_token, 0), score)
                    num_tokens[reconstructed_token] = num_tokens.get(reconstructed_token, 0) + 1

                # print()

            
            # tokens = stem_list_tokens(filter_list_tokens(snowball_tokenize(text)))
            # total_tokens = len(tokens)
            # num_tokens = Counter(tokens)

            sparse_vector = {}

            token_score = rescore_vector(max_token_weight)

            for token, token_count in num_tokens.items():
                score = token_score[token]
                tf = score + token_count - 1
                # tf = token_count
                sparse_vector[token] = calc_tf(tf, total_tokens)

            out_file.write(json.dumps(sparse_vector) + "\n")

            total_tokens_overall += total_tokens
            num_docs += 1
    
    print("Average tokens per document:", total_tokens_overall / num_docs)

if __name__ == '__main__':
    main()
