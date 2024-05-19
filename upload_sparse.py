import json
from typing import Iterable, Union
from qdrant_client import QdrantClient, models
import tqdm
import os
import math
import mmh3
import requests


DATASET = os.getenv("DATASET", "quora")

MAX_VOCAB_SIZE = 2 ** 31


def token_to_idx(token: Union[str, int]) -> int:
    if isinstance(token, str):
        return mmh3.hash(token) % MAX_VOCAB_SIZE
    return token



def read_vectors(file_path) -> Iterable[dict]:

    with open(file_path, 'r') as file:
        for line in file:
            row = json.loads(line)
            yield row


def read_corpus_jsonl(file_path) -> Iterable[dict]:
    with open(file_path, 'r') as file:
        for line in file:
            row = json.loads(line)
            yield row


def conver_sparse_vector(sparse_vector: dict) -> models.SparseVector:
    indices = []
    values = []

    for (token, value) in sparse_vector.items():
        indices.append(int(token_to_idx(token)))
        values.append(value)

    return models.SparseVector(
        indices=indices,
        values=values
    )


def read_data() -> Iterable[models.PointStruct]:

    number_of_document = 0
    for _ in read_corpus_jsonl(f'data/{DATASET}/corpus.jsonl'):
        number_of_document += 1

    n = 0
    for (sparse_vector, meta) in zip(read_vectors(f'data/{DATASET}/collection_vectors.jsonl'), read_corpus_jsonl(f'data/{DATASET}/corpus.jsonl')):
        yield models.PointStruct(
            id=n,
            vector={
                "attention": conver_sparse_vector(sparse_vector)
            },
            payload={
                "text": meta['text'],
                "id": meta["_id"],
            }
        )
        n += 1

def main():

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY", None)

    client = QdrantClient(url, api_key=api_key, prefer_grpc=True)

    # Create collection
    collection_name = DATASET

    client.delete_collection(collection_name=collection_name)

    requests.put(
        f"{url}/collections/{collection_name}",
        headers={
            "api-key": api_key
        },
        json={
            "sparse_vectors": {
                "attention": {
                    "index": {
                        "on_disk": False
                    },
                    "modifier": "idf"
                }
            }
        }
    )

    client.upload_points(
        collection_name=collection_name,
        points=tqdm.tqdm(read_data())
    )
    

if __name__ == '__main__':
    main()
