import json
from typing import Iterable
from qdrant_client import QdrantClient, models
import tqdm
import os
import math


DATASET = os.getenv("DATASET", "quora")

def calc_idf(n, df):
    # Fancy way to compute IDF
    return math.log((n - df + 0.5) / (df + 0.5) + 1.)


def read_frequencies() -> dict:
    with open(f'data/{DATASET}/idf.json', 'r') as file:
        return json.load(file)


def rescore_vector(vector: dict, idf: dict, n: int) -> dict:
    new_vector = {}

    sorted_vector = sorted(vector.items(), key=lambda x: x[1], reverse=True)

    for num, (idx, _value) in enumerate(sorted_vector):
        new_vector[idx] = calc_idf(n, idf.get(idx, 0)) * math.log(1./(num + 2) + 1.) # * value
    return new_vector

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

    for (idx, value) in sparse_vector.items():
        indices.append(int(idx))
        values.append(value)

    return models.SparseVector(
        indices=indices,
        values=values
    )


def read_data() -> Iterable[models.PointStruct]:
    idf = read_frequencies()

    number_of_document = 0
    for _ in read_corpus_jsonl(f'data/{DATASET}/corpus.jsonl'):
        number_of_document += 1

    n = 0
    for (sparse_vector, meta) in zip(read_vectors(f'data/{DATASET}/collection_vectors.jsonl'), read_corpus_jsonl(f'data/{DATASET}/corpus.jsonl')):
        yield models.PointStruct(
            id=n,
            vector={
                "attention": conver_sparse_vector(rescore_vector(sparse_vector, idf, number_of_document))
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

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "attention": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        }
    )

    client.upload_points(
        collection_name=collection_name,
        points=tqdm.tqdm(read_data())
    )
    

if __name__ == '__main__':
    main()
