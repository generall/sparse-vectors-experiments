import json
from typing import Iterable
from qdrant_client import QdrantClient, models
import tqdm
import os



def read_vectors(file_path) -> Iterable[dict]:

    with open(file_path, 'r') as file:
        for line in file:
            row = json.loads(line)
            yield row


def read_collection_tsv(file_path) -> Iterable[dict]:
    with open(file_path, 'r') as file:
        for line in file:
            idx, text = line.strip().split('\t')
            row = {'idx': int(idx), 'text': text}
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
    for (sparse_vector, meta) in zip(read_vectors('data/collection_vectors.jsonl'), read_collection_tsv('data/collection.tsv')):
        yield models.PointStruct(
            id=meta['idx'],
            vector={
                "attention": conver_sparse_vector(sparse_vector)
            },
            payload={
                "text": meta['text']
            }
        )

def main():

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY", None)

    client = QdrantClient(url, api_key=api_key, prefer_grpc=True)

    # Create collection
    collection_name = "msmarco"

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