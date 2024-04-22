from qdrant_client import QdrantClient, models
import os
import json
import math

from remap_tokens import filter_list_tokens, snowball_tokenize, stem_list_tokens
from upload_sparse import token_to_idx

DATASET = os.getenv("DATASET", "quora")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)


def conver_sparse_vector(sparse_vector: dict) -> models.SparseVector:
    indices = []
    values = []

    for (idx, value) in sparse_vector.items():
        indices.append(token_to_idx(idx))
        values.append(value)

    return models.SparseVector(
        indices=indices,
        values=values
    )


def load_queries():
    queries = {}

    with open(f"data/{DATASET}/queries.jsonl", "r") as file:
        for line in file:
            row = json.loads(line)
            queries[row["_id"]] = { **row, "doc_ids": [] }
    
    with open(f"data/{DATASET}/qrels/test.tsv", "r") as file:
        next(file)
        for line in file:
            query_id, doc_id, score = line.strip().split("\t")
            if int(score) > 0:
                queries[query_id]["doc_ids"].append(doc_id)
    
    queries_filtered = {}
    for query_id, query in queries.items():
        if len(query["doc_ids"]) > 0:
            queries_filtered[query_id] = query

    return queries_filtered


def main():
    n = 0
    hits = 0
    limit = 10
    number_of_queries = 200

    # model = SparseModel()

    queries = load_queries()

    client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)


    def search_sparse(query, limit):
        sparse_vector = dict(map(
            lambda x: (x, 1),
            stem_list_tokens(filter_list_tokens(snowball_tokenize(query)))
        ))

        sparse_vector = conver_sparse_vector(sparse_vector)
        result = client.search(
            collection_name=DATASET,
            query_vector=models.NamedSparseVector(
                name="attention",
                vector=sparse_vector
            ),
            with_payload=True,
            limit=limit
        )

        return result

    for idx, query in enumerate(queries.values()):
        if idx >= number_of_queries:
            break

        result = search_sparse(query["text"], limit)
        found_ids = []

        for hit in result:
            found_ids.append(hit.payload["id"])

        query_hits = 0
        for doc_id in query["doc_ids"]:
            n += 1
            if doc_id in found_ids:
                hits += 1
                query_hits += 1

        print(f"Processing query: {query}, hits: {query_hits}")

    print(f"Recall @ {limit}: {hits} out of {n} = {hits/n}")


if __name__ == "__main__":
    main()
