import tantivy
import re
import json
import os

DATASET = os.getenv("DATASET", "quora")


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


def sanitize_query_for_tantivy(query):
    # remove special characters: ()[]{}^"~*?:
    query = re.sub(r'[\(\)\[\]\{\}\^\"\~\*\?\:\-\`]', ' ', query)
    return query


def main():
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("body", stored=True)
    schema_builder.add_text_field("doc_id", stored=True)
    schema = schema_builder.build()
    index = tantivy.Index(schema, path=f"data/{DATASET}/bm25.tantivy/")

    searcher = index.searcher()

    def search_bm25(query, limit):
        query = index.parse_query(sanitize_query_for_tantivy(query), ['body'])
        hits = searcher.search(query, limit).hits
        docs = [
            searcher.doc(doc_address)
            for (score, doc_address) in hits
        ]
        return docs
    
    n = 0
    hits = 0
    limit = 10
    number_of_queries = 200

    queries = load_queries()


    for idx, query in enumerate(queries.values()):
        if idx >= number_of_queries:
            break

        result = search_bm25(query["text"], limit)
        found_ids = []

        for hit in result:
            found_ids.append(hit["doc_id"][0])

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
