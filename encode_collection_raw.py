from inference import SparseModel
from semantic_text_splitter import CharacterTextSplitter
from tqdm import tqdm
import json
from typing import Iterable
import os


DATASET = os.getenv("DATASET", "quora")

def read_file(file_name: str) -> Iterable[str]:
    with open(file_name, "r") as file:
        for line in file:
            row = json.loads(line)
            yield row["_id"], row["text"]


def read_file_batched(file_name: str, batch_size: int) -> Iterable[list]:
    buffer = []
    for doc_id, doc_text in tqdm(read_file(file_name)):
            buffer.append((doc_id, doc_text))
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
    if buffer:
        yield buffer


def main():
    splitter = CharacterTextSplitter()
    model = SparseModel()

    file_name = f"data/{DATASET}/corpus.jsonl" # MS MARCO collection
    file_out = f"data/{DATASET}/collection_vectors_raw.jsonl" # output file

    with open(file_out, "w") as out_file:
        for batch in read_file_batched(file_name, 1):
            splitted_docs = [splitter.chunks(doc_text, chunk_capacity=(128,256)) for _, doc_text in batch]
            for doc in splitted_docs:
                encoded_doc = model.encode_raw(doc)
                out_file.write(json.dumps(encoded_doc) + "\n")

if __name__ == '__main__':
    main()
