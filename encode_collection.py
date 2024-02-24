from inference import SparseModel
from semantic_text_splitter import CharacterTextSplitter
from tqdm import tqdm
import json
from typing import Iterable, List


def read_file(file_name: str) -> Iterable[str]:
    with open(file_name, "r") as file:
        for line in file:
            doc_id, doc_text = line.strip().split("\t")
            yield doc_id, doc_text


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

    file_name = "data/collection.tsv" # MS MARCO collection
    file_out = "data/collection_vectors.jsonl" # output file
    
    buffer = []

    with open(file_out, "w") as out_file:
        for batch in read_file_batched(file_name, 16):
            splitted_docs = [splitter.chunks(doc_text, chunk_capacity=(128,256)) for _, doc_text in batch]
            encoded_docs = model.encode_documents(splitted_docs)

            for doc_vector in encoded_docs:
                out_file.write(json.dumps(doc_vector) + "\n")


if __name__ == '__main__':
    main()