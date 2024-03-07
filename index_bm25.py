import os
from tqdm import tqdm
import tantivy
from typing import Iterable, List
import shutil


def read_file(file_name: str) -> Iterable[str]:
    with open(file_name, "r") as file:
        for line in file:
            doc_id, doc_text = line.strip().split("\t")
            yield int(doc_id), doc_text


def main():

    file_name = "data/collection.tsv" # MS MARCO collection
    file_out = "data/bm25.tantivy" # output file
    

    if os.path.exists(file_out):
        # remove direcotry recursively
        shutil.rmtree(file_out)


    if not os.path.exists(file_out):
        os.makedirs(file_out)

    buffer = []

    # Declaring our schema.
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("body", stored=True)
    schema_builder.add_integer_field("doc_id", stored=True, indexed=True)
    schema = schema_builder.build()

    # Creating our index (in memory)
    index = tantivy.Index(schema, path=file_out)

    writer = index.writer()

    for doc_id, doc_text in tqdm(read_file(file_name)):
        doc = tantivy.Document(
            doc_id=doc_id,
            body=doc_text
        )
        writer.add_document(doc)
        if doc_id % 1000 == 0:
            writer.commit()

    writer.commit()

if __name__ == '__main__':
    main()