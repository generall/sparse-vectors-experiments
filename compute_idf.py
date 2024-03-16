import os
import json
import tqdm
from typing import Iterable
from collections import defaultdict

DATASET = os.getenv("DATASET", "quora")

def read_vectors(file_path) -> Iterable[dict]:

    with open(file_path, 'r') as file:
        for line in file:
            row = json.loads(line)
            yield row

def main():
    counts = defaultdict(int)

    for row in tqdm.tqdm(read_vectors(f'data/{DATASET}/collection_vectors.jsonl')):
        for key in row.keys():
            counts[key] += 1

    # Create directory if it doesn't exist
    os.makedirs(f'data/{DATASET}', exist_ok=True)
    
    with open(f'data/{DATASET}/idf.json', 'w') as file:
        json.dump(counts, file, indent=2)

if __name__ == "__main__":
    main()
