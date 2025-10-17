from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader


import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = str(pathlib.Path(__file__).parents[3] / "data" / "ir" / "beir")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load()

#### Inspect the structure and first few elements of each
print("Corpus structure:")
print("Type:", type(corpus))
print("Number of docs:", len(corpus))
print("First 3 docs:")
for i, (doc_id, doc) in enumerate(corpus.items()):
    if i >= 3: break
    print(f"Doc ID: {doc_id}")
    print(f"Title: '{doc['title']}'")
    print(f"Text: '{doc['text'][:200]}...'")
    print("---")

print("\nQueries structure:")
print("Type:", type(queries))
print("Number of queries:", len(queries))
print("First 3 queries:")
for i, (q_id, q) in enumerate(queries.items()):
    if i >= 3: break
    print(f"Query ID: {q_id}")
    print(f"Query: '{q}'")

print("\nQrels structure:")
print("Type:", type(qrels))
print("Number of qrels:", len(qrels))
print("First 3 qrels:")
for i, (q_id, rels) in enumerate(qrels.items()):
    if i >= 3: break
    print(f"Query ID: {q_id}")
    print(f"Relevances: {rels}")
