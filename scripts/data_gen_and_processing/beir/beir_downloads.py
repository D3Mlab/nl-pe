import argparse
import itertools
import pprint
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

def main(force_download=False):
    expected_data_path = pathlib.Path(out_dir) / dataset
    if expected_data_path.exists() and not force_download:
        logging.info(f"Dataset already exists at {expected_data_path}, skipping download.")
        data_path = str(expected_data_path)
    else:
        data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load()

    #### Inspect the structure and first few elements of each
    pp = pprint.PrettyPrinter(indent=4, width=120)
    print("corpus =")
    pp.pprint(dict(itertools.islice(corpus.items(), 3)))
    print("\nqueries =")
    pp.pprint(dict(itertools.islice(queries.items(), 3)))
    print("\nqrels =")
    pp.pprint(dict(itertools.islice(qrels.items(), 3)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_download', action='store_true', help='Force download even if dataset already exists')
    args = parser.parse_args()
    main(args.force_download)
