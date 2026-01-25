#How to process and embed beir datasets

1. Download using `python scripts\data_gen_and_processing\beir\beir_downloads.py <dataset_name>`

Gives:

webis-touche2020/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    ├── train.tsv
    └── test.tsv

2. Convert to experiment format `python scripts\data_gen_and_processing\beir\format_conversion.py <path_to_dataset_dir>` :

corpus.jsonl → docs.csv (d_id, d_text)
queries.jsonl → queries.csv (q_id,q_text), or train_queries.csv and test_queries.csv
qrels/*.tsv → qrels/*.txt

