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

3. Embed (full dimensionality):
python src\nl_pe\experiment_manager.py -e index_corpus -c <path_to_embedder_config>

4. Reduce dimensionality (matryoshka only)
embed using DimTruncator class, eg via config that has

```
embedding:
  class: DimTruncator  
  inference_batch_size: 100
  index_method: truncate_faiss_exact
  init_index_path: data/ir/beir/nfcorpus/gemini-3072/faiss/index
  matryoshka_dim: 48
```

5. Embedding queries
*index_doc_ids.pkl will be the suffix -- keep it, some legacy index_ids.pkl may need an update