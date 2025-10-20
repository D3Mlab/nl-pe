import sys
import os
import json
import csv

def process_qrels_tsv_to_txt(tsv_path):
    """Convert TSV qrels to TXT qrels in standard format."""
    if not os.path.isfile(tsv_path):
        print(f"Warning: {tsv_path} not found, skipping.")
        return
    dir_name = os.path.dirname(tsv_path)
    base_name = os.path.basename(tsv_path)
    name_without_ext = os.path.splitext(base_name)[0]
    txt_path = os.path.join(dir_name, name_without_ext + '.txt')

    try:
        with open(tsv_path, 'r', newline='', encoding='utf-8') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            next(reader, None)  # Skip header
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                for row in reader:
                    if len(row) == 3:
                        qid, did, rel = row
                        txt_file.write(f"{qid} 0 {did} {rel}\n")
                    else:
                        print(f"Warning: Skipping invalid row in {tsv_path}: {row}")
        print(f"Converted qrels {tsv_path} to {txt_path}")
    except Exception as e:
        print(f"Error processing qrels {tsv_path}: {e}")

def process_corpus_to_docs(corpus_path, docs_path):
    """Convert corpus.jsonl to docs.csv with d_id and d_text."""
    with (open(corpus_path, 'r', encoding='utf-8') as f_in,
          open(docs_path, 'w', newline='', encoding='utf-8') as f_out):
        writer = csv.writer(f_out)
        writer.writerow(['d_id', 'd_text'])
        for line in f_in:
            line = line.strip()
            if line:
                data = json.loads(line)
                d_id = data['_id']
                title = data.get('title', '')
                text = data.get('text', '')
                d_text = f"Title: {title} Text:{text}"
                writer.writerow([d_id, d_text])
    print(f"Converted {corpus_path} to {docs_path}")

def process_queries_to_csv(queries_path, train_ids=None, test_ids=None, train_queries_path=None, test_queries_path=None, queries_csv_path=None):
    """Convert queries.jsonl to queries.csv, or split into train_queries.csv and test_queries.csv if train_ids and test_ids provided."""
    queries_data = []
    with open(queries_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if line:
                data = json.loads(line)
                q_id = data['_id']
                q_text = data['text']
                queries_data.append((q_id, q_text))
    
    if train_ids is not None and test_ids is not None and train_queries_path and test_queries_path:
        with (open(train_queries_path, 'w', newline='', encoding='utf-8') as f_train,
              open(test_queries_path, 'w', newline='', encoding='utf-8') as f_test):
            train_writer = csv.writer(f_train)
            test_writer = csv.writer(f_test)
            train_writer.writerow(['q_id', 'q_text'])
            test_writer.writerow(['q_id', 'q_text'])
            for q_id, q_text in queries_data:
                if q_id in train_ids:
                    train_writer.writerow([q_id, q_text])
                elif q_id in test_ids:
                    test_writer.writerow([q_id, q_text])
        print(f"Converted {queries_path} to {train_queries_path} and {test_queries_path}")
    else:
        with open(queries_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['q_id', 'q_text'])
            for q_id, q_text in queries_data:
                writer.writerow([q_id, q_text])
        print(f"Converted {queries_path} to {queries_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python format_conversion.py <path_to_beir_dataset_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    corpus_path = os.path.join(dataset_dir, 'corpus.jsonl')
    queries_path = os.path.join(dataset_dir, 'queries.jsonl')
    docs_path = os.path.join(dataset_dir, 'docs.csv')
    queries_csv_path = os.path.join(dataset_dir, 'queries.csv')

    if not os.path.exists(corpus_path):
        print(f"Error: {corpus_path} not found.")
        sys.exit(1)
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found.")
        sys.exit(1)

    process_corpus_to_docs(corpus_path, docs_path)

    # Check for train and test qrels to split queries
    qrels_dir = os.path.join(dataset_dir, 'qrels')
    train_qrels = os.path.join(qrels_dir, 'train.tsv')
    test_qrels = os.path.join(qrels_dir, 'test.tsv')
    train_ids = None
    test_ids = None

    if os.path.exists(train_qrels) and os.path.exists(test_qrels):
        train_ids = set()
        with open(train_qrels, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row:
                    train_ids.add(row[0])
        test_ids = set()
        with open(test_qrels, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row:
                    test_ids.add(row[0])
        train_queries_path = os.path.join(dataset_dir, 'train_queries.csv')
        test_queries_path = os.path.join(dataset_dir, 'test_queries.csv')
        process_queries_to_csv(queries_path, train_ids=train_ids, test_ids=test_ids, train_queries_path=train_queries_path, test_queries_path=test_queries_path)
    else:
        process_queries_to_csv(queries_path, queries_csv_path=queries_csv_path)

    # Convert qrels TSV to TXT if they exist
    if os.path.exists(train_qrels):
        process_qrels_tsv_to_txt(train_qrels)
    if os.path.exists(test_qrels):
        process_qrels_tsv_to_txt(test_qrels)

    print("Conversion complete.")
