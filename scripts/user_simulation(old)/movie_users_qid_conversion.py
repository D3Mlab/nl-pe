import sys
import os
import pandas as pd

def process_users(input_dir, parent_dir):
    users_path = os.path.join(input_dir, 'users.csv')
    df = pd.read_csv(users_path)
    df.insert(0, 'u_id', range(len(df)))
    output_users = os.path.join(parent_dir, 'users.csv')
    df.to_csv(output_users, index=False)
    print(f"Processed users.csv saved to {output_users}")

def process_qrels(input_dir, parent_dir, docs_csv):
    qrels_path = os.path.join(input_dir, 'qrels.txt')
    output_qrels = os.path.join(parent_dir, 'qrels.txt')

    # Load d_ids
    docs_df = pd.read_csv(docs_csv)
    d_ids = docs_df['d_id'].tolist()

    with open(qrels_path, 'r') as f_in, open(output_qrels, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, old_did, rel = parts
            new_did = d_ids[int(old_did)]
            f_out.write(f"{qid}\t{new_did}\t{rel}\n")

    print(f"Processed qrels.txt saved to {output_qrels}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python movie_users_qid_conversion.py <input_dir> <docs_csv>")
        print("input_dir should contain users.csv and qrels.txt")
        print("docs_csv should be path to docs.csv")
        sys.exit(1)

    input_dir = sys.argv[1]
    docs_csv = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Input dir {input_dir} not found")
        sys.exit(1)

    if not os.path.isfile(docs_csv):
        print(f"Docs csv {docs_csv} not found")
        sys.exit(1)

    parent_dir = os.path.dirname(input_dir)

    process_users(input_dir, parent_dir)
    process_qrels(input_dir, parent_dir, docs_csv)

if __name__ == "__main__":
    main()
