import sys
import pandas as pd

def add_doc_id_column(input_csv_path: str):
    """
    Reads a CSV, adds a 'doc_id' column with indices 0 to n-1 as the first column, and overwrites the file.
    """
    df = pd.read_csv(input_csv_path, header=0)
    df.insert(0, 'd_id', range(len(df)))
    df.to_csv(input_csv_path, index=False)
    print(f"Added 'doc_id' column to {input_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_doc_id_col.py <csv_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    add_doc_id_column(input_csv)
