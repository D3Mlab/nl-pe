import sys
import pandas as pd

def remove_first_column(input_csv_path: str):
    """
    Reads a CSV, removes the first column, and overwrites the file.
    """
    df = pd.read_csv(input_csv_path, header=0)
    df = df.iloc[:, 1:]  # remove first column
    df.to_csv(input_csv_path, index=False)
    print(f"Overwritten {input_csv_path} without the first column")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove-doc_id_col.py <csv_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    remove_first_column(input_csv)
