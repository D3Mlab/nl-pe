import sys
import os
import csv

def main():
    if len(sys.argv) != 2:
        print("Usage: python qrels_tsv_to_txt.py <path_to_tsv_file>")
        sys.exit(1)

    tsv_path = sys.argv[1]
    if not os.path.isfile(tsv_path):
        print(f"Error: File '{tsv_path}' does not exist.")
        sys.exit(1)

    dir_name = os.path.dirname(tsv_path)
    base_name = os.path.basename(tsv_path)
    name_without_ext = os.path.splitext(base_name)[0]
    txt_path = os.path.join(dir_name, name_without_ext + '.txt')

    try:
        with open(tsv_path, 'r', newline='', encoding='utf-8') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            # Skip header row
            next(reader, None)
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                for row in reader:
                    if len(row) == 3:
                        qid, did, rel = row
                        txt_file.write(f"{qid} 0 {did} {rel}\n")
                    else:
                        print(f"Warning: Skipping invalid row with {len(row)} columns: {row}")
        print(f"Converted {tsv_path} to {txt_path}")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
