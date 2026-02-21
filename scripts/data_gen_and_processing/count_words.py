#!/usr/bin/env python3
import sys
import csv
import statistics
import ctypes

csv.field_size_limit(ctypes.c_ulong(-1).value // 2)

def simple_token_estimate(text: str) -> int:
    """
    Rough OpenAI token estimate.
    Rule of thumb: ~1 token ≈ 0.75 words
    So tokens ≈ words / 0.75 ≈ words * 1.33
    """
    words = len(text.split())
    return int(words * 1.33)


def try_tiktoken():
    """
    Try to load tiktoken encoder.
    Returns encoder if available, else None.
    """
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python count_words.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    word_counts = []
    token_counts = []

    encoder = try_tiktoken()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue

            passage = row[1].strip()
            if not passage:
                continue

            words = len(passage.split())
            word_counts.append(words)

            if encoder:
                tokens = len(encoder.encode(passage))
            else:
                tokens = simple_token_estimate(passage)

            token_counts.append(tokens)

    if not word_counts:
        print("No valid passages found.")
        sys.exit(1)

    median_words = statistics.median(word_counts)
    median_tokens = statistics.median(token_counts)

    print(f"Median word count: {median_words}")
    if encoder:
        print(f"Median OpenAI token count (cl100k_base): {median_tokens}")
    else:
        print(f"Estimated median OpenAI token count (approximation): {median_tokens}")


if __name__ == "__main__":
    main()