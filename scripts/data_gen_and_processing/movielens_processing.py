#!/usr/bin/env python3
"""
Process MovieLens data to build documents with >=10 unique tags
and save them in a structured CSV.
"""

import os
import pandas as pd
import numpy as np

# === CONFIG ===
movie_lens_path = "C:/Users/anton/source/data/ml_25m/"
output_path = "data/real_docs/movielens/movies_1.csv"

# === LOAD DATA ===
movies_path = os.path.join(movie_lens_path, "movies.csv")
tags_path = os.path.join(movie_lens_path, "tags.csv")

print("Loading data...")
movies_df = pd.read_csv(movies_path)
tags_df = pd.read_csv(tags_path)

# === FILTER MOVIES WITH >=10 UNIQUE TAGS ===
unique_tag_counts = tags_df.groupby("movieId")["tag"].nunique()
movies_with_tags = unique_tag_counts[unique_tag_counts >= 10].index
print(f"Movies with >=10 unique tags: {len(movies_with_tags)}")

filtered_movies = movies_df[movies_df["movieId"].isin(movies_with_tags)]
filtered_tags = tags_df[tags_df["movieId"].isin(movies_with_tags)]

# === AGGREGATE TAGS ===
movie_tags = (
    filtered_tags.groupby("movieId")["tag"]
    .apply(lambda x: sorted(set(str(t).strip() for t in x.dropna())))  # clean, dedup, sort
)

# Merge movie info with tags
movies_merged = filtered_movies.merge(
    movie_tags, left_on="movieId", right_index=True
)

# === BUILD DOCUMENT TEXT ===
def build_doc_text(row):
    tags_str = ", ".join(row["tag"])
    return f"Movie Title: {row['title']} \n Genres: {row['genres']} \n Tags: {tags_str}"

movies_merged["d_text"] = movies_merged.apply(build_doc_text, axis=1)

# === ANALYTICS ===
movies_merged["doc_word_len"] = movies_merged["d_text"].str.split().str.len()
movies_merged["tag_count"] = movies_merged["tag"].str.len()

print("\nSummary statistics:")
print("Tag counts per movie:")
print(movies_merged["tag_count"].describe())
print("\nDocument word lengths:")
print(movies_merged["doc_word_len"].describe())

# === SORT BY POPULARITY (tag_count) ===
movies_merged = movies_merged.sort_values(
    ["tag_count", "d_text"], ascending=[False, True]  # break ties alphabetically
).reset_index(drop=True)

# Assign d_id sequentially
movies_merged["d_id"] = np.arange(1, len(movies_merged) + 1)

# Final dataframe
final_df = movies_merged[["d_id", "d_text", "movieId"]].rename(
    columns={"movieId": "movie_id"}
)

# === SAVE ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)

print(f"\nSaved processed dataset to: {output_path}")
