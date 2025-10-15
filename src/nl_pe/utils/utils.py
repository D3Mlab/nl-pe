import json
import os
import math
from nl_pe.utils.setup_logging import setup_logging
import random
from copy import deepcopy
from math import prod
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

#linear regression
def lr(x,y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    model = LinearRegression()
    model.fit(x, y)

    lr_slope = model.coef_[0]
    lr_intercept = model.intercept_

    return lr_slope, lr_intercept

#other misc helper functions
def get_doc_text_list(ids, corpus_path):
    #ids: list of doc_ids
    #return [{"docID": d1, "text": <text_d1>},...]

    if not ids:
        return []

    id_set = set(ids)
    id_and_text_dict = {}

    # Iterate through the corpus and collect only the required documents
    for doc in jsonl_line_generator(corpus_path):
        doc_id = doc.get('docID')
        if doc_id in id_set:
            id_and_text_dict[doc_id] = doc.get('text')
            if len(id_and_text_dict) == len(ids):
                break

    id_and_text_list = [{'docID': doc_id, 'text': id_and_text_dict[doc_id]} for doc_id in ids if doc_id in id_and_text_dict]

    return id_and_text_list

def jsonl_line_generator(path):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)
