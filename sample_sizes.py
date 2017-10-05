import csv

import numpy as np 
import pandas as pd

import spacy 

import index_numbers 

def build_training_data(csv_path="ctgov_sample_sizes.csv"):
    # read in data, drop citations w/no abstract
    df = pd.read_csv(csv_path).dropna(subset=["ab"])

    # for segmentation
    nlp = spacy.load('en')

    # normalize numbers
    number_tagger = index_numbers.NumberTagger()

    ###
    # 1. tokenize abstract
    # 2. find sample size(s)
    # 3. create label vector
    ###
    df['ab_numbers']   = df['ab'].apply(lambda x : number_tagger.swap(x))
    df['tokenized_ab'] = df['ab_numbers'].apply(lambda x : list(nlp(x)))

    label_vectors = []
    for row_tuple in df.iterrows():
        row = row_tuple[1]
        nums_to_labels = {row["enrolled_totals"]:"N", row["enrolled_P1"]:"n1", row["enrolled_P2"]:"n2"}
        y = annotate(row["tokenized_ab"], nums_to_labels)
        label_vectors.append(y)
    df["y"] = label_vectors

def annotate(tokenized_abstract, nums_to_labels):
    # nums_to_labels : dictionary mapping numbers to labels
    y = []
    for t in tokenized_abstract:
        try: 
            t_num = int(t.text)
            if t_num in nums_to_labels.keys():
                y.append(nums_to_labels[t_num])
            else:
                y.append("o")
        except:
            y.append("o")
    return y 



