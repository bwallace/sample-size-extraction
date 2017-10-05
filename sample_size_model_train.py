from itertools import chain

import numpy as np

import gensim 
from gensim.models import Word2Vec

import tensorflow as tf 
from tensorflow.contrib import learn

import pandas as pd 

import spacy 

import pycrfsuite

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer

def train_CRF(val_size=.2, model_out_name="sample-size.crfsuite"):
    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size)

    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.train(model_out_name)
    return trainer, (X_train, X_test, y_train, y_test)

def load_data(csv_path="ctgov_with_tags.csv"):
    return pd.read_csv(csv_path)

def preprocess(df):

    nlp = spacy.load('en') # for POS tagging

    X, y = [], []
    for instance in df.iterrows():
        instance = instance[1]
        abstract_token_vectors = []
        tokens, tags = POS_tag_and_tokenize(instance["ab_numbers"], nlp)

        nums_to_labels = {instance["enrolled_totals"]:"N", instance["enrolled_P1"]:"n1", instance["enrolled_P2"]:"n2"}
        cur_y = annotate(tokens, nums_to_labels)

        #if not "N" in set(cur_y):
        # import pdb; pdb.set_trace()

        y.append(cur_y)
        # import pdb; pdb.set_trace()

        abstract = (tokens, tags)
        for word_idx in range(len(tokens)):
            features = word2features(abstract, word_idx)
            abstract_token_vectors.append(features)
           
        X.append(abstract_token_vectors)
        
        
    #return X, df["y"].values
    
    return X, y

def POS_tag_and_tokenize(abstract, nlp=None):
    if nlp is None:
        nlp = spacy.load('en')

    tokens, POS_tags = [], []
    ab = nlp(abstract)
    for word in ab:
        tokens.append(word.text)
        POS_tags.append(word.pos_)
    return (tokens, POS_tags)

def annotate(tokenized_abstract, nums_to_labels):
    # nums_to_labels : dictionary mapping numbers to labels
    y = []
    for t in tokenized_abstract:
        try: 
            t_num = int(t)
            if t_num in nums_to_labels.keys():
                y.append(nums_to_labels[t_num])
            else:
                y.append("O")
        except:
            y.append("O")
    return y 


''' for CRF model '''
# based on: https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
def word2features2(abstract, i):
    words, POS_tags = abstract 
    word =   words[i]
    postag = POS_tags[i]
    features = [
        'word.lower=' + word.lower(),
        'word.isdigit=%s' % word.isdigit(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
    ]
    if i > 0:
        word1 = words[i-1]
        postag1 = POS_tags[i-1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
        ])
    else:
        features.append('BOS')
        
    if i < len(words)-1:
        word1 = words[i+1]
        postag1 = POS_tags[i+1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
        ])
    else:
        features.append('EOS')
                
    return features


''' for CRF model '''
# based on: https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
def word2features(abstract, i):
    words, POS_tags = abstract 
    word =   words[i]
    postag = POS_tags[i]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word.isdigit=%s' % word.isdigit(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
    ]
    if i > 0:
        word1 = words[i-1]
        postag1 = POS_tags[i-1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
        ])
    else:
        features.append('BOS')
        
    if i < len(words)-1:
        word1 = words[i+1]
        postag1 = POS_tags[i+1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
        ])
    else:
        features.append('EOS')
                
    return features


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "o" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'o'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )



