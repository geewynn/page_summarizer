#!/usr/bin/env python
# coding: utf-8


import os
import re
from pyment import PyComment
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

FILENAME = 'pagesummarizer.py'

C = PyComment(FILENAME)
C.proceed()
C.diff_to_FILE(os.path.basename(FILENAME) + ".patch")
for s in C.get_output_docs():
    print(s)


# reading FILE

FILE = open("mayowa.txt", "r")
DATA = FILE.readlines()
FILE.close()


def preprocessor(text):
    """Lowers the case of the input, removes everything
    inside [] then removes 's and fetches only ascii characters

    :param TEXT: blah
    :type TEXT: string
    :returns: tokens-> blah
    :rtype: string

    """
    new_string = text.lower()
    new_string = re.sub(r'\([^)]*\)', '', new_string)
    new_string = re.sub('"', '', new_string)
    new_string = re.sub(r"'s\b", "", new_string)
    new_string = re.sub("[^a-zA-Z]", " ", new_string)
    new_string = re.sub('[m]{2,}', 'mm', new_string)
    tokens = new_string.split()
    tokens = (" ".join(tokens)).strip()
    return tokens


# call above function
TEXT = []
for i in DATA:
    TEXT.append(preprocessor(i))

ALL_SENTENCES = []
for i in TEXT:
    sentences = i.split(".")
    for i in sentences:
        if (i != ''):
            ALL_SENTENCES.append(i.strip())


# tokenizing the sentences for training word2vec
TOKENIZED_TEXT = []
for i in ALL_SENTENCES:
    TOKENIZED_TEXT.append(i.split())


# define word2vec model
MODEL_W2V = gensim.models.Word2Vec(
    TOKENIZED_TEXT,
    size=200,  # desired no. of features/independent variables
    window=5,  # conTEXT window size
    min_count=2,
    sg=0,  # 1 for cbow model
    hs=0,
    negative=10,  # for negative sampling
    workers=2,  # no.of cores
    seed=34)


# train word2vec
MODEL_W2V.train(TOKENIZED_TEXT, total_examples=len(TOKENIZED_TEXT), epochs=MODEL_W2V.epochs)


# define function to obtain sentence embedding
def word_vector(tokens, size):
    """

    :param tokens: blah
    :type tokens: string
    :param size: blah
    :type size: integer
    :returns: vec->blah
    :rtype: vector

    """
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += MODEL_W2V[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary

            continue
    if count != 0:
        vec /= count
    return vec


# call above function
WORDVEC_ARRAYS = np.zeros((len(TOKENIZED_TEXT), 200))
for i in range(len(TOKENIZED_TEXT)):
    WORDVEC_ARRAYS[i, :] = word_vector(TOKENIZED_TEXT[i], 200)


# similarity matrix
SIM_MAT = np.zeros([len(WORDVEC_ARRAYS), len(WORDVEC_ARRAYS)])


# compute similarity score
for i in range(len(WORDVEC_ARRAYS)):
    for j in range(len(WORDVEC_ARRAYS)):
        if i != j:
            SIM_MAT[i][j] = cosine_similarity(
                WORDVEC_ARRAYS[i].reshape(1, 200),
                WORDVEC_ARRAYS[j].reshape(1, 200))[0, 0]


# Generate a graph
NX_GRAPH = nx.from_numpy_array(SIM_MAT)


# compute pagerank SCORES
SCORES = nx.pagerank(NX_GRAPH)


# sort the SCORES
SORTED_X = sorted(SCORES.items(), key=lambda kv: kv[1], reverse=True)

SENT_LIST = []
for i in SORTED_X:
    SENT_LIST.append(i[0])


# extract top 10 sentences
NUM = 10
SUMMARY = ''
for i in range(NUM):
    SUMMARY = SUMMARY + ALL_SENTENCES[SENT_LIST[i]] + '. '
print(SUMMARY)