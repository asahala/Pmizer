#!/usr/bin/python
# -*- coding: utf-8 -*-

""" asahala 2020 

    Use this for building PMI vectors.

    just call make()

"""

import pmizer2 as P
import numpy as np
import gzip
import json
from numpy.linalg import norm
from collections import Counter
from sklearn.decomposition import TruncatedSVD
translations = {}

def load_dict():
    with open('all_akk.dict', 'r', encoding='utf-8', errors='ignore') as data:
        for line in data.read().splitlines():
            key, value = line.split('\t')
            translations[key] = value

def load_vectors(filename):
    """ Load lookup table from JSON """
    print('> reading %s ...' % filename)
    with gzip.GzipFile(filename, 'r') as data:
        #self.word_vectors = json.load(data)
        return json.loads(data.read().decode('utf-8'))

def save_vectors(filename, vectors, dimension=0):
    """ Save lookup table as JSON """
    print('> writing %s ...' % filename)
    with gzip.GzipFile(filename, 'w') as data:
        #json.dump(self.word_vectors, data)
        data.write(json.dumps(vectors).encode('utf-8'))

def save_vectors_txt(filename, vectors, dimension):
    """ Default Word2vec vector (text) format """

    print(': Saving %s' % filename)

    vector_n = len(vectors.keys())

    with open(filename, 'w', encoding='utf-8') as data:
        # First write number of words and dimension
        data.write('%i %i\n' % (vector_n, dimension))
        for key, value in vectors.items():
            # Then for every vector print word and the vector
            # each float separated by space. value[0] = word freq
            # value[-1] vector
            data.write(key + ' ')
            value[-1].tofile(data, sep=' ')
            data.write('\n')

def build_vectors(scores, measure, dimension=300, algorithm='arpack'):
    
    """ Build sparse matrix from scores 
    :arg scores        pmizer score JSON
    :arg measure       pmizer measure object """

    indextable = {}
    matrix = []

    print(scores.keys())

    print(': Building sparse matrix')
    i = 0
    for w1 in set(scores['words1']):
        row = []
        for w2 in set(scores['words2']):
            score = 0 #measure.minimum
            c = scores['collocations']
            if w1 in c.keys():
                if w2 in c[w1].keys():
                    score = c[w1][w2]['score']
                row.append(np.float32(score))
        if any(row):
            indextable[w1] = i
            matrix.append(row)
            i += 1
    
    print(': Converting to numpy matrix')
    matrix = np.matrix(matrix)
            
    print(": Truncating: ", matrix.shape, '->', (matrix.shape[0], dimension))
    svd = TruncatedSVD(n_components=dimension, algorithm=algorithm)
    matrix = svd.fit_transform(matrix)

    """ Build word vectors: (word freq, vector) """
    vecs = {}
    for word, index in indextable.items():
        vecs[word] = (scores['freqs'][word], np.array(matrix[index]).tolist())

    save_vectors('best.pmizer', vecs, dimension)
    #save_vectors_txt('ppmicsw.vec', vecs, dimension)

def lookup(vectors, word, minfreq=0, amount=25):

    load_dict()

    def cosine_similarity(word1, word2, minfreq=0):
        """ Compare cosine similarity between two words """

        #if word2.isupper():
        #    if word2[0].isupper:
        #        return 0
        
        try:
            freq1, vec1 = vectors.get(word1, None)
            freq2, vec2 = vectors.get(word2, None)
        except:
            return 0

        if all((vec1, vec2)) and freq2 >= minfreq:
            return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def get_best(word, minfreq=0, amount=25):
        table = []
        for word2 in vectors.keys():
            similarity = cosine_similarity(word, word2, minfreq)
            if similarity is not None:
                table.append((str(round(similarity, 3)), 
                    word2, translations.get(word2, None)))

        for entry in sorted(table, reverse=True)[1:amount+1]:
            print('\t'.join(entry))

    get_best(word, minfreq, amount)

def make():
    text = P.Text('all.raw')
    text.read_dict()
    wz = 3
    words = ['*']

    x = P.Associations(text,
                   words1=words,
                   preweight=False,
                   formulaic_measure=P.Lazy,
                   minfreq_b = 1,
                   minfreq_ab = 1,
                   symmetry=True,
                   windowsize=wz,
                       factorpower=2)

    scores = x.score(P.PPMI)
    build_vectors(scores, P.PPMI)

#make()

"""
v = load_vectors('best.pmizer')
oath = ['māmītu', 'adû', 'tamītu', 'nīšu']
loyalty = ['kittu', 'kīnūtu', 'pazāru', 'pesēnu', 'harādu']
topography = ['erṣetu', 'sūqu', 'dūru', 'abullu', 'bītu', 'ēkurru',
              'parakku', 'mazzāzu', 'šubtu', 'ešertu', 'kiṣṣu', 'ibratu']
t = ['tillatu']
words = ["kīnūtu_I", "kittu_I", "pazāru_I", "pesēnu_I", "harādu_IV", "adû_I", "māmītu_I", "nīšu_II", "tamītu_I"]
#words = ["erṣetu_I", "sūqu_I", "dūru_I", "abullu_I", "bītu_I", "ēkurru_I", "parakku_I", "mazzāzu_I", "šubtu_I", "ešertu_I", "kiṣṣu_I"]
#words = ['šagāšu_I']
for w in words:
    print(w.upper() + ' ' + translations.get(w, ""))
    lookup(v, w, minfreq=3, amount=50)
    print('\n')

"""
