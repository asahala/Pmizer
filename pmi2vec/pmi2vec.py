#!/usr/bin/python
# -*- coding: utf-8 -*-

""" =================================================
PMI2VEC -- asahala 2019-03-14

Builds word vectors from Morfessor-preprocessed data
by using pointwise mutual information and single value
decomposition.

================================================= """

import sys
import itertools
import math
import json
import gzip
import numpy as np
from numpy.linalg import norm
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

BUFFER = '<BF>'
LOGBASE = 2

def _log(n):
    if LOGBASE is None:
        return math.log(n)
    else:
        return math.log(n, LOGBASE)

def trim_float(number):
    if isinstance(number, int):
        return number
    else:
        return float('{0:.3f}'.format(number))

class PPMI:
    """ Positive Pointwise Mutual Information.
    Score orientation is -log p(a,b) > 0, 0 """
    minimum = 0
    
    @staticmethod
    def score(ab, a, b, cz):
        return max(_log(ab*cz) - _log(a*b), 0)


class PMI2VEC:

    def __init__(self):
        self.text = []
        self.window_size = 0
        self.keywords = []
        self.scored = {}
        self.words1 = []
        self.words2 = []
        self.index_table = {}
        self.word_vectors = {}
        
    def _read_file(self, filename):
        """ Read corpus and buffer each line to prevent skipgrams
        being calulated over given constraints (e.g. paragraphs);

        Input should be already tokenized! """

        print("> reading %s..." % filename)
        with open(filename, 'r', encoding='utf-8',
                  errors='ignore') as data:
            self.text = [BUFFER] * self.window_size
            buffers = 1
            morphs = []
            for line in data.readlines():
                words = line.strip('\n').strip().split(' ')
                self.text.extend(words + [BUFFER] * self.window_size)
                for word in words:
                    morphs.extend(word.split('+'))
                buffers += 1
                
        self.morph_count = len(morphs)
        self.morph_freqs = Counter(morphs)
        self.corpus_size = len(self.text) - (buffers * self.window_size)
        self.word_freqs = Counter(self.text)

    def _is_wordofinterest(self, word, index):
        return word in self.words[index]

    def load_vectors(self, filename):
        """ Load lookup table from JSON """
        print('> reading %s ...' % filename)
        with gzip.GzipFile(filename, 'r') as data:
            #self.word_vectors = json.load(data)
            self.word_vectors = json.loads(data.read().decode('utf-8'))

    def save_vectors(self, filename):
        """ Save lookup table as JSON """
        print('> writing %s ...' % filename)
        with gzip.GzipFile(filename, 'w') as data:
            #json.dump(self.word_vectors, data)
            data.write(json.dumps(self.word_vectors).encode('utf-8'))

    def scale(self, bigram_freq):
        """ Scale frequencies according to window size as in
        Church & Hanks 1990 """
        if not self.symmetry:
            return bigram_freq / (self.window_size - 1)
        else:
            return bigram_freq / ((self.window_size - 1) * 2)
        
    def score_bigrams(self, filename, measure, windowsize=2,
                      symmetry=False, words='all'):
        """ This function finds and scores bigrams (or skipgrams) from
        the given corpus file by using given measure """
        self.symmetry = symmetry
        self.window_size = windowsize
        self._read_file(filename)
        self.minimum = measure.minimum
        print("> counting bigrams...")

        def get_bigrams_symmetric():
            """ Symmetric window """
            wz = self.window_size - 1
            for w in zip(*[self.text[i:] for i in range(1+wz*2)]):
                #if w[wz] in self.keywords:
                for bigram in itertools.product([w[wz]], w[0:wz]+w[wz+1:]):
                    yield bigram

        def get_bigrams_forward():
            """ Asymmetric window """
            for w in zip(*[self.text[i:] for i in range(self.window_size)]):
                #if w[0] in self.keywords:
                for bigram in itertools.product([w[0]], w[1:]):
                    yield bigram

        def get_subunits(bigrams):
            """ Count morphemic bigrams """
            for w1, w2 in bigrams:
                for m1 in w1.split('+'):
                    for m2 in w2.split('+'):
                        yield (m1, m2)
            
        if symmetry:
            bigrams = get_bigrams_symmetric()
            morphs = Counter(get_subunits(bigrams))
        else:
            bigrams = get_bigrams_forward()
            morphs = Counter(get_subunits(bigrams))

        print("> scoring bigrams...")
        for bigram in morphs.keys():
            w1, w2 = bigram
            """ Ignore buffer symbols """
            if BUFFER not in bigram:
                bf = morphs[bigram]
                w1_freq = self.morph_freqs[w1]
                w2_freq = self.morph_freqs[w2]
                score = measure.score(self.scale(bf), w1_freq, w2_freq,
                                      self.morph_count)
                data = {'score': score,
                        'bigram_freq': bf,
                        'w1_freq': w1_freq,
                        'w2_freq': w2_freq}
                self.words1.append(w1)
                self.words2.append(w2)
                """ Force stuff into dictionary to make it faster """
                try:
                    self.scored[w1][w2] = data
                except KeyError:
                    self.scored[w1] = {}
                    self.scored[w1][w2] = data
                finally:
                    pass
                
    def build_vectors(self, value, dimension=300, algorithm='arpack'):
        """ Initialize minimum scores for the matrix """
        if value == 'score':
            min_ = self.minimum
        else:
            min_ = 0
            
        matrix = []
        i = 0
        for w1 in set(self.words1):
            row = []
            for w2 in set(self.words2):
                score = min_
                if w1 in self.scored.keys():
                    if w2 in self.scored[w1].keys():
                        bigram = self.scored[w1][w2]
                        score = trim_float(bigram[value])
                row.append(score)
            if any(row):
                self.index_table[w1] = i
                matrix.append(row)
                i += 1
        
        matrix = np.matrix(matrix)
        print("> truncating dimensions", matrix.shape, '->', (matrix.shape[0], dimension))
        svd = TruncatedSVD(n_components=dimension, algorithm=algorithm)
        matrix = svd.fit_transform(matrix)

        """ Combine wordform vectors from morph vectors """
        for word in self.word_freqs.keys():
            vecs = []
            for morph in word.split('+'):
                index = self.index_table.get(morph, None)
                if index is not None:
                    vecs.append(matrix[index])
            if vecs:
                self.word_vectors[word] = list(np.mean(vecs, axis=0))

    def cosine_similarity(self, word1, word2):
        """ Compare cosine similarity between two words """
        vec1 = self.word_vectors.get(word1, None)
        vec2 = self.word_vectors.get(word2, None)
        if all((vec1, vec2)):
            print(np.dot(vec1, vec2 ) / (norm(vec1) * norm(vec2)))

def demo():
    """ How to build and sace vectors from new text """
    a = PMI2VEC()
    txt = 'Tiglath-pileserSargon_texts'
    a.score_bigrams(txt, measure=PPMI, windowsize=5, symmetry=True)
    a.build_vectors('score', dimension=300, algorithm='arpack')
    a.save_vectors('tiglath.vec.gzip')

def demo2():
    """ How to load vectors and test similarities """
    a = PMI2VEC()
    a.load_vectors('tiglath.vec.gzip')
    a.cosine_similarity('bēlu', 'šarru')       # lord and king: very high similarity
    a.cosine_similarity('Akkad_P', 'Šumeru_P') # Sumer and Akkad: high similarity
    a.cosine_similarity('kaspu', 'hurāṣu')     # silver and gold: high similarity
    a.cosine_similarity('abu', 'māru')         # father and son: medium-low similarity
    a.cosine_similarity('kakku', 'qabû')       # weapon and speak: very low similarity

#demo()
demo2()    
            
