#!/usr/bin/python
# -*- coding: utf-8 -*-

""" =================================================
PMI2VEC -- asahala 2019-03-14

Builds word vectors from Morfessor-preprocessed data
by using pointwise mutual information and singular
value decomposition.

================================================= """
import time
import sys
import re
import itertools
import math
import json
import gzip
import numpy as np
from numpy.linalg import norm
from collections import Counter
from sklearn.decomposition import TruncatedSVD

LACUNA = '_'
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


class PMI:
    """ Pointwise Mutual Information (Church & Hanks 1990). 
    Score orientation is -log p(a,b) > 0 > -inf """
    minimum = -math.inf

    @staticmethod
    def score(ab, a, b, cz):
        return _log(ab*cz) - _log(a*b)

class PPMI:
    """ Positive Pointwise Mutual Information.
    Score orientation is -log p(a,b) > 0, 0 """
    minimum = 0
    
    @staticmethod
    def score(ab, a, b, cz):
        return max(_log(ab*cz) - _log(a*b), 0)

class NPMI:
    """ Normalized PMI (Bouma 2009).
    Score orientation is  +1 > 0 > -1 """
    minimum = -1.0

    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) / -_log(ab/cz)

class PMI2:
    """ PMI^2 (Daille 1994). Fixes the low-frequency bias 
    of PMI and NPMI by squaring the numerator to compensate 
    multiplication done in the denominnator.
    Scores are oriented as: 0 > log p(a,b) > -inf """
    minimum = -math.inf
    
    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) - (-_log(ab/cz))


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
        self.translations = {}
        
    def _read_file(self, filename):
        """ Read corpus and buffer each line to prevent skipgrams
        being calulated over given constraints (e.g. paragraphs);

        Input should be already tokenized! """
        st = time.time()

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
        et = time.time() - st
        print('> readfile (s)', et)

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

    def save_vectors_txt(self, filename):
        """ Default Word2vec vector (text) format """
        vector_n = len(self.word_vectors.keys())
        with open(filename, 'w', encoding='utf-8') as data:
            # First write number of words and dimension
            data.write('%i %i\n' % (vector_n, self.dimension))
            for key, value in self.word_vectors.items():
                # Then for every vector print word and the vector
                # each float separated by space value[0] = word freq
                # value[-1] vector
                data.write(key + ' ')
                value[-1].tofile(data, sep=' ')
                data.write('\n')

    def load_dictionary(self, filename):
        with open(filename, 'r', encoding="utf-8") as data:
            for line in data.read().splitlines():
                if line:
                    word, translation = line.split('\t')
                    self.translations[word] = translation

    def scale(self, bigram_freq):
        """ Scale frequencies according to window size as in
        Church & Hanks 1990 """
        if not self.symmetry:
            return bigram_freq / (self.window_size - 1)
        else:
            return bigram_freq / (self.window_size - 1)
        
    def score_bigrams(self, filename, measure, windowsize=2,
                      symmetry=False, minfreq=2, words='all'):
        """ This function finds and scores bigrams (or skipgrams) from
        the given corpus file by using given measure, window size and 
        symmetry. ´words´ argument does not do anything at the moment.
        ´minfreq´ can be used to adjust the minimum frequency of b in
        PMI(a, b) (that is, the collocate frequency) """

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
                    if self.word_freqs[bigram[-1]] >= minfreq:
                        yield bigram

        def get_bigrams_forward():
            """ Asymmetric window """
            for w in zip(*[self.text[i:] for i in range(self.window_size)]):
                #if w[0] in self.keywords:
                for bigram in itertools.product([w[0]], w[1:]):
                    if self.word_freqs[bigram[-1]] >= minfreq:
                        yield bigram

        def get_subunits(bigrams):
            """ Count morphemic bigrams """
            for w1, w2 in bigrams:
                for subunit_bigram in itertools.product(w1.split('+'), w2.split('+')):
                    yield subunit_bigram
       
        st = time.time()
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
            if BUFFER not in bigram and LACUNA not in bigram:
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

        et = time.time() - st
        print('> scoring (s)', et)
                
    def build_vectors(self, value, dimension=300, algorithm='arpack', precombine=False):
        """ ´value´ can be either ´score´ (that is the
        association measure) or ´bigram_freq´ """
        st = time.time()
        self.dimension = dimension

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
                    row.append(np.float32(score))
            if any(row):
                self.index_table[w1] = i
                matrix.append(row)
                i += 1
        
        et = time.time() - st
        print('> building sparse matrix (s)', et)

        st = time.time()
        matrix = np.matrix(matrix)
        print("> truncating dimensions", matrix.shape, '->', (matrix.shape[0], dimension))
        svd = TruncatedSVD(n_components=dimension, algorithm=algorithm)
        matrix = svd.fit_transform(matrix)

        """ Combine wordform vectors from morph vectors if ´precombine´
        is set True, then every morph will not have their own vectors """
        if precombine:
            for word in self.word_freqs.keys():
                vecs = []
                for morph in word.split('+'):
                    index = self.index_table.get(morph, None)
                    if index is not None:
                        vecs.append(matrix[index])
                if vecs:
                    self.word_vectors[word.replace('+', '')] = (self.word_freqs[word],
                                                                np.array(np.mean(vecs, axis=0)))
        else:
            for morph in self.morph_freqs.keys():
                index = self.index_table.get(morph, None)
                if index is not None:
                    self.word_vectors[morph] = (self.morph_freqs[morph], np.array(matrix[index]))

        et = time.time() - st
        print('> svd (s)', et)

    def cosine_similarity(self, word1, word2, minfreq=0):
        """ Compare cosine similarity between two words """
        freq1, vec1 = self.word_vectors.get(word1, None)
        freq2, vec2 = self.word_vectors.get(word2, None)
        if all((vec1, vec2)) and freq2 >= minfreq:
            return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def get_best(self, word, minfreq=0, amount=15):
        table = []
        for word2 in self.word_vectors.keys():
            similarity = self.cosine_similarity(word, word2, minfreq)
            if similarity is not None:
                table.append((similarity, word2, self.translations.get(word2, None)))

        for entry in sorted(table, reverse=True)[0:amount]:
            print(entry)

def demo():
    """ How to build and save vectors from new text """
    a = PMI2VEC()
    #txt = 'testi.txt'
    #txt = 'MeBo-123.2015_stamd_sunder_vokaallangde.txt'
    #txt = 'MeBo-123.2015_stamd.txt'
    #txt = 'MeBo-123.2015_segmenteerd.txt'
    txt = 'noSynonyms_text_Mar19'
    a.score_bigrams(txt, measure=PPMI, windowsize=5, symmetry=True, minfreq=1)
    a.build_vectors('score', dimension=300, algorithm='arpack')
    #a.save_vectors('akkadi2_bfsqrt.vec.gzip')
    #a.save_vectors('test.vec.gzip')
    a.save_vectors_txt('akkadikoe.vec')

def query():
    """ How to load vectors and test similarities; use load_dictionary()
    only if you have translations available (they should be in .tsv like

    kakku     weapon
    inu       eye
    kalbu     dog

    If translation file is not available, just uncomment load_dictionary().
    """
    
    a = PMI2VEC()
    a.load_dictionary('words_noSynonymsMar19')
    a.load_vectors('akkadi.vec.gzip')
    #words = ['kakku_1', 'niāku_1', 'asakku_1', 'ikkibu_1', 'anzillu_1', 'wardatu_1', 'utukku_1','šarāqu_1', 'qaštu_1',]
    #words = ["ennettu_1","gillatu_1","gullultu_1","gullulu_1","pippilû_1",
    #        "šettu_1","šērtu_1","arnu_1","hiṭītu_1","hīṭu_1"]
    #['râmu_1','adāru_2','galātu_1','palāhu_1','parādu_1','šahātu_1','agāgu_1','ezēzu_1','kamālu_1','labābu_1','raʾābu_1','šabāsu_1','šamāru_1','zenû_1']
    words = ['sisû_1', 'qaštu_1']
    for emotion in words[-4:]:
        print('\n')
        print(emotion)
        a.get_best(emotion, minfreq=3, amount=25)

def test():
    a = PMI2VEC()
    a.load_dictionary('words_noSynonymsMar19')
    a.load_vectors('akkadi.vec.gzip')
    words = ['immeru_1', 'alpu_1', 'sisû_1']
    for word1 in words:
        for word2 in words:
            print(word1, word2, a.cosine_similarity(word1, word2))
        
demo()
#query()
#test()    
