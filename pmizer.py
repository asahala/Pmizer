#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import time
import itertools
import math
import re

WINDOW_SCALING = False  # Use window scaling for bigram freqs
LOGBASE = None            # Logarithm base; set to None for ln

""" Association measures - Aleksi Sahala 2018 - University of Helsinki
/ Deep Learning and Semantic Domains in Akkadian Texts
/ Centre of Excellence in Ancient Near Eastern Empires

This script calculates different association scores for pairs of words.
Different constraints may be set by using Associations.set_properties().
These include:

  ´windowsize´      (int) collocational window that defines the
                    maximum mutual distance of the elements of a bigram.
                    Minimum distance is 2, which means that the words
                    are next to each another.

  ´freq_threshold´  (int) minimum allowed bigram frequency.

  ´words1´          (list) words of interest: Bigram(word1, word2)
  ´words2´          Words of interest may also be expressed as compiled
                    regular expressions. See exampes in ´stopwords´.

  ´stopwords´       (list) discard uninteresting words like
                    prepositions, numbers etc. May be expressed as
                    compiled regular expressions. For example
                    [re.compile('\d+?'), re.compile('^[A-ZŠṢṬĀĒĪŪ].+')]
                    will discard all numbers written as digits, as well
                    as all words that begin with a capital letter.

                    NOTE: It may be wise to express series of regular
                    expressions as disjunctions (regex1|...|regexn) as it
                    makes the comparison significantly faster, e.g. 
                    [re.compile('^(\d+?|[A-ZŠṢṬĀĒĪŪ].+)'].

Score bigrams by using Associations.score_bigrams(arg), where ´arg´ is
one of the following measures:

    PMI             Pointwise mutual information (Church & Hanks 1990)
    NPMI            Normalized PMI (Bouma 2009)
    PMI2            PMI^2 (Daille 1994)
    PMI3            PMI^3 (Daille 1994)
    PPMI            Positive PMI. As PMI but discards negative scores.
    PPMI2           Positive PMI^2 (Role & Nadif 2011)

Measure property comparison. LFB stands for low-frequency bias. Measures
having this property tend to give very high scores for low-frequency
words. Thus they are not recommended to be used with low frequency
thresholds.

              LFB    max    ind    min
    PMI       high   no     yes    yes                  
    NPMI      high   yes    yes    yes
    PPMI      high   no     yes    yes
    PMI2      low    yes    no     yes
    PMI3      neg    yes    no     yes
    PPMI2     low    yes    no     yes

"""

def _log(n):
    if LOGBASE is None:
        return math.log(n)
    else:
        return math.log(n, LOGBASE)
    
class PMI:
    """ Pointwise Mutual Information: -log p(a,b) > 0 > -inf """
    @staticmethod
    def score(ab, a, b, cz):
        return _log(ab*cz) - _log(a*b)

class NPMI:
    """ Normalized PMI: 1 > 0 > -1 """
    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) / -_log(ab/cz)

class PMI2:
    """ PMI^2 (fixes the low-frequency bias of PMI and NPMI:
    0 > log p(a,b) > -inf """
    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) - (-_log(ab/cz))

class PMI3:
    """ PMI^3 (fixes the low-frequency bias of PMI and NPMI:
    0 > log p(a,b) > -inf """
    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) - (-(2*_log(ab/cz)))

class PPMI:
    """ Positive PMI: -log p(a,b) > 0 = 0"""
    @staticmethod
    def score(ab, a, b, cz):
        return max(PMI.score(ab, a, b, cz), 0)

class PPMI2:
    """ Positive derivative of PMI^2. Shares exaclty the same
    properties but the score orientation is on the positive
    plane: 1 > 2^log p(a, b) > 0 """
    @staticmethod
    def score(ab, a, b, cz):
        return 2**PMI2.score(ab, a, b, cz)


class Associations:

    def __init__(self):
        self.text = []
        self.windowsize = 2
        self.freq_threshold = 1
        self.words = {1: [], 2: []}
        self.stopwords = ['', '_']
        self.regex_stopwords = []
        self.regex_words = {1: [], 2: []}
    
    def set_constraints(self, **kwargs):
        """ Set constraints. Separate regular expressions from the
        string variables, as string comparison is significantly faster
        than re.match() """
        for key, value in kwargs.items():
            if key == 'stopwords':
                for stopword in value:
                    if isinstance(stopword, str):
                        self.stopwords.append(stopword)
                    else:
                        self.regex_stopwords.append(stopword)
            if key in ['words1', 'words2']:
                index = int(key[-1])
                for word in value:
                    if isinstance(word, str):
                        self.words[index].append(word)
                    else:
                        self.regex_words[index].append(word)
            else:
                setattr(self, key, value)

        """ Combined tables for faster comparison """
        self.anywords = any([self.words[1], self.words[2],
                         self.regex_words[1], self.regex_words[2]])
        self.anywords1 = any([self.words[1], self.regex_words[1]])
        self.anywords2 = any([self.words[2], self.regex_words[2]])
        
    def read_file(self, filename):
        """ Open lemmatized input file with one text per line.
        Add buffer equal to window size after each text to prevent
        words from different texts being associated. """
        print('reading %s...' % filename)
        with open(filename, 'r', encoding="utf-8") as data:
            self.text = ' '.join([line.strip('\n') + ' _'*self.windowsize\
                             for line in data.readlines()]).split(' ')

            self.word_freqs = Counter(self.text)
            self.corpus_size = len(self.text)
            
    def score_bigrams(self, measure):

        def scale(bf):
            """ Scale bigram frequency with window size (if selected)
            as done in NLTK.  """
            if WINDOW_SCALING:
                return bf / (self.windowsize - 1)
            else:
                return bf

        def match_regex(words, regexes):
            """ Matches a list of regexes to list of words """
            return any([re.match(r, w) for r in regexes for w in words])

        def is_stopword(*words):
            """ Compare words with stop word list and regexes. Return
            True if not in the list """
            if not self.regex_stopwords:
                return any(w in self.stopwords for w in words)
            else:
                return match_regex(words, self.regex_stopwords)\
                       or any(w in self.stopwords for w in words)

        def is_wordofinterest(word, index):
            """ Compare words with the list of words of interest.
            Return True if in the list """
            if not self.regex_words[index]:
                return word in self.words[index]
            else:
                return match_regex([word], self.regex_words[index])\
                       or word in self.words[index]
        
        def is_valid(w1, w2, freq):
            """ Validate bigram. Discard all, which
                a) are rarer than the frequency threshold
                b) contain lacunae ´_´
                c) are ampty ´´
                d) do not belong into predefined words of interest
                e) belong in into stop words """
            if freq >= self.freq_threshold:    
                if not self.anywords:
                    return not is_stopword(w1, w2)
                elif self.anywords and self.anywords2:
                    return is_wordofinterest(w1, 1) and is_wordofinterest(w2, 2)
                else:
                    if self.anywords1:
                        return is_wordofinterest(w1, 1) and not is_stopword(w2)
                    if self.anywords2:
                        return is_wordofinterest(w2, 2) and not is_stopword(w1)
                    else:
                        return False
            else:
                return False

        def count_bigrams():
            """ Calculate bigrams within each window; buffer text
            to prevent going out of range. """
            print('counting bigrams...')
            for w in zip(*[self.text[i:] for i in range(self.windowsize)]):
                for bigram in itertools.product([w[0]], w[1:]):
                    yield bigram

        bigram_freqs = Counter(count_bigrams())
        scored = []

        """ Score bigrams by given measure """
        print('calculating scores...')
        for bigram in bigram_freqs.keys():
            w1, w2 = bigram[0], bigram[1]
            if is_valid(w1, w2, bigram_freqs[bigram]):                
                score = measure.score(scale(bigram_freqs[bigram]),
                            self.word_freqs[w1], self.word_freqs[w2],
                            self.corpus_size)
                scored.append((bigram, bigram_freqs[bigram],
                            self.word_freqs[w1], self.word_freqs[w2], score))
        scored = sorted(scored)
        for x in scored:
            print(x)

a = Associations()
a.set_constraints(windowsize = 10,
                  freq_threshold = 2,
                  words1=['kakku'])

a.read_file('neoA')
st = time.time()
a.score_bigrams(PPMI)
et = time.time() - st
print('time', et)


