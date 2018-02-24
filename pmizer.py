#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import time
import itertools
import math
import re
try:
    from dictionary import dct as akkadian_dict
except ImportError:
    print('dictionary.py not found!')
    akkadian_dict = {}



__version__ = "2018-02-25"

WINDOW_SCALING = False    # Apply window size penalty to scores
LOGBASE = 2               # Logarithm base; set to None for ln
LACUNA = '_'              # Symbol for lacunae in cuneiform languages
BUFFER = '<buffer>'       # Buffer symbol; added after each line

""" ====================================================================
Association measures - Aleksi Sahala 2018 - University of Helsinki =====
========================================================================

/ Deep Learning and Semantic Domains in Akkadian Texts
/ Center of Excellence in Ancient Near Eastern Empires

========================================================================
Associations.set_properties(*kwargs) ===================================
========================================================================

Constraints and properties may be set by using the following kwargs:

  ´windowsize´      (int) collocational window that defines the
                    maximum mutual distance of the elements of a bigram.
                    Minimum distance is 2, which means that the words
                    are next to each another.

  ´freq_threshold´  (int) minimum allowed bigram frequency.

  ´symmetry´        (bool) Use symmetric window. If not used, the window
                    is forward-looking. For example, with a window size
                    of 3 and w4 being our word of interest:

                            w1 w2 w3 w4 w5 w6 w7
                    symm.       +--+--^--+--+
                    asymm.            ^--+--+
  
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

  ´track_distance´  (bool) calculate average distance between the words
                    of the bigram. If the same bigram can be found
                    several times within the window, only the closest
                    distance is taken into account.

                    NOTE: Slows bigram counting 2 or 3 times depending
                    on the window size.

========================================================================
Associations.score_bigrams(filename, measure) ==========================
========================================================================

Input file should contain lemmas separated by spaces, e.g.

 monkey eat coconut and the sun be shine

Bigrams are not allowed to span from line to another. If you want
to disallow, for example, bigrams being recognized between paragraphs,
the text should contain one paragraph per line.

Argument ´measure´ must be one of the following:

    PMI             Pointwise mutual information (Church & Hanks 1990)
    NPMI            Normalized PMI (Bouma 2009)
    PMI2            PMI^2 (Daille 1994)
    PMI3            PMI^3 (Daille 1994)
    PPMI            Positive PMI. As PMI but discards negative scores.
    PPMI2           Positive PMI^2 (Role & Nadif 2011)

Each measure has its pros and cons. The table below indicates if the
measure has a low-frequency bias (LFB) (i.e. it tends to give high
scores for low-frequency bigrams). Measures with low or negative LFB
are generally more reliable if low frequency thresholds are used.

MAX, IND and MIN indicate if the measure has fixed upper bound,
independence threshold and lower bound. Measures with fixed bounds
are easier to compare with each other.

              LFB    MAX    IND    MIN      
    PMI       high   no     yes    yes
    NPMI      high   yes    yes    yes    
    PPMI      high   no     yes    yes      
    (P)PMI2   low    yes    no     yes      
    PMI3      neg    yes    no     yes    

==================================================================== """

def _log(n):
    if LOGBASE is None:
        return math.log(n)
    else:
        return math.log(n, LOGBASE)
    
class Raw_freq:
    """ Score bigrams by their frequency """
    @staticmethod
    def score(ab, a, b, cz):
        return ab

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
    """ PMI^3 (no low-freq bias, favors common bigrams) """
    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) - (-(2*_log(ab/cz)))

class PPMI:
    """ Positive PMI: -log p(a,b) > 0 = 0 """
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
        self.scored = []
        self.text = []
        self.windowsize = 2
        self.freq_threshold = 1
        self.symmetry = False
        self.words = {1: [], 2: []}
        self.stopwords = ['', LACUNA, BUFFER]
        self.regex_stopwords = []
        self.regex_words = {1: [], 2: []}
        self.distances = {}
        self.track_distance = True
    
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
        
        with open(filename, 'r', encoding="utf-8") as data:
            print('reading %s...' % filename)
            self.text = [BUFFER]*self.windowsize
            buffers = 1
            for line in data.readlines():
                self.text.extend(line.strip('\n').split(' ')
                                 + [BUFFER]*self.windowsize)
                buffers += 1
                
        self.word_freqs = Counter(self.text)
        self.corpus_size = len(self.text) - (buffers * self.windowsize)

    def _trim_float(self, number):
        return float('{0:.4f}'.format(number))

    def get_translation(self, word):
        """ Get translation from dictionary """
        try:
            translation = '[{}]'.format(akkadian_dict[word])
        except:
            translation = '[?]'
        return translation

    def get_distance(self, bigram):
        """ Calculate average distance for bigram's words """
        if self.track_distance:
            distance = self._trim_float(sum(self.distances[bigram])
                                    / len(self.distances[bigram]))
        else:
            distance = ''
        return distance

    def score_bigrams(self, filename, measure):

        self.read_file(filename)
        
        def scale(bf):
            """ Scale bigram frequency with window size. Makes the
            scores comparable with NLTK/Collocations PMI measure """
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
            """ Validate bigram. Discard stopwords and those which
            do not match with the word of interest lists """
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

        def count_bigrams_symmetric():
            """ Calculate bigrams within each symmetric window """
            print('counting bigrams...')
            wz = self.windowsize-1
            for w in zip(*[self.text[i:] for i in range(1+wz*2)]):
                for bigram in itertools.product([w[wz]], w[0:wz]+w[wz+1:]):
                    yield bigram

        def count_bigrams_symmetric_dist():
            """ Calculate bigrams within each symmetric window """
            print('counting bigrams...')
            def slice_zip(w1, w2):
                r = [' ']*len(w1+w2)
                r[::2] = w1
                r[1::2] = w2
                return r
            
            wz = self.windowsize-1
            for w in zip(*[self.text[i:] for i in range(1+wz*2)]):
                left = list(w[0:wz])
                right = list(w[wz+1:])
                for bigram in itertools.product([w[wz]], left+right):
                    context = slice_zip(left[::-1], right)
                    min_dist = context.index(bigram[1])
                    """ Force items into dictionary as it is faster
                    than performing key comparisons """
                    try:
                        self.distances[bigram].append(min_dist)
                    except:
                        self.distances[bigram] = [min_dist]
                    finally:
                        yield bigram

        def count_bigrams_forward():
            """ Calculate bigrams within each forward-looking window """
            print('counting bigrams...')
            for w in zip(*[self.text[i:] for i in range(self.windowsize)]):
                for bigram in itertools.product([w[0]], w[1:]):    
                    yield bigram

        def count_bigrams_forward_dist():
            """ Calculate bigrams within each forward-looking window,
            calculate also average distance between words. Distance
            tracking is not included into count_bigrams_forward()
            for better efficiency """
            print('counting bigrams...')
            for w in zip(*[self.text[i:] for i in range(self.windowsize)]):
                for bigram in itertools.product([w[0]], w[1:]):
                    """ Force items into dictionary as it is faster
                    than performing key comparisons """
                    try:
                        self.distances[bigram].append(w[1:].index(bigram[1]))
                    except:
                        self.distances[bigram] = [w[1:].index(bigram[1])]
                    finally:
                        yield bigram

        """ Selector for window type and distance tracking """
        if self.symmetry:
            if self.track_distance:
                bigram_freqs = Counter(count_bigrams_symmetric_dist())
            else:
                bigram_freqs = Counter(count_bigrams_symmetric())
        else:
            if self.track_distance:
                bigram_freqs = Counter(count_bigrams_forward_dist())
            else:
                bigram_freqs = Counter(count_bigrams_forward())

        """ Score bigrams by given measure """
        print('calculating scores...')
        for bigram in bigram_freqs.keys():
            w1, w2 = bigram[0], bigram[1]
            if is_valid(w1, w2, bigram_freqs[bigram]):
                translation = self.get_translation(w2)
                distance = self.get_distance(bigram)
                freq_w1 = self.word_freqs[w1]
                freq_w2 = self.word_freqs[w2]
                score = measure.score(scale(bigram_freqs[bigram]),
                                      freq_w1, freq_w2, self.corpus_size)
                self.scored.append((bigram, translation, bigram_freqs[bigram],
                            freq_w1, freq_w2, self._trim_float(score),
                            distance))

        scored = sorted(self.scored)

        for x in scored:
            print(x)

def demo():
    st = time.time()
    a = Associations()
    a.set_constraints(windowsize = 10,
                      freq_threshold = 2,
                      symmetry=False,
                      track_distance=False,
                      words1=['Aššur'])
    
    a.score_bigrams('neoA', PPMI2)
    et = time.time() - st
    print('time', et)

demo()
