#!/usr/bin/python
# -*- coding: utf-8 -*-

#from dictionary import dct

import itertools
import json
import math
import re
import statistics
import sys
import time
from urllib.parse import quote
import random
from collections import Counter

__version__ = "2021-01-01"

print('pmizer.py version %s\n' % __version__)

""" Constants """
WINDOW_SCALING = True     # Apply window size penalty to scores
LOGBASE = 2               # Logarithm base; set to None for ln
LACUNAE = ['_']           # List of symbols for lacunae or removed words
LINEBREAK = '<LB>'        # Line break or text boundary
BUFFER = '<BF>'           # Buffer/padding symbol 
INDENT = 4                # Indentation level for print
METASEPARATOR = '|'       # Character for separating multi-dimensional
                          # metadata for JSON 
WRAPCHARS = ['[', ']']    # Wrap translations/POS-tags between these
                          # symbols, e.g. ['"'] for "string". Give two
                          # if beginning and end symbols are different
DECIMALSEPARATOR = '.'    # Decimal separator for output files
HIDE_MIN_SCORE = True     # Hide minimum scores in matrices
VERBOSE = True            # Print more info

IGNORE = [LINEBREAK, BUFFER]

""" /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

Critical bug fixes:

        2019-06-26: changed buffering from the end of the line
                    to the beginning to prevent rare crash that
                    occurred if the keyword was coincidentally
                    in the middle of the first symmetric window.
                    now linebreak is always the first symbol in text.

        2020-02-10: lazy context similarity algorithm now preserves
                    lacuna positions. other algorithms are not
                    recommended as they are not fixed yet.

        2020-03-03: regular expressions now work properly in forward-
                    looking windows.

                    symmetric window scaling is now correct and
                    Σ f(a,b) = Σ f(a) = Σ f(b) = N when I(σ+;σ+)
                    where σ is symbol of the alphabet.

        2020-05-20: Fix incorrect bounds for PMI3. Lazy context
                    similarity measure now discards collocates
                    from counts properly (i.e. subtracts 1 from
                    the denominator).


Other fixes:

        2020-11-27: Preweight is now default.

        2020-01-01: Additional measures such as Jaccard etc.
                    

How to use? =====================================================

(1) Create text object (text per line, lemmas separated by space)

    text = Text('oracc-akkadian.txt')

(2) Calculate co-occurrencies for the text object

    cooc = Associations(text,
                 words1=['*'],            # All words to all words
                 formulaic_measure=Lazy,  # Use CSW
                 minfreq_b = 1,           # Min freq of b 
                 minfreq_ab = 1,          # Min co-oc freq of a and b
                 symmetry=True,           # Window symmetry 
                 windowsize=5,            # Window size 
                 factorpower=2)           # k-value

(3) Calculate PMI from co-occurrences from the associations object

    results = cooc.score(PMI2)            # Select association measure

(4) Print results from

    x.print_scores(results, limit=1000, gephi=True, filename='oracc.pmi')



/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ """

""" ====================================================================
Logarithm base definiton; log-base 2 should be used by default =========
==================================================================== """

def _log(n):
    if LOGBASE is None:
        return math.log(n)
    else:
        return math.log(n, LOGBASE)


def make_korp_oracc_url(w1, w2, wz):
    """ Generate URL for Oracc in Korp """
    w1 = re.sub('(.+)_.+?', r'\1', w1)
    w2 = re.sub('(.+)_.+?', r'\1', w2)
    base = 'https://korp.csc.fi/test-as/?mode=other_languages#'\
           '?lang=fi&stats_reduce=word'
    cqp = '&cqp=%5Blemma%20%3D%20%22{w1}%22%5D%20%5B%5D%7B0,'\
          '{wz}%7D%20%5Blemma%20%3D%20%22{w2}%22%5D'\
          .format(w1=quote(w1), w2=quote(w2), wz=wz)
    corps = '&corpus=oracc_adsd,oracc_ario,oracc_blms,oracc_cams,oracc_caspo,oracc_ctij'\
            ',oracc_dcclt,oracc_dccmt,oracc_ecut,oracc_etcsri,oracc_hbtin,oracc_obmc,'\
            'oracc_riao,oracc_ribo,oracc_rimanum,oracc_rinap,oracc_saao,'\
            'oracc_others&search_tab=1&search=cqp&within=paragraph'
    return base+cqp+corps

""" ====================================================================
Input / Output tools ===================================================
==================================================================== """

class IO:

    """ Basic file IO-operations and verbose """

    def read_file(filename):
        print(': Reading %s' % filename)
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as data:
                return data.read().splitlines()
        except FileNotFoundError:
            IO.errormsg("File not found: %s" % filename)
            sys.exit(0)

    def write_file(filename, content):
        with open(filename, 'w', encoding='utf-8') as data:
            data.write(content)
        print(': Saved %s' % filename)

    def export_json(filename, content):
        """ Save lookup table as JSON """
        with open(filename, 'w', encoding="utf-8") as data:
            json.dump(content, data)
        print(': Saved %s' % filename)
        
    def import_json(filename):
        """ Load lookup table from JSON """
        try:
            print(': Reading %s' % filename)
            with open(filename, encoding='utf-8') as data:
                return json.load(data)
        except FileNotFoundError:
            IO.errormsg("File not found: %s" % filename)
            sys.exit(0)

    def show_time(time_, process):
        if VERBOSE:
            print("%s%s took %0.2f seconds" \
                  % (" "*INDENT, process, round(time_, 2)))

    def printmsg(message):
        if VERBOSE:
            print(message)

    @staticmethod
    def errormsg(message):
        print(": Error! %s" % message)

            
""" ====================================================================
Word association measures ==============================================

Measures take four arguments:

   ab = co-oc freq
   a  = freq of a
   b  = freq of b
   cz = corpus size
   factor = CSW value (this is used if postweight is set)
   oo = estimated number of all bigrams in the corpus

==================================================================== """

## POINTWISE MUTUAL INFORMATION BASED MEASURES

class PMI:
    """ Pointwise Mutual Information. The score orientation is
    -log p(a,b) > 0 > -inf. As in Church & Hanks 1990. """
    minimum = -math.inf

    @staticmethod
    def raw(ab, a, b, cz, oo=None):
        return (ab/cz) / ((a/cz)*(b/cz))

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        return factor * (_log(ab*cz) - _log(a*b))

class PMIDELTA:
    """ Smooth PMI (Pantel & Lin 2002). This measure reduces
    the PMI score more, the rarer the words are, thus reducing
    the low-frequency bias """
    minimum = -math.inf

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        weight = (ab/(ab+1)) * (min(a,b)/(min(a,b,)+1))
        return factor * weight * (_log(ab*cz) - _log(a*b))

class PMICDS:
    """ Context distribution smoothed PMI (Levy, Goldberg &
    Dagan 2015). This measure raises the f(b) to the power of
    alpha = 0.75 """
    minimum = -math.inf

    ## WORKS ONLY IN MATRIX FACTORIZATION

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        alpha = 0.75
        return max(factor * _log((cz*ab) / (a*(b**alpha))),0)

class NPMI:
    """ Normalized PMI. The score orientation is  +1 > 0 > -1
    as in Bouma 2009 """
    minimum = -1.0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        return factor * (PMI.score(ab, a, b, cz, 1) / -_log(ab/cz))


class PMISIG:
    """ Washtell & Markert (2009). """
    minimum = -math.inf

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        pa = a / cz
        pb = b / cz
        return factor * (math.sqrt(min(pa, pb))\
                         * (PMI.raw(ab, a, b, cz)))

class SCISIG:
    """ Washtell & Markert (2009) The original publication does
    not tell the score orientation, but it can be shown to be
    +1 > √p(a,b) > 0 """
    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        pa = a / cz
        pb = b / cz
        pab = ab / cz
        return factor * (math.sqrt(min(pa, pb))\
                         * (pab / ((pa) * math.sqrt(pb))))


class cPMI:
    """ Corpus Level Significant PMI as in Damani 2013. According to
    the original research paper, delta value of 0.9 is recommended """
    minimum = -math.inf

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        delta = 0.9
        t = math.sqrt(_log(delta) / (-2 * a))
        return _log(ab / (a * b / cz) + a * t) * factor


class PMI2:
    """ PMI^2. Fixes the low-frequency bias of PMI and NPMI by squaring
    the numerator to compensate multiplication done in the denominnator.
    Scores are oriented as: 0 > log p(a,b) > -inf. As in Daille 1994 """
    minimum = -math.inf
    
    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        return (PMI.score(ab, a, b, cz, 1) - (-_log(ab/cz))) / factor

        
class PMI3:
    """ PMI^3 (no low-freq bias, favors common bigrams). Although not
    mentioned in any papers at my disposal, the scores are oriented
    log p(a,b) > 2 log p(a,b) > -inf. As in Daille 1994"""
    minimum = -math.inf
    
    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        return (PMI.score(ab, a, b, cz, 1) - (-(2*_log(ab/cz)))) / factor

class SPMI:
    """ Positive shifted PMI. Works as the regular PMI but discards negative
    scores: -log p(a,b) > 0 = 0; Shift by 3 """
    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        return max(3 + PMI.score(ab, a, b, cz, factor), 0)

class PPMI:
    """ Positive PMI. Works as the regular PMI but discards negative
    scores: -log p(a,b) > 0 = 0 """
    minimum = 0
    
    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        return max(PMI.score(ab, a, b, cz, factor), 0)


class PPMI2:
    """ Positive derivative of PMI^2 as in Role & Nadif 2011.
    Shares exaclty the same properties but the score orientation
    is on the positive plane: 1 > 2^log p(a,b) > 0 """
    minimum = 0
    
    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        return factor * (2 ** PMI2.score(ab, a, b, cz, 1))

class PPMI3:
    """ Positive derivative of PMI^3. Shares exaclty the same properties
    but the score orientation is: p(a,b) > p(a,b)^2 > 0

    Not mentioned in Role & Nadif """
    
    minimum = 0
    
    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        return factor * (2 ** PMI3.score(ab, a, b, cz, 1))
 

class NPMI2:
    """ NPMI^2. Removes the low-frequency bias as PMI^2 and has
    a fixed score orientation as NPMI: 1 > 0 = 0. Take cube root
    of the result to trim excess decimals. Sahala & Linden (2020) """

    minimum = 0
    
    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        ind = 2 ** _log(ab / cz)
        base_score = 2 ** PMI2.score(ab, a, b, cz, 1) - ind
        return (max(base_score / (1 - ind), 0) * factor) ** (1/3)


class NPMI3:
    """ NPMI^3. Removes the low-frequency bias as PMI^3 and has
    a fixed score orientation as NPMI: 1 > 0 = 0. Take cube root
    of the result to trim excess decimals. Sahala & Linden (2020) """

    minimum = 0
    
    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        pab = (ab/cz)
        base_score = 2 ** PMI3.score(ab, a, b, cz, 1) - (pab**2)
        return (max(base_score / (pab - (pab**2)), 0) * factor) ** (1/3)

# Other measures and statistical tests

class NormalizedExpectation:
    """ Normalized Expectation as in Pecina 2006

       2f(xy) / f(x)f(y) """

    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        return (2*ab*factor) / (a+b)


class tTest:
    """ Student's t-test """

    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        ind = a*b/cz
        return ((ab*factor)-ind) / math.sqrt(factor*ab*(1-(factor*ab/cz)))


class zScore:
    """ Z-score """

    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        ind = a*b/cz
        return ((ab*factor)-ind) / math.sqrt(ind*(1-(ind/cz)))

# Association coefficients, See Pecina 2006

def c_table(ab, a, b, cz, factor, oo=None):
    """ Return contingency table. Note that the total number of
    co-occurrences in the corpus is based on an estimate, because
    not all bigrams are calculated for efficiency """
    ab = ab*factor
    A = ab
    B = a - ab
    C = b - ab
    D = oo - ab
    return A, B, C, D


class Jaccard:

    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        A, B, C, D = c_table(ab, a, b, cz, factor, oo)
        return A / (A+B+C)


class Odds:

    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        A, B, C, D = c_table(ab, a, b, cz, factor, oo)
        return (A*D)/(B*C)



class Simpson:
    """ Note: this has an extreme low-freq bias """
    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        A, B, C, D = c_table(ab, a, b, cz, factor, oo)
        
        return A / min(A+B, A+C)


class BraunBlanquet:

    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        A, B, C, D = c_table(ab, a, b, cz, factor, oo)
        
        return A / max(A+B, A+C)


class Pearson:

    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        A, B, C, D = c_table(ab, a, b, cz, factor, oo)
        u = (A * D) - (B * C)
        d = math.sqrt((A+B)*(A+C)*(D+B)*(D+C))
        return u / d

    
class UnigramSubtuples:

    minimum = 0

    @staticmethod
    def score(ab, a, b, cz, factor, oo=None):
        A, B, C, D = c_table(ab, a, b, cz, factor, oo)

        subs = [1/A, 1/B, 1/C, 1/D]
        return _log((A*D)/(B*C)) - (3.29 * math.sqrt(sum(subs)))

""" ====================================================================
Context Similarity Weighting ===========================================
========================================================================

An initial version of CSW as in Sahala & Linden 2020. Performs similarly
but has a space complexity of O(n×m^2) (n = corpus size, m = window len)

==================================================================== """

class FormulaicMeasures:
    @staticmethod
    def compensate(words, collocate, uniqs):
        """ Compensate window matrix counts:
        ´compensation´ is a number of each word that should be
        ignored in counting (metasymbols, collocates) """            
        compensation = uniqs
        for symbol in LACUNAE + [BUFFER, LINEBREAK, collocate]:
            compensation += max(words.count(symbol) - 1, 0)
        return compensation


class Greedy:
    """ Removed temporarily """
    pass


class Lazy:   
    @staticmethod
    def score(windows, collocate):
        diffs = []
        for window in zip(*windows[::-1]):
            """ Count the number of unique words in transposed window matrix
            ie. stack windows and count uniques, see compensate() for more
            info """
            uniqs = len(set(window))
            compensated = FormulaicMeasures.compensate(window, collocate, uniqs)
            diffs.append((len(window) - compensated) / len(window))
            """ Uncomment to see probabilities """
            #print('\t'.join(window), (len(window) - compensated) / len(window))
        return sum(diffs) / max((len(diffs) - 1),1)


""" ====================================================================
Text container and basic text analysis tools ===========================
==================================================================== """

class Text(object):

    def __init__(self, filename, ignore=LACUNAE):
        self.filename = filename
        self.content = []
        self.content_uniq = []
        self.metadata = []
        self.documents = []
        self.translations = {}
        self.ignore = ignore
        
        self._read(filename)

    def __repr__(self):
        return self.stats
    
    @staticmethod
    def _tokenize(string):
        return string.strip().split(' ')

    @staticmethod
    def _count(symbols, line):
        return len([s for s in line if s in symbols])

    def _read(self, filename):

        st = time.time()

        """ Read input text from lemmatized raw text file """
        metalength = None
        self.lacunacount = 0
        self.maxlen = 0
        self.linecount = 0
        self.corpus_size = 0
        
        for line in IO.read_file(self.filename):
            if line:
                self.linecount += 1
                fields = line.split('\t')
                text = fields[-1]

                if metalength is None:
                    metalength = [len(fields)]
                elif len(fields) not in metalength:
                    IO.errormsg('(%s at line %i): Inconsistent '\
                          'number of metadata fields.' % (self.filename,
                                                          self.linecount))
                    sys.exit(0)
                
                if len(fields) > 1:
                    """ Collect metadata """
                    meta = METASEPARATOR.join(fields[0:-1])
                    self.metadata.append(meta)

                lemmas = self._tokenize(text)
                """ Store documents for TF-IDF """
                self.documents.append(lemmas)
                self.lacunacount += self._count(self.ignore, lemmas)

                """ Lacunae are count as lemmas but buffers are not """
                lemmacount = len(lemmas)
                self.corpus_size += lemmacount
                if lemmacount > self.maxlen:
                    self.maxlen = lemmacount
                    
                self.content.extend([LINEBREAK] + [BUFFER] + lemmas)

        """ Add buffer to the end of the file """
        self.content.extend([BUFFER])

        """ Make frequency list """
        self.word_freqs = Counter(self.content)

        st2 = time.time() - st
        IO.show_time(st2, "Reading")

        IO.printmsg(self.stats)

    @property
    def stats(self):
        """ Return corpus statistics """
        tab = INDENT * ' '
        non_words = [BUFFER, LINEBREAK]
        freqs = sorted([f for f in self.word_freqs.values()])
        log = [('\n: Text statistics:'),
               ('%sLines: %i' % (tab, self.linecount)),
               ('%sLongest line: %i' % (tab, self.maxlen)),
               ('%sWord count: %i' % (tab, self.corpus_size)),
               ('%sWord count (non-lacunae): %i' \
                % (tab, self.corpus_size - self.lacunacount)),
               ('%sLacunae or ignored symbols: %i' % (tab, self.lacunacount)),
               ('%sUnique words: %i' \
                % (tab, len(self.word_freqs.keys()) - len(non_words))),
               ('%sMedian word frequency: %i' \
                % (tab, statistics.median(freqs))),
               ('%sAverage word frequency: %.2f' \
                % (tab, sum(freqs) / len(freqs)))]
        return '\n'.join(log) + '\n'

    @property
    def metadata_stats(self):
        """ Return word and line counts for each metadata group """
        if not self.metadata:
            return None
        
        meta = {}
        tab = ' '*INDENT
        for index, content in enumerate(self.documents):
            wordcount = len(content)
            meta.setdefault(self.metadata[index], {'words': 0, 'lines': 0})
            meta[self.metadata[index]]['words'] += wordcount
            meta[self.metadata[index]]['lines'] += 1

        return '\n'+'\n'.join([('%s%s\t%i\t%i' \
                % (tab, k.replace(METASEPARATOR, '\t'), v['words'], v['lines']))\
                          for k, v in sorted(meta.items())]) + '\n'
    
    def iterate(self, windowsize=1):
        """ Iterate content of the text and extend buffers to match
        window size """
        for word in self.content:
            if word == BUFFER:
                for i in range(0, windowsize):
                    yield word
            else:
                yield word


    def tf_idf(self, threshold=0):
        """ Returns a TF-IDF based stopword list based on the text.
        This can be passed to Associations() as a keyword argument

        ´threshold´ defines the size of the list, if no argument is
        give, will return a list relative to corpus size """
        
        print(': Making TF-IDF stopword list')
        
        st = time.time()
        tf_idfs = {}
        words = []
        
        if threshold == 0:
            threshold = int(0.000005 * self.corpus_size)

        for document in self.documents:
            N = len(document)

            for word in set(document):
                t = document.count(word)
                tf_idfs.setdefault(word, {'tf': [], 'found_in': 0})
                tf_idfs[word]['tf'].append(t/N)
                tf_idfs[word]['found_in'] += 1

        for word, vals in tf_idfs.items():
            scores = []
            for tf in vals['tf']:
                scores.append(tf * math.log(len(self.documents)/vals['found_in'], 10))
            words.append([sum(scores), word])

        st2 = time.time() - st
        IO.show_time(st2, "TF-IDF")
        return [x[1] for x in sorted(words, reverse=True)[0:threshold]]

    def read_dict(self):
        filename = self.filename.split('.')[0] + '.dict'
        with open(filename, 'r', encoding='utf-8', errors='ignore') as data:
            for line in data.read().splitlines():
                key, value = line.split('\t')
                self.translations[key] = value

    def uniquify(self, wz):
        # This feature is not finished
        
        """ Produce a version of text that do not need window scaling.
        Iterate text and disallow words occurring more than once
        withing a given distance from each other. Replace non-unique
        words with lacunae """
        print(': Uniquifying windows')
        st = time.time()
        count = 0
        count_non_lacunae = 0
        removed = 0
        for word in self.content:
            """ Initialize buffer """
            if word == LINEBREAK:
                buffer = []
            """ Keep buffer length """
            if len(buffer) == wz + 1:
                buffer.pop(0)
            """ Replace non-uniques with lacunae """
            if word in buffer and word not in LACUNAE + IGNORE:
                removed += 1
                word = LACUNAE[0]

            if word not in IGNORE:
                count += 1
                if word not in LACUNAE:
                    count_non_lacunae += 1
            buffer.append(word)
            self.content_uniq.append(word)

        self.corpus_size_uniq = Counter(self.content_uniq)

        st2 = time.time() - st
        IO.show_time(st2, "Uniquifying")
        print("%s--> %i words removed" \
              % (' '*INDENT, removed))
        
        #for k, v in sorted(Counter(removed).items()):
        #    print(v, k)

    """ ================================================================
    Random sampling tools (for measure evaluation purposes)
    ================================================================ """

    """ Sample a population of words from frequency list from
    given frequency range. ´quantity´ is the population size and
    ´freq_range´ a list that contains the min and max freq,
    e.g. [30,50] """

    def pick_random(self, quantity, freq_range):
        sampled = []
        for k, v in self.word_freqs.items():
            if freq_range[1] > v > freq_range[0]:
                sampled.append(k)
        self.random_sample = random.sample(sampled, quantity)
        return self.random_sample

    """ Random samples can be saved and loaded with the
    following funtions """

    def save_random(self, filename):
        with open(filename, 'w', encoding='utf-8') as data:
            data.write('\n'.join(self.random_sample))

    def load_random(self, filename):
        with open(filename, 'r', encoding='utf-8') as data:
            self.random_sample = data.read().splitlines()
            return self.random_sample
        
""" ====================================================================
Association measure tools ==============================================
==================================================================== """

class Associations:

    def __init__(self, text, **kwargs):
        if not isinstance(text, Text):
            IO.errormsg('Association must have Text object as argument.')
            sys.exit(0)
            
        self.text = text
        self.word_freqs = self.text.word_freqs
        self.corpus_size = self.text.corpus_size
    
        self.windowsize = 2
        self.minfreq_b = 1
        self.minfreq_ab = 1
        
        self.distances = {}
        self.WINS = {}

        self.translations = {}
        if self.text.translations:
            self.translations = self.text.translations

        self.track_distance = False
        self.symmetry = False
        self.track_distance = False
        self.positive_condition = False
        self.formulaic_measure = None
        self.postweight = False
        self.factorpower = 1

        self.words = {1: [], 2: []}
        self.regex_words = {1: [], 2: []}

        self.metadata = {}
        
        self.conditions = {'stopwords': LACUNAE + ['', BUFFER, LINEBREAK],
                           'stopwords_regex': [],
                           'conditions': [],
                           'conditions_regex': []}

        self.set_constraints(**kwargs)
        self.count_bigrams()

        if self.corpus_size < self.windowsize:
            IO.errormsg('Window size exceeds corpus size.')
            sys.exit(1)

    def __repr__(self):
        """ Define what is not shown in .log files """
        debug = []
        tab = max([len(k)+2 for k in self.__dict__.keys()])
        for k in sorted(self.__dict__.keys()):
            if k not in ['scored', 'text', 'regex_stopwords', 'metadata',
                         'regex_words', 'distances', 'anywords',
                         'anywords1', 'output', 'anywords2', 'bigram_freqs',
                         'anycondition', 'word_freqs', 'positive_condition',
                         'minimum', 'WINS', 'documents', 'translations']:
                v = self.__dict__[k]
                debug.append('%s%s%s' % (k, ' '*(tab-len(k)+1), str(v)))

        return '\n'.join(debug) + '\n' + '-'*20 +\
               ' \npmizer version: ' + __version__

    def set_constraints(self, **kwargs):
        """ Set constraints. Separate regular expressions from the
        string variables, as string comparison is significantly faster
        than re.match() """
        
        for key, value in kwargs.items():
            if key in ['stopwords', 'conditions']:
                for word in value:
                    if isinstance(word, str):
                        self.conditions[key].append(word)
                    else:
                        self.conditions[key+'_regex'].append(word)
            elif key in ['words1', 'words2']:
                index = int(key[-1])
                for word in value:
                    if isinstance(word, str):
                        self.words[index].append(word)
                    else:
                        self.regex_words[index].append(word)
            else:
                setattr(self, key, value)

        """ Combine tables for faster comparison """
        self.anywords = any([self.words[1], self.words[2],
                         self.regex_words[1], self.regex_words[2]])
        self.anywords1 = any([self.words[1], self.regex_words[1]])
        self.anywords2 = any([self.words[2], self.regex_words[2]])
        self.anycondition = any([self.conditions['conditions'],
                                 self.conditions['conditions_regex']]) 

    """ ================================================================
    Helper funtions ====================================================
    ================================================================ """

    def _trim_float(self, number):
        if not number:
            return number
        elif isinstance(number, int):
            return number
        else:
            return round(number, 3)

    def _get_translation(self, word):
        """ Get translation from dictionary """
        try:
            translation = '%s%s%s' % (WRAPCHARS[0], self.translations[word], WRAPCHARS[-1])
        except:
            translation = '%s?%s' % (WRAPCHARS[0], WRAPCHARS[-1])
        return translation

    def _get_distance(self, bigram):
        """ Calculate average distance for bigram's words; if not
        used, the distance will be equal to window size. """
        if self.track_distance:
            distance = self._trim_float(sum(self.distances[bigram])
                                    / len(self.distances[bigram]))
        else:
            distance = ''
        return distance

    def _match_regex(self, words, regexes):
        """ Matches a list of regexes to list of words """
        return any([re.match(r, w) for r in regexes for w in words])

    def _meets_anycondition(self, condition, words):
        """ Compare words with stopword/conditions list and regexes. """
        if not self.conditions[condition +'_regex']:
            return any(w in self.conditions[condition] for w in words)
        else:
            return self._match_regex(words, self.conditions[condition+'_regex'])\
                   or any(w in self.conditions[condition] for w in words)

    def _is_wordofinterest(self, word, index):
        """ Compare words with the list of words of interest.
        Return True if in the list; never accept lacunae or buffers """

        if self.words[1] == ['*'] and word not in [LINEBREAK, BUFFER]:
            return True

        if word in [LINEBREAK, BUFFER]:
            return False
        
        if not self.regex_words[index]:
            return word in self.words[index]
        else:
            return self._match_regex([word], self.regex_words[index])\
                   or word in self.words[index]
        
    def _is_valid(self, w1, w2, freq):
        """ Validate bigram. Discard stopwords and those which
        do not match with the word of interest lists """
        if freq >= self.minfreq_ab and self.word_freqs[w2] >= self.minfreq_b:    
            if not self.anywords:
                return not self._meets_anycondition('stopwords', [w1, w2])
            elif self.anywords and self.anywords2:
                return self._is_wordofinterest(w1, 1) and\
                       self._is_wordofinterest(w2, 2)
            else:
                if self.anywords1:
                    return self._is_wordofinterest(w1, 1) and\
                           not self._meets_anycondition('stopwords', [w2])
                if self.anywords2:
                    return self._is_wordofinterest(w2, 2) and\
                           not self._meets_anycondition('stopwords', [w1])
                else:
                    return False
        else:
            return False

    def _has_condition(self, window):
        """ Check if conditions are defined. Validate window if true """
        if not self.anycondition:
            return True
        else:
            if self.positive_condition:
                if self._meets_anycondition('conditions', window):
                    return True
                else:
                    return False
            elif not self.positive_condition:
                if not self._meets_anycondition('conditions', window):
                    return True
                else:
                    return False
            else:
                print('positive_condition must be True or False')
                sys.exit(1)


    """ ================================================================
    Bigram counting ====================================================
    ================================================================ """

    def count_bigrams(self):

        print(': Counting bigrams')

        st = time.time()
        
        """ Set has_meta if metadata is available. """
        has_meta = len(self.text.metadata) > 0
        
        text = list(self.text.iterate())
        
        def _check_formulaic(bigram, window, index):
            """ Store windows only if formulaic_measures are used,
            otherwise skip this to save memory and time. Remove
            index of the collocate from the window to preserve only
            context of the bigram """
            if self.formulaic_measure is not None:
                #window.pop(index)
                window[index] = LACUNAE[0]
                self.WINS.setdefault(bigram, []).append(window)
            return bigram

        def _gather_meta(bigram, meta):
            """ Get bigram distribution by metadata """
            self.metadata.setdefault(bigram, {})
            self.metadata[bigram].setdefault(meta, 0)
            self.metadata[bigram][meta] += 1
        
        def count_bigrams_symmetric():
            """ Symmetric window """
            wz = self.windowsize - 1
            #dupes = []
            for w in zip(*[text[i:] for i in range(1+wz*2)]):
                #W = [o for o in w if o not in ('_', '<LB>', '<BF>')]
                #UNIQ = sorted(list(set(W)))
                #if UNIQ == sorted(W):
                #    pass
                #else:
                #    dupes.append(W)
                    
                if w[0] == LINEBREAK:
                    if has_meta:
                        meta = self.text.metadata.pop(0)
                if self._is_wordofinterest(w[wz], 1) and \
                   self._has_condition(w[0:wz]+w[wz+1:]):
                    for index, bigram in enumerate(itertools.product([w[wz]],
                                                    w[0:wz]+w[wz+1:])):
                        if has_meta:
                            _gather_meta(bigram, meta)
                        yield _check_formulaic(bigram,
                                               list(w[0:wz]+w[wz+1:]),
                                               index)
            #for x in dupes:
            #    print(x)

        def count_bigrams_symmetric_dist():
            """ Symmetric window and distance tracking. """

            def chain(w1, w2):
                """ Return convolution chain of two lists.
                [a, b], [c, d] -> [a, c, b, d] """
                chain = [' '] * len(w1+w2)
                chain[::2] = w1
                chain[1::2] = w2
                return chain

            wz = self.windowsize - 1
            for w in zip(*[text[i:] for i in range(1+wz*2)]):
                left = list(w[0:wz])
                right = list(w[wz+1:])
                if w[0] == LINEBREAK:
                    if has_meta:
                        meta = self.text.metadata.pop(0)
                if self._is_wordofinterest(w[wz], 1) and \
                   self._has_condition(left+right):
                    for index, bigram in enumerate(itertools.product([w[wz]],
                                                    left+right)):
                        bigram = _check_formulaic(bigram, left+right, index)
                        context = chain(left[::-1], right)
                        min_dist = math.floor(context.index(bigram[1])/2) + 1
                        if has_meta:
                            _gather_meta(bigram, meta)
                        self.distances.setdefault(bigram, []).append(min_dist)
                        yield bigram

        def count_bigrams_forward():
            """ Calculate bigrams within each forward-looking window """
            for w in zip(*[text[i:] for i in range(self.windowsize)]):
                if w[0] == LINEBREAK:
                    """ Keep track of lines and their metadata """
                    if has_meta:
                        meta = self.text.metadata.pop(0)
                if self._is_wordofinterest(w[0], 1) \
                        and self._has_condition(w[1:]):
                    for index, bigram in enumerate(itertools.product([w[0]],
                                                    w[1:])):
                        if has_meta:
                            """ If metadata is available, store it """
                            _gather_meta(bigram, meta)
                        yield _check_formulaic(bigram, list(w[1:]), index)

        def count_bigrams_forward_dist():
            """ Calculate bigrams within each forward-looking window,
            calculate also average distance between words. Distance
            tracking is not included into count_bigrams_forward()
            for better efficiency """
            for w in zip(*[text[i:] for i in range(self.windowsize)]):
                if w[0] == LINEBREAK:
                    if has_meta:
                        meta = self.text.metadata.pop(0)
                if self._is_wordofinterest(w[0], 1) and self._has_condition(w[1:]):
                    for index, bigram in enumerate(itertools.product([w[0]], w[1:])):
                        bg = _check_formulaic(bigram, list(w[1:]), index)
                        if has_meta:
                            _gather_meta(bg, meta)
                        d = index + 1
                        self.distances.setdefault(bg, []).append(d)
                        yield bg

        """ Selector for window type and distance tracking """
        # TODO: Slice text into smaller parts to prevent memory errors
        # when analysing large texts
        
        if self.symmetry:
            if self.track_distance:
                self.bigram_freqs = Counter(count_bigrams_symmetric_dist())
            else:
                self.bigram_freqs = Counter(count_bigrams_symmetric())
        else:
            if self.track_distance:
                self.bigram_freqs = Counter(count_bigrams_forward_dist())
            else:
                self.bigram_freqs = Counter(count_bigrams_forward())

        st2 = time.time() - st
        IO.show_time(st2, "Bigram counting")
        
    def score(self, measure):

        """ Score text by using given measure and return dictionary
        containing the results """

        print(': Scoring bigrams')

        st = time.time()        

        def scale(bf):
            """ Scale bigram frequency with window size to assure
            Σ f(a,b) = Σ f(a) = Σ f(b) = N regardless of window size """
            if WINDOW_SCALING:
                if self.symmetry:
                    return bf / (self.windowsize - 1) / 2
                else:
                    return bf / (self.windowsize - 1)
            else:
                return bf

        def apply_weight(ab, a, b, cs, F):
            # Apply only to joint-probability
            abf = ab * (F**self.factorpower)

            return abf, a, b, cs
            
        self.measure = measure.__name__

        """ Declare container for collocation data """
        scored = {'freqs': {},
                  'translations': {},
                  'collocations': {},
                  'words1': [],
                  'words2': []}
        w1list, w2list = [], []

        """ Score and store bigrams, translations etc. """
        F_MEASURE = self.formulaic_measure

        # DEBUG: Test that Σ f(a,b) = Σ f(a) = Σ f(b) = N holds
        # when calculating I(σ+;σ+)
        
        _abs = 0
        _as = sum([v for k, v in self.word_freqs.items() if k not in IGNORE])

        """ Set scoring function according to weighting type """
        #if self.preweight:
        #    SCORE = measure.score_pre_csim
        #else:
        #    SCORE = measure.score

        """ EXPERIMENTAL:
        Estimate number of all bigrams for contingency tables """
        all_bigrams = scale((self.windowsize-1) * self.corpus_size)
        
        for bigram in self.bigram_freqs.keys():
            w1, w2 = bigram

            # DEBUG, see above
            _abs += scale(self.bigram_freqs[bigram])

            if self._is_valid(w1, w2, self.bigram_freqs[bigram]):

                freq_w1 = self.word_freqs[w1]
                freq_w2 = self.word_freqs[w2]
                distance = self._get_distance(bigram)

                """ Apply context similarity measure"""
                if self.formulaic_measure is not None:
                    csim_factor = F_MEASURE.score(self.WINS[bigram], w2)
                else:
                    csim_factor = 0

                """ Smooth for zero-division errors """
                csim_factor = max(1-csim_factor, 00000.1)

                """ Apply CSW to joint distribution if postweight not selected;
                otherwise apply CSW on the final score """               
                if self.formulaic_measure is not None and not self.postweight:
                    ab, a, b, cs = apply_weight(self.bigram_freqs[bigram],
                                                freq_w1,
                                                freq_w2,
                                                self.corpus_size,
                                                csim_factor)
                    factor = 1
                else:
                    ab, a, b, cs = (self.bigram_freqs[bigram],
                                    freq_w1,
                                    freq_w2,
                                    self.corpus_size)
                    factor = csim_factor
                    
                score = measure.score(scale(ab), a, b, cs, factor, all_bigrams)

                data = {'score': score,
                        'distance': distance,
                        'frequency': self.bigram_freqs[bigram],
                        'similarity': 1-csim_factor,
                        'metadata': self.metadata.get(bigram, None)}
                scored['translations'][w1] = self._get_translation(w1)
                scored['translations'][w2] = self._get_translation(w2)
                scored['freqs'][w1] = freq_w1
                scored['freqs'][w2] = freq_w2
                w1list.append(w1)
                w2list.append(w2)
                # TODO: use setdefault instead of force
                # TODO: rearrange results into printable format on the fly
                #       instead of nested JSON which is slow to parse
                try:
                    scored['collocations'][w1][w2] = data
                except KeyError:
                    scored['collocations'][w1] = {}
                    scored['collocations'][w1][w2] = data
                finally:
                    pass

        # DEBUG, see above: print f(a,b) and f(b) = f(a) if calculated for *
        print('DEBUG:sanity check:',round(_abs), _as)
        for k,v in self.bigram_freqs.items():
            pass#print(k,v)

        """ Store words of interest for JSON """
        scored['words1'] = list(set(w1list))
        scored['words2'] = list(set(w2list))
        scored['minimum'] = measure.minimum

        st2 = time.time() - st
        IO.show_time(st2, "Scoring bigrams")
        
        return scored

    """ ================================================================
    Pretty-printing results ============================================
    ================================================================ """

    def _stringify(self, array):
        """ Convert all values in table into strings and
        localize decimal markers. """
        def _format_decimal(item):
            if isinstance(item, float):
                return str(item).replace('.', DECIMALSEPARATOR)
            else:
                return str(item)
        return [_format_decimal(x) for x in array]

    def _sort_by_index(self, table, indices):
        """ Sort table by given two indices, i.e. [0, 4] sorts
        the table by 1st and 5th values """
        i = indices[0]
        j = indices[-1]
        return sorted(table, key=lambda item: (item[i], item[j]), reverse=True)

    def print_matrix(self, scores, value='score', filename=None):
        """ Arguments: ´scores´ dictionary (or JSON) produced by
        Associations.score(); ´value´ must be ´score´, ´frequency´ or
        ´distance´; set ´filename´ to write output into a file """

        print(': Building score matrix')
        st = time.time()        

        output = []
        heading = [value.upper() + ' W1 -->']
        heading += ['{}'.format(w + ' ' + scores['translations'][w])\
                    for w in scores['words2']]
        rows = [heading]        
        for w1 in sorted(scores['words1']):
            row = []
            for w2 in sorted(scores['words2']):
                if HIDE_MIN_SCORE:
                    score = ''
                else:
                    score = scores['minimum']
                if w1 in scores['collocations'].keys():
                    if w2 in scores['collocations'][w1].keys():
                        bigram = scores['collocations'][w1][w2]
                        score = bigram.get(value, None)
                        if score is None:
                            IO.errormsg('bad argument "%s" for print_matrix().' % value)
                            sys.exit(0)
                row.append(self._trim_float(score))
            if any(row) and w1 not in LACUNAE + [BUFFER, LINEBREAK]:
                rows.append([w1] + row)

        """ Rotate to clean empty columns """
        for r in [row for row in zip(*rows) if any(row[1:])]:
            output.append('\t'.join([str(x) for x in r]))

        st2 = time.time() - st
        IO.show_time(st2, "Building matrix")

        if filename is not None:
            IO.write_file(filename, '\n'.join(output))
        else:
            print('\n'.join(output))

    def print_scores(self, scores, limit=10000, sortby=('word1', 'score'),
                     gephi=False, filename=None):

        st = time.time()
        print(': Building score table')
        
        def merge_dict(dict1, dict2):
           """ Merge dictionaries and sum the sign frequencies """
           combined = {**dict1, **dict2}
           for key, value in combined.items():
               if key in dict1 and key in dict2:
                   combined[key] = value + dict1[key]
           return combined

        def collate_meta(meta):
            """ Combine multi-value metadata; e.g. SB|Nippur|Literary
            will be split into three additional subdictionaries and
            frequency data is collated. """

            # TODO: Aliases

            if meta is None:
                return meta
            
            vals = len(list(meta.keys())[0].split('|'))
            original = {i+1:{} for i in range(0, vals)}
            original[0] = meta            
            for key, val in meta.items():
                for slot, k in enumerate(key.split('|')):
                    original[slot+1] = merge_dict({k: val}, original[slot+1])
            return original  

        header = ['word1', 'attr1', 'word2', 'attr2', 'word1 freq', 'word2 freq',
                  'bigram freq', 'score', 'distance', 'similarity', 'url']
        sort_indices = [header.index(x) for x in sortby]

        """ Set headers """
        if gephi:
            output = [(';'.join(['source', 'target', 'weight']))]
        else:
            output = ['\t'.join([x for x in header])]
            
        rows = []
        for w1 in scores['collocations'].keys():
            for w2 in scores['collocations'][w1].keys():
                bigram = scores['collocations'][w1][w2]
                freqs = scores['freqs']
                meta = collate_meta(bigram['metadata'])
                items = {'word1': w1,
                         'word2': w2,
                         'attr1': scores['translations'][w1],
                         'attr2': scores['translations'][w2],
                         'word1 freq': freqs[w1],
                         'word2 freq': freqs[w2],
                         'bigram freq': bigram['frequency'],
                         #'metaa': meta,
                         'score':
                             float(self._trim_float(bigram['score'])),
                         'distance': bigram['distance'],
                         'similarity': self._trim_float(bigram['similarity']),
                         'url': make_korp_oracc_url(w1, w2, self.windowsize-2)}
                rows.append([items[key] for key in header])

        """ Sort and convert into tsv """
        lastword = ''
        for line in self._sort_by_index(rows, sort_indices):
            if not gephi:
                if lastword != line[0]:
                    i = 0
                if i < limit:
                    output.append('\t'.join(self._stringify(line)))
                lastword = line[0]
                i += 1
            if gephi:
                if lastword != line[0]:
                    i = 0
                if i < limit:
                    data = [line[0]+' '+line[1]] + [line[2]+' '+line[3]+' ('+str(line[5]) +')'] + [line[7]]
                    output.append('\t'.join(self._stringify(data)))
                lastword = line[0]
                i += 1


        st2 = time.time() - st
        IO.show_time(st2, "Building score table")
        
        if filename is None:
            print('\n'.join(output))
        else:
            IO.write_file(filename, '\n'.join(output))



z = Text('data/akk.txt')
z.read_dict()
wz = 5
x = Associations(z, words1=['nakru'],
                 formulaic_measure=Lazy,
                 minfreq_b = 2,
                 minfreq_ab = 2,
                 symmetry=True,
                 windowsize=wz,
                 factorpower=3)
A = x.score(UnigramSubtuples)
x.print_scores(A, limit=15, gephi=True, filename='oracc.pmi')


