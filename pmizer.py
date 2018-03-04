#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import sys
import time
import itertools
import math
import json
import re
try:
    from dictionary import dct as akkadian_dict
except ImportError:
    print('dictionary.py not found!')
    akkadian_dict = {}


__version__ = "2018-03-03"

WINDOW_SCALING = False    # Apply window size penalty to scores
LOGBASE = 2               # Logarithm base; set to None for ln
LACUNA = '_'              # Symbol for lacunae in cuneiform languages
BUFFER = '<BUFFER>'       # Buffer symbol; added after each line

""" ====================================================================
Association measures - Aleksi Sahala 2018 - University of Helsinki =====
========================================================================

/ Deep Learning and Semantic Domains in Akkadian Texts
/ Centre of Excellence in Ancient Near Eastern Empires
/ Language Bank of Finland


========================================================================
General description ====================================================
========================================================================

This script calculates different word association measures derived from
PMI (Pointwise Mutual Information). In Language Technology, the PMI is
used to find collocations and associations between words.

By its basic definition, PMI is the ratio of p(w1,w2), i.e. the actual
probability that two words are co-occurring within a certain distance
from each other to p(a)p(b), i.e. the expected chance of those words
co-occurring independently.

                    p(w1,w2)          
  PMI(w1,w2) = log ----------         
                   p(w1)p(w2)

PMI score of 0 indicates perfect independence. Scores greater than 0 may
point to a possible collocation, and scores lesser than 0 indicate that
words are found together less often than expected by chance.

With small window sizes (i.e. the maximum distance between the words),
PMI can be used to find fixed expressions and compound words such as
´police car´, ´subway station´, ´kick (the) bucket´. With larger
window sizes PMI usually finds more abstract semantic concepts:
´Microsoft ... software´, ´banana ... fruit´, ´admiral ... navy´.

For a practical example, below are listed the best ten collocates for
the Neo-Assyrian word ´kakku´ [weapon], using a window size of 10:

 PMI     Colloc.   Translation.
 9.081   ezzu      [to be furious, angry] 
 7.9028  maqâtu    [to fall, to defeat]
 7.7721  tiâmtu    [sea] (*)
 7.2058  tâhâzu    [battle]
 6.5646  nakru     [enemy]
 6.2976  sharrûtu  [kingship]
 6.1073  dannu     [having strength]
 5.9267  nashû     [lifted]
 5.8615  Ashur     [God Ashur]
 5.0414  âlu       [city]

(* this comes from a formulaic expression found in the Neo-Assyrian
royal inscriptions, where all the people from the ´Upper Sea´ (Medi-
terranean) to the ´Lower Sea´ (Persian Gulf) were subjugated under
the Assyrian rule by ´great weapons granted by the god Ashur´).

From these, we can reconstruct some kind of prototypical semantic field
for ´weapon´ as it was probably perceived by the ancient Assyrians, at
least in the context of royal inscriptions:

    WEAPON is an object that is
      - LIFTED to intimidate and to show power
      - associated with FURY, ANGER and STRENGTH
      - used in BATTLES to DEFEAT ENEMIES and their CITIES between
        the SEAS in order to enforce the Assyrian KINGSHIP.
      - great weapons (= success in battle) is granted by the GOD ASHUR


========================================================================
Associations.set_properties(*kwargs) ===================================
========================================================================

Constraints and properties may be set by using the following kwargs:

 ´windowsize´       (int) collocational window that defines the
                    maximum mutual distance of the elements of a bigram.
                    Minimum distance is 2, which means that the words
                    are next to each another.

 ´freq_threshold´   (int) minimum allowed bigram frequency.

 ´symmetry´         (bool) Use symmetric window. If not used, the window
                    is forward-looking. For example, with a window size
                    of 3 and w4 being our word of interest:

                                 w1 w2 w3 w4 w5 w6 w7
                    symmetric       +--+--^--+--+
                    asymmetric            ^--+--+
  
 ´words1´           (list) words of interest: Bigram(word1, word2)
 ´words2´           Words of interest may also be expressed as compiled
                    regular expressions. See exampes in ´stopwords´.

 ´stopwords´        (list) discard uninteresting words like
                    prepositions, numbers etc. May be expressed as
                    compiled regular expressions. For example
                    [re.compile('\d+?'), re.compile('^[A-ZŠṢṬĀĒĪŪ].+')]
                    will discard all numbers written as digits, as well
                    as all words that begin with a capital letter.

                    NOTE: It may be wise to express series of regular
                    expressions as disjunctions (regex1|...|regexn) as it
                    makes the comparison significantly faster, e.g. 
                    [re.compile('^(\d+?|[A-ZŠṢṬĀĒĪŪ].+)'].

 ´track_distance´   (bool) calculate average distance between the words
                    of the bigram. If the same bigram can be found
                    several times within the window, only the closest
                    distance is taken into account.

                    NOTE: Slows bigram counting 2 or 3 times depending
                    on the window size. Using large (>15) symmetric
                    window and distance tracking takes lots of time.

 ´distance_scaling´ Scale scores by using mutual distances instead of
                    window size.


========================================================================
Associations.read_XXXXX(filename) ======================================
========================================================================

Associations.read_raw(filename)

  Takes a lemmatized raw text file as an input. For example, a text
  "Monkeys ate coconuts and the sun was shining" would be:

     monkey eat coconut and the sun be shine

  Bigrams are NOT allowed to span from line to another. Thus, if you
  want to disallow, collocations spanning from sentence to another, the
  text should contain one sentence per line.


Associations.read_vrt(filename, word_attribute, delimiter)

  Reads VRT files. You must define the ´word_attribute´ index (int),
  from which the lemmas can be found. ´delimiter´ is used to set the
  boundary, over which collocates are not allowed to span. Normally this
  is either ´<text>´, ´<paragraph>´ or ´<sentence>´, but may as well be
  ´<clause>´ or ´<line>´, too, if such are available in the file.

NOTE: Window size must always be specified before reading the file!


========================================================================
Associations.score_bigrams(measure) ====================================
========================================================================

Argument ´measure´ must be one of the following:

    PMI             Pointwise mutual information (Church & Hanks 1990)
    NPMI            Normalized PMI (Bouma 2009)
    PMI2            PMI^2 (Daille 1994)
    PMI3            PMI^3 (Daille 1994)
    PPMI            Positive PMI. As PMI but discards negative scores.
    PPMI2           Positive PMI^2 (Role & Nadif 2011)


========================================================================
Associations.export_json(filename), Associations.import_json(filename) =
========================================================================

Association scores can be exported as a JSON dump by using method
Associations.export_json(filename). This file can be later imported to
produce different outputs without need to recalculate the scores.

When a JSON is imported, the results can be filtered by using the
set_constraints(). Naturally, changing the window size won't have any
effect as the scores have been calculated by using a certain window,
but the results can be filtered with frequency threshold, stop words and
new words of interest.

Thus, a score table that might take 45 seconds to compute, can be
re-searched with new parameters in a fraction of that time.


========================================================================
Output formats =========================================================
========================================================================

Associations.print_matrix(value, scoretable)

 ´value´           Value that will be used in the matrix: ´score´,
                   bigram ´frequency´ or ´distance´.

 ´scoretable´      Imported JSON score table. Use this in case you have
                   previously exported your scores. If you import a
                   scoretable, you must also use set_constraints()
                   in order to limit the search.

                   NOTE: matrices should only be used with a small
                   pre-defined set of words of interest.

==================================================================== """

def _log(n):
    if LOGBASE is None:
        return math.log(n)
    else:
        return math.log(n, LOGBASE)
    
class PMI:
    """ Pointwise Mutual Information. The score orientation is
    -log p(a,b) > 0 > -inf """
    @staticmethod
    def score(ab, a, b, cz):
        return _log(ab*cz) - _log(a*b)

class NPMI:
    """ Normalized PMI. The score orientation is  +1 > 0 > -1 """
    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) / -_log(ab/cz)

class PMI2:
    """ PMI^2. Fixes the low-frequency bias of PMI and NPMI by squaring
    the numerator to compensate multiplication done in the denominnator.
    Scores are oriented as: 0 > log p(a,b) > -inf """
    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) - (-_log(ab/cz))

class PMI3:
    """ PMI^3 (no low-freq bias, favors common bigrams). Scores are
    oriented from 0 > -(k-1)*log p(a,b) > -inf, where the k stands for
    the power of the numerator, here hardcoded as 3. """
    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) - (-(2*_log(ab/cz)))

class PPMI:
    """ Positive PMI. Works as the regular PMI but discards negative
    scores: -log p(a,b) > 0 = 0 """
    @staticmethod
    def score(ab, a, b, cz):
        return max(PMI.score(ab, a, b, cz), 0)

class PPMI2:
    """ Positive derivative of PMI^2. Shares exaclty the same
    properties but the score orientation is on the positive
    plane: 1 > 2^log p(a,b) > 0 """
    @staticmethod
    def score(ab, a, b, cz):
        return 2 ** PMI2.score(ab, a, b, cz)


class Associations:

    def __init__(self):
        self.text = []
        self.scored = {'freqs': {},
                       'translations': {},
                       'collocations': {},
                       'words1': [],
                       'words2': []}
        self.windowsize = None
        self.freq_threshold = 1
        self.symmetry = False    
        self.words = {1: [], 2: []}
        self.stopwords = ['', LACUNA, BUFFER]
        self.regex_stopwords = []
        self.regex_words = {1: [], 2: []}
        self.distances = {}
        self.track_distance = False
        self.distance_scaling = False
        self.log_base = LOGBASE
        self.window_scaling = WINDOW_SCALING

    def __repr__(self):
        """ Return variables for log file """
        debug = []
        tab = max([len(k)+2 for k in self.__dict__.keys()])
        for k, v in self.__dict__.items():
            if k not in ['scored', 'text', 'regex_stopwords',
                         'regex_words', 'distances', 'anywords', 'anywords1',
                         'anywords2']:
                debug.append('%s%s%s' % (k, ' '*(tab-len(k)+1), str(v)))
        return '\n'.join(debug) + '\n'

    def _readfile(self, filename):
        if self.windowsize is None:
            print('Window size not defined ...')
            sys.exit()

        self.filename = filename
        with open(filename, 'r', encoding="utf-8") as data:
            print('reading %s ...' % filename)
            self.text = [BUFFER]*self.windowsize
            return data.readlines()

    def read_raw(self, filename):
        """ Open raw lemmatized input file with one text per line.
        Add buffer equal to window size after each text to prevent
        words from different texts being associated. """
        buffers = 1
        for line in self._readfile(filename):
            self.text.extend(line.strip('\n').split(' ')
                             + [BUFFER]*self.windowsize)
            buffers += 1
        self.corpus_size = len(self.text) - (buffers * self.windowsize)

    def read_vrt(self, filename, lemmapos, delimiter=''):
        """ Open VRT file. Takes arguments ´lemmapos´ (int), which
        indicates the word attribute count for lemmas, i.e. 1 would
        be the first attribute, as the 0th position is reserved for
        the word. Argument ´delimiter´ splits the text, e.g. by its
        ´sentence´, ´paragraph´ or ´text´ element and disallows
        collocations being recognized if the delimiter is found
        between them. """
        delimiter = '</{}>'.format(re.sub('\W', '', delimiter))
        buffers = 1
        for line in self._readfile(filename):
            l = line.strip('\n')
            if l == delimiter:
                self.text.extend([BUFFER]*self.windowsize)
                buffers += 1
            if not l.startswith('<'):
                self.text.append(l.split('\t')[lemmapos])
            else:
                pass
        self.corpus_size = len(self.text) - (buffers * self.windowsize)

    def import_json(self, filename):
        """ Load lookup table from JSON """
        print('reading %s ...' % filename)
        with open(filename) as data:
            return json.load(data)
        
    def export_json(self, filename):
        """ Save lookup table as JSON """
        print('writing %s ...' % filename)
        with open(filename, 'w', encoding="utf-8") as data:
            json.dump(self.scored, data)
            
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
        
    def _trim_float(self, number):
        if number == '':
            return number
        elif isinstance(number, int):
            return number
        else:
            return float('{0:.4f}'.format(number))

    def get_translation(self, word):
        """ Get translation from dictionary """
        try:
            translation = '[{}]'.format(akkadian_dict[word])
        except:
            translation = '[?]'
        return translation

    def get_distance(self, bigram):
        """ Calculate average distance for bigram's words; if not
        used, the distance will be equal to window size. """
        if self.track_distance:
            distance = self._trim_float(sum(self.distances[bigram])
                                    / len(self.distances[bigram]))
        else:
            distance = self.windowsize
        return distance

    def match_regex(self, words, regexes):
        """ Matches a list of regexes to list of words """
        return any([re.match(r, w) for r in regexes for w in words])

    def is_stopword(self, *words):
        """ Compare words with stop word list and regexes. Return
        True if not in the list """
        if not self.regex_stopwords:
            return any(w in self.stopwords for w in words)
        else:
            return self.match_regex(words, self.regex_stopwords)\
                   or any(w in self.stopwords for w in words)

    def is_wordofinterest(self, word, index):
        """ Compare words with the list of words of interest.
        Return True if in the list """
        if not self.regex_words[index]:
            return word in self.words[index]
        else:
            return self.match_regex([word], self.regex_words[index])\
                   or word in self.words[index]
    
    def is_valid(self, w1, w2, freq):
        """ Validate bigram. Discard stopwords and those which
        do not match with the word of interest lists """
        if freq >= self.freq_threshold:    
            if not self.anywords:
                return not self.is_stopword(w1, w2)
            elif self.anywords and self.anywords2:
                return self.is_wordofinterest(w1, 1) and\
                       self.is_wordofinterest(w2, 2)
            else:
                if self.anywords1:
                    return self.is_wordofinterest(w1, 1) and\
                           not self.is_stopword(w2)
                if self.anywords2:
                    return self.is_wordofinterest(w2, 2) and\
                           not self.is_stopword(w1)
                else:
                    return False
        else:
            return False

    def score_bigrams(self, measure):
        """ Main function for bigram scoring """
        
        if not self.text:
            print('Input text not loaded.')
            sys.exit()
        
        def scale(bf, distance):
            """ Scale bigram frequency with window size. Makes the
            scores comparable with NLTK/Collocations PMI measure """
            if WINDOW_SCALING and not self.distance_scaling:
                return bf / (self.windowsize - 1)
            if WINDOW_SCALING and self.distance_scaling:
                return bf / (distance)
            else:
                return bf

        def count_bigrams_symmetric():
            """ Calculate bigrams within each symmetric window """
            print('counting bigrams ...')
            wz = self.windowsize-1
            for w in zip(*[self.text[i:] for i in range(1+wz*2)]):
                for bigram in itertools.product([w[wz]], w[0:wz]+w[wz+1:]):
                    yield bigram

        def count_bigrams_symmetric_dist():
            """ Calculate bigrams within each symmetric window and
            track distances. """

            def chain(w1, w2):
                """ Make zip-chain from two lists.
                [a, b], [c, d] -> [a, c, b, d] """
                chain = [' ']*len(w1+w2)
                chain[::2] = w1
                chain[1::2] = w2
                return chain

            print('counting bigrams ...')
            wz = self.windowsize-1
            for w in zip(*[self.text[i:] for i in range(1+wz*2)]):
                left = list(w[0:wz])
                right = list(w[wz+1:])
                for bigram in itertools.product([w[wz]], left+right):
                    context = chain(left[::-1], right)
                    min_dist = math.floor(context.index(bigram[1])/2) +1
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
            print('counting bigrams ...')
            for w in zip(*[self.text[i:] for i in range(self.windowsize)]):
                for bigram in itertools.product([w[0]], w[1:]):
                    yield bigram

        def count_bigrams_forward_dist():
            """ Calculate bigrams within each forward-looking window,
            calculate also average distance between words. Distance
            tracking is not included into count_bigrams_forward()
            for better efficiency """
            print('counting bigrams ...')
            for w in zip(*[self.text[i:] for i in range(self.windowsize)]):
                bigrams = enumerate(itertools.product([w[0]], w[1:]))
                for bigram in bigrams:
                    """ Force items into dictionary as it is faster
                    than performing key comparisons """
                    try:
                        self.distances[bigram[1]].append(bigram[0] + 1)
                    except:
                        self.distances[bigram[1]] = [bigram[0] + 1]
                    finally:
                        yield bigram[1]

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

        """ Score bigrams by given measure. Produces a lookup dictionary
        in the following structure:

        ´collocations´
         | 
         +--´word 1´
         |    |  
         |    +-- ´collocate 1 for word 1´
         |    |     |
         |    |     +-- ´bigram freq´  (int)
         |    |     +-- ´word freqs´   [w1, w2] (int)
         |    |     +-- ´translations´ [w1, w2] (str)
         |    |     +-- ´score´        (float)
         |    |     +-- ´distance´     (float)
         |    |
         |    +-- ´collocate 2 for word 1´
         |    |     |

        Translations and frequencies are stored under ´translations´
        and ´freqs´, which include keys for each word/bigram.
        
        """
        
        word_freqs = Counter(self.text)
        w1list, w2list = [], [] # containers for found words of interest
        print('calculating scores ...')        
        for bigram in bigram_freqs.keys():
            w1, w2 = bigram[0], bigram[1]
            if self.is_valid(w1, w2, bigram_freqs[bigram]):
                distance = self.get_distance(bigram)
                freq_w1 = word_freqs[w1]
                freq_w2 = word_freqs[w2]
                score = measure.score(scale(bigram_freqs[bigram], distance),
                                      freq_w1, freq_w2, self.corpus_size)
                data = {'score': score,
                        'distance': distance,
                        'frequency': bigram_freqs[bigram]}
                self.scored['translations'][w1] = self.get_translation(w1)
                self.scored['translations'][w2] = self.get_translation(w2)
                self.scored['freqs'][w1] = freq_w1
                self.scored['freqs'][w2] = freq_w2
                w1list.append(w1)
                w2list.append(w2)
                try:
                    self.scored['collocations'][w1][w2] = data
                except KeyError:
                    self.scored['collocations'][w1] = {}
                    self.scored['collocations'][w1][w2] = data
                finally:
                    pass

        """ Store words of interest for JSON """
        self.scored['words1'] = list(set(w1list))
        self.scored['words2'] = list(set(w2list))

    def _filter_json(self, words, index):
        """ Validate and return words of interest """
        return [w for w in sorted(words) if self.is_wordofinterest(w, index)\
                and not self.is_stopword(w)]

    def print_matrix(self, value, table=None):
        print('generating {} matrix ...'.format(value))
        """ Make a collocation matrix of two sets of words of
        interest. Argument ´value´ must be ´score´, ´frequency´
        or ´distance´. """
        
        """ Use self.scored if imported JSON is not given.
        Apply word filters if JSON is loaded """
        if table is None:
            from_json = True
            table = self.scored
            words1 = sorted(table['words1'])
            words2 = sorted(table['words2'])
        else:
            from_json = True
            words1 = self._filter_json(table['words1'], 1)
            words2 = self._filter_json(table['words2'], 2)

        rows = [[value.upper() + ' MATRIX W1 ->'] + words2]
        for w1 in words1:
            row = []
            for w2 in words2:
                score = ''
                if w1 in table['collocations'].keys():
                    if w2 in table['collocations'][w1].keys():
                        bigram = table['collocations'][w1][w2]
                        bigram_freq = bigram['frequency']
                        if from_json:
                            if bigram_freq > self.freq_threshold:
                                score = bigram[value]
                        else:
                            score = bigram[value]
                row.append(self._trim_float(score))
            if any(row):
                rows.append([w1] + row)

        """ Rotate matrix to clean empty columns """
        for r in [row for row in zip(*rows) if any(row[1:])]:
            print('\t'.join([str(m) for m in r]))


def demo():
    st = time.time()
    
    # Initialize Associations
    a = Associations()
    a.set_constraints(windowsize = 15,
                      freq_threshold = 20,
                      symmetry=True,
                      track_distance=False,
                      distance_scaling=False,
                      words1=[re.compile('E.+?')],
                      words2=[re.compile('[A-Z].+?')])

    #a.read_vrt('testi.vrt', 1, '<sentence>')
    a.read_raw('neoA')
    a.score_bigrams(PMI)
    a.export_json('kokeilu.json')
    #b = a.import_json('kokeilu.json')
    a.print_matrix('score')
    et = time.time() - st
    print('time', et)

demo()
