#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import urllib
import sys
import datetime
import time
import itertools
import math
import json
import re
import godlist
import random
try:
    from dictionary import dct as dct #root
except ImportError:
    print('dictionary.py not found!')
    dct = {}

__version__ = "2019-06-26"
print('pmizer.py version %s\n' % __version__)

WINDOW_SCALING = True    # Apply window size penalty to scores
LOGBASE = 2               # Logarithm base; set to None for ln
LACUNA = '_'              # Symbol for lacunae in cuneiform languages
BUFFER = '<BF>'           # Buffer symbol; added after each line/text
LINEBREAK = '<LB>'        # Marks line / text boundaries
DECIMAL = '.'             # Decimal point marker
INDENT = ' '*4            # Indentation depth
WRAPCHARS = ['[', ']']    # Wrap translations/POS-tags between these
                          # symbols, e.g. ['"'] for "string". Give two
                          # if beginning and end symbols are different
MYLLY = False             # Add Mylly-prefixes
HIDE_MIN_SCORE = True     # Hide minimum scores in matrices

X = []                    # Debugging (store stuff here)
Z = {}                    # Debugging (store stuff here)
DEBUG = 'ilu'


""" ====================================================================
pmizer.py - Aleksi Sahala 2018 - University of Helsinki ================
========================================================================

/ Deep Learning and Semantic Domains in Akkadian Texts
/ Centre of Excellence in Ancient Near Eastern Empires
/ Language Bank of Finland

/ http://github.com/asahala

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

JSON structure:

  JSON
    |
    +---´collocations´
    |    | 
    |    +--´word 1´
    |    |    |  
    |    |    +-- ´collocate 1 for word 1´
    |    |    |     |
    |    |    |     +-- ´frequency´    (int) bigram freq
    |    |    |     +-- ´translations´ [w1, w2] (str)
    |    |    |     +-- ´score´        (float)
    |    |    |     +-- ´distance´     (float)
    |    |    |
    |    |    +-- ´collocate 2 for word 1´
    |    |    |     |
    |   ...  ...   ...
    |
    +---´translations´
    |    |
    |    +--´word´  (dict)
    |    |
    |   ...
    |
    +---´freqs´
    |    |
    |    +--´word´  (int)
    |    |
    |   ...
    |
    +---´words1´ (list of words1 of interest)
    +---´words2´ (list of words2 of interest)
     
========================================================================
Using dictionaries and/or POS-tagging ==================================
========================================================================

If you are processing a foreign language and want to have translations
in your score tables, you may import a dictionary structured as

   {´lemma´: ´translation´, ...}

This dictionary should be saved as a Python file and imported as ´dct´.
Dictionaries can be used to define your words of interest and stopwords
by their translations. Translations can be matched by using strings
or compiled regular expressions. For example:

  wis = has_translation(['enemy', 'opponent', 'rival'])

will find all the Akkadian words that have any of these translations.
The result can be passed to the set_constraints() like any list of
stopwords or words of interest.

The dictionary doesn't necessarily have to contain translations, but
they may as well be POS-tags or whatever you may find useful. However,
it may be more useful to suffix the POS tags after the lemmas and use
regular expressions to filter them, e.g. dog_N, cat_N, eat_V.

Dictionary search methods are:

:: get_freqs_by_translation(translations, sort_by)

  DESCRIPTION        This method will return a frequency list of words
                     by their translation in your corpus.

  ´translations´     (list) A list of translations or compiled regular
                     expressions.

  ´sort_by´          (int) Sort index.

:: get_freqs_by_lemma(lemmas)

  DESCRIPTION        Prints frequencies of each lemma in the corpus.

  ´lemmas´           (list) A list of lemmas.

:: has_translation(translations)

  DESCRIPTION        This method will return a list of lemmas that match
                     the given translations. This list can be passed
                     to set_constraints() as described above.
  ´translations´     As above.

==================================================================== """

def _log(n):
    if LOGBASE is None:
        return math.log(n)
    else:
        return math.log(n, LOGBASE)


def _make_korp_oracc_url(w1, w2, wz):
    """ Generate URL for Oracc in Korp """
    w1 = re.sub('(.+)_.+?', r'\1', w1)
    w2 = re.sub('(.+)_.+?', r'\1', w2)
    base = 'https://korp.csc.fi/test-as/?mode=other_languages#'\
           '?lang=fi&stats_reduce=word'
    cqp = '&cqp=%5Blemma%20%3D%20%22{w1}%22%5D%20%5B%5D%7B0,'\
          '{wz}%7D%20%5Blemma%20%3D%20%22{w2}%22%5D'\
          .format(w1=urllib.parse.quote(w1), w2=urllib.parse.quote(w2), wz=wz)
    corps = '&corpus=oracc_adsd,oracc_ario,oracc_blms,oracc_cams,oracc_caspo,oracc_ctij'\
            ',oracc_dcclt,oracc_dccmt,oracc_ecut,oracc_etcsri,oracc_hbtin,oracc_obmc,'\
            'oracc_riao,oracc_ribo,oracc_rimanum,oracc_rinap,oracc_saao,'\
            'oracc_others&search_tab=1&search=cqp&within=paragraph'
    return base+cqp+corps


""" ====================================================================
Word association measures ==============================================
==================================================================== """

class PMI:
    """ Pointwise Mutual Information. The score orientation is
    -log p(a,b) > 0 > -inf """
    minimum = -math.inf

    @staticmethod
    def score(ab, a, b, cz):
        return _log(ab*cz) - _log(a*b)

class NPMI:
    """ Normalized PMI. The score orientation is  +1 > 0 > -1 """
    minimum = -1.0

    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) / -_log(ab/cz)

class PMIsig:
    """ Washtell & Markert (2009) """
    minimum = -math.inf

    @staticmethod
    def score(ab, a, b, cz):
        pa = a / cz
        pb = b / cz
        return math.sqrt(min(pa, pb)) * 2**(PMI.score(ab, a, b, cz))

class SCISIG:
    """ Washtell & Markert (2009) """
    minimum = -math.inf

    @staticmethod
    def score(ab, a, b, cz):
        pa = a / cz
        pb = b / cz
        pab = ab / cz
        return math.sqrt(min(pa, pb)) * (pab / ((pa) * math.sqrt(pb)))
    
class cPMI:
    """ Corpus Level Significant PMI as in Damani 2013. According to
    the original research paper, delta value of 0.9 is recommended """
    minimum = -math.inf

    @staticmethod
    def score(ab, a, b, cz):
        delta = 0.9
        t = math.sqrt(_log(delta) / (-2 * a))
        return _log(ab / (a * b / cz) + a * t)

class PMI2:
    """ PMI^2. Fixes the low-frequency bias of PMI and NPMI by squaring
    the numerator to compensate multiplication done in the denominnator.
    Scores are oriented as: 0 > log p(a,b) > -inf """
    minimum = -math.inf
    
    @staticmethod
    def score(ab, a, b, cz):

        #pab = (ab/cz)**2
        #papb = (a/cz)*(b/cz)
        #return pab, papb
        return PMI.score(ab, a, b, cz) - (-_log(ab/cz))
    
class PMI3:
    """ PMI^3 (no low-freq bias, favors common bigrams). Scores are
    oriented from 0 > -(k-1)*log p(a,b) > -inf, where the k stands for
    the power of the numerator, here hardcoded as 3. """
    minimum = -math.inf
    
    @staticmethod
    def score(ab, a, b, cz):
        return PMI.score(ab, a, b, cz) - (-(2*_log(ab/cz)))

class PPMI:
    """ Positive PMI. Works as the regular PMI but discards negative
    scores: -log p(a,b) > 0 = 0 """
    minimum = 0
    
    @staticmethod
    def score(ab, a, b, cz):
        return max(PMI.score(ab, a, b, cz), 0)

class PPMI2:
    """ Positive derivative of PMI^2. Shares exaclty the same
    properties but the score orientation is on the positive
    plane: 1 > 2^log p(a,b) > 0 """
    minimum = 0
    
    @staticmethod
    def score(ab, a, b, cz):
        return math.sqrt(2 ** PMI2.score(ab, a, b, cz))

class NPMI2:
    """ NPMI^2. Removes the low-frequency bias as PMI^2 and has
    a fixed score orientation as NPMI: 1 > 0 = 0. Take square root
    of the result to trim excess decimals. Sahala (2019) """
    @staticmethod
    def score(ab, a, b, cz):
        ind = 2 ** _log(ab / cz)
        base_score = 2 ** PMI2.score(ab, a, b, cz) - ind
        return math.sqrt(max(base_score / (1 - ind), 0))
        #return math.log(max(base_score / (1 - ind), 0) + 0.000000000001)

""" ====================================================================
Repetitiveness "Formulaic" measures ====================================
========================================================================

These can be used to detect and score formulaic expressions. There
are three measures. In the examples W stands for our keyword
and X for its collocate. Other letters represent arbitrary words. Note
that linebreaks and paddings are taken into account, thus a bigram that
always occurs in a beginning or end of a line and will be considered
formulaic.

Greedy:

    Greedy measure represents the amount of repetitiveness within
    the windows where our bigrams occur. A score of 1.0 would indicate
    that the collocate only occurs in formulaic expressions (regardless
    if there are one or several different expressions!). A score
    of 0.5 indicates that half of the contexts are repeated. The
    contexts do not have to be exactly similar in order to be
    accepted as formulaic, e.g. if in some instance one
    word in the expression is different it will only slightly
    decrease the repetitiveness score (as in the case below).

             symmetric window of 3

             a   b   W   d    X
             a   b   W   d    X
             a   b   W   e    X
    sim.     1.0 1.0 _   0.66 _   --> repetitiveness: 0.89 (avg)


Strict:

    Strict measure represents the ratio of unique expressions to
    formulaic expressions that are exactly similar. A score of
    0.66 would indicate that 2/3 of the contexts are strictly
    formulaic.
    
             a   b   W   c   d   X    1.0
             a   b   W   c   d   X    1.0
             a   b   W   c   e   X    0.0
    sim.                              0.66


Lazy:

    Lazy measure represents the percentage of repetitive content,
    but it does not take into account one instance of each repeated
    word in the same position in an expression. Thus, a score of 0.56
    would indicate that 56% of the content in expressions is repeated.

             a    b    W   d    X   
             a    b    W   d    X
             a    b    W   e    X 
    sim.     0.66 0.66 _   0.33 _   = 0.56 (avg)
    
    The greedy and lazy measures can give quite different results.
    In the following, greedy indicates that 100% of the expressions
    are formulaic, while lazy indicates thet 50% of the content is
    repeated.

    Greedy   a    b    W    d     X
             a    b    W    d     X 
             1.0  1.0  _    1.0   _   = 1.0

    Lazy     a    b    W    d     X
             a    b    W    d     X 
             0.5  0.5  _    0.5   _   = 0.5

==================================================================== """

class FormulaicMeasures:
    @staticmethod
    def compensate(words, collocate, uniqs):
        """ Compensate window matrix counts:
        ´compensation´ is a number of each word that should be
        ignored in counting (metasymbols, collocates) """            
        compensation = uniqs
        for symbol in [LACUNA, BUFFER, LINEBREAK, collocate]:
            compensation += max(words.count(symbol) - 1, 0)
        return compensation

class Greedy:
    @staticmethod
    def score(windows, forced, collocate):
        repeated = []
        for words in zip(*windows[::-1]):
            counts = Counter(words).values()
            singles = len([x for x in counts if x == 1])
            repeated.append(1-(singles/len(words)))
            #if collocate == 'wēdu': print(words)
        return sum(repeated) / len(repeated)

class Strict:
    @staticmethod
    def score(windows, forced, collocate):
        all_ = len(windows)
        uniqs = len(set([''.join(x) for x in windows]))
        if uniqs == 1:
            return 1.0
        else:
            return 1 - ((all_ - uniqs) / all_)

class Lazy:
    @staticmethod
    def score(windows, forced, collocate):
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
        #print(1 - (sum(diffs) / len(diffs)), sum(diffs), len(diffs))
        return sum(diffs) / len(diffs)

class Associations:

    def __init__(self):
        """ set_constraints() populates constructor """
        self.text = []
        self.output = []
        self.output_format = None
        self.scored = {'freqs': {},
                       'translations': {},
                       'collocations': {},
                       'words1': [],
                       'words2': []}
        self.measure = None
        self.windowsize = None
        self.freq_threshold = 1
        self.freq_threshold_collocate = 1
        self.symmetry = False    
        self.words = {1: [], 2: []}
        self.conditions = {'stopwords': ['', LACUNA, BUFFER, LINEBREAK],
                           'stopwords_regex': [],
                           'conditions': [],
                           'conditions_regex': []}
        self.positive_condition = False
        self.regex_words = {1: [], 2: []}
        self.distances = {}
        self.track_distance = False
        self.log_base = LOGBASE
        self.window_scaling = WINDOW_SCALING
        self.date = datetime.datetime.now()
        self.WINS = {}
        self.formulaic_measure = None ## DOCUMENT THIS
        self.formulaic_forced = False
        self.metalist = [] # container for metadata
        self.metadata = {}
        self.documents = []
        """ Dictionary of translations """
        self.translations = dct

    def __repr__(self):
        """ Define what is not shown in .log files """
        debug = []
        tab = max([len(k)+2 for k in self.__dict__.keys()])
        for k in sorted(self.__dict__.keys()):
            if k not in ['scored', 'text', 'regex_stopwords', 'regex_words',
                         'distances', 'anywords', 'anywords1', 'output',
                         'anywords2', 'anycondition', 'word_freqs',
                         'positive_condition', 'minimum', 'WINS', 'documents',
                         'translations']:
                v = self.__dict__[k]
                debug.append('%s%s%s' % (k, ' '*(tab-len(k)+1), str(v)))

        return '\n'.join(debug) + '\n' + '-'*20 +\
               ' \npmizer version: ' + __version__

    """ ================================================================
    Metadata ===========================================================
    ================================================================ """

    def _abbreviate(self, meta):
        """ Abbreviate and combine metadata according to definitions
        in godlist.py """
        
        m = {'period': meta[0].lower(), 'genre': meta[1].lower()}
        for k, v in m.items():
            if v in godlist.abbrevs[k].keys():
                yield godlist.abbrevs[k][v]
            else:
                print('  Metadata warning: "%s" unspecified.' % v)
                yield '_'
    
    """ ================================================================
    File ops ===========================================================
    ================================================================ """

    def _readfile(self, filename):
        """ General file reader """
        if self.windowsize is None:
            print('Window size not defined ...')
            sys.exit()
            
        self.filename = filename
        with open(filename, 'r', encoding="utf-8", errors="ignore") as data:
            print('Reading %s ... \n' % filename)
            self.text = []#[BUFFER]*self.windowsize
            return data.readlines()

    def _writefile(self, fn, content):
        with open(fn, 'w', encoding='utf-8') as data:
            data.write(content)        

    def read_raw(self, filename):
        """ Open raw lemmatized input file with one text per line.
        Add buffer equal to window size before each text to prevent
        words from different texts being associated.

        2019-06-26: changed buffering from the end of the line
                    to the beginning to prevent rare crash that
                    occurred if the keyword was coincidentally
                    in the middle of the first symmetric window.
                    now linebreak is always the first symbol in text.

        """
        
        buffers = 1
        maxlen = [] # store line lengths
        lines = 0
        meta = None # no metadata available
        lacunae = 0
        for l in self._readfile(filename):
            linedata = l.split('\t')
            line = linedata[-1]
            
            if len(linedata) > 1:
                meta = linedata[0:-1]

            if meta is not None:
                self.metalist.append(self._abbreviate(meta))
                
            lemmas = line.strip('\n').strip().split(' ')

            """ Store documents for TF-IDF """
            self.documents.append(lemmas)

            lacunae += lemmas.count('_')
            maxlen.append(len(lemmas))
            self.text.extend([LINEBREAK] + [BUFFER] * self.windowsize + lemmas)
            buffers += 1
            lines += 1

        """ Add buffer to the end of the text """
        self.text.extend([BUFFER] * self.windowsize)
        
        self.corpus_size = len(self.text) - (buffers * self.windowsize) - lines
        self.word_freqs = Counter(self.text)

        """ Note that these statistics are sensitive to linebreaks,
        buffers etc. metasymbols, remember to subtract them from the
        results in case you add new meta symbols; e.g. you have to
        subtract 3 from unique lemmata because there are <BF>, <LB> and '_'
        """

        """ Save size of true useable corpus (place holders removed) """
        self.corpus_size_true = self.corpus_size - lacunae
        
        freqs = sorted([v for k, v in self.word_freqs.items()])
        print('-'*60)
        print('Corpus statistics:')
        print('%sLine count: %i' % (INDENT, lines))
        print('%sLongest line: %i words' % (INDENT, max(maxlen)))
        print('%sMedian line length: %i words' % (INDENT,
                            sorted(maxlen)[int(len(maxlen)/2)]))
        print('%sAverage line length: %i words' % (INDENT,
                            sum(maxlen)/len(maxlen)))
        print('%sWord count: %i' % (INDENT, self.corpus_size))
        print('%sUnique lemmata: %i' % (INDENT, len(self.word_freqs.keys()) -3))
        print('%sMedian word frequency: %i' % (INDENT,
                            freqs[int(len(freqs)/2)]))
        print('%sAverage word frequency: {0:.2f}'.format(sum(freqs)/len(freqs)) % INDENT)
        print('%sLacunae or underscores: %i' % (INDENT, lacunae))
        print('%sLemmas (not lacunae or underscores): %i' % (INDENT, self.corpus_size_true))
        print('-'*60 + '\n')
        
    def read_vrt(self, filename, lemmapos, pospos, delimiter='text'):
        #
        # FIX BUFFERING, ADD LINEBREAKS!
        #
        
        """ Open VRT file.

        ´lemmapos´ (int) indicates the word attribute index for lemmas.
        ´pospos´ (int) defines the word attribute index for POS-tags. If
        POS-tags are available has_postag() can be used to filter words
        by their POS-tags.

        ´delimiter´ splits the text by its ´sentence´, ´paragraph´
        or ´text´ element and disallows collocations being recognized if
        the delimiter is found between them. """
        delimiter = '</{}>'.format(re.sub('\W', '', delimiter))
        buffers = 1
        self.filename = filename
        with open(filename, 'r', encoding="utf-8") as data:
            print('parsing %s ...' % filename)
            self.text = [BUFFER] * self.windowsize 
            for line in data:
                l = line.strip('\n')
                if l == delimiter:
                    self.text.extend([BUFFER] * self.windowsize)
                    buffers += 1
                if not l.startswith('<'):
                    word_attrs = l.split('\t')
                    if len(word_attrs) > 3:
                        self.text.append(word_attrs[lemmapos])
                        if pospos is not None:
                            self.translations[word_attrs[lemmapos]] = word_attrs[pospos]
                else:
                    pass

        self.corpus_size = len(self.text) - (buffers * self.windowsize)
        self.word_freqs = Counter(self.text)

    def write_tsv(self, filename=None):
        """ Write output as .tsv """
        if filename is None:
            prefix = self.measure + '_%i_%s_%s_' % (self.windowsize,
                                                    self.freq_threshold,
                                                    self.output_format)
            fn = prefix + re.sub('\..+', '', self.filename)
        else:
            fn = filename
            
        print('Writing %s ...' % (fn))
        self._writefile(fn, '\n'.join(self.output))
        self._writefile(re.sub('\..+', '', fn) + '.log', self.__repr__())

    def import_json(self, filename):
        """ Load lookup table from JSON """
        print('Reading %s ...' % filename)
        with open(filename, encoding='utf-8') as data:
            return json.load(data)
        
    def export_json(self, filename):
        """ Save lookup table as JSON """
        print('Writing %s ...' % filename)
        with open(filename, 'w', encoding="utf-8") as data:
            json.dump(self.scored, data)

    def read_dictionary(self, filename):
        with open(filename, 'r', encoding="utf-8") as data:
            for line in data.read().splitlines():
                if line:
                    word, translation = line.split('\t')
                    ## STRIP ; from the end of line
                    self.translations[word] = translation.replace(';', ',')
            
    """ ================================================================
    Properties =========================================================
    ================================================================ """
            
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
        
    def set_window(self, size=None, symmetry=False):
        self.windowsize = size
        self.symmetry = symmetry

    """ ================================================================
    Helper funtions ====================================================
    ================================================================ """

    def _trim_float(self, number):
        if number == '':
            return number
        elif isinstance(number, int):
            return number
        else:
            return float('{0:.3f}'.format(number))

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
        Return True if in the list """
        if not self.regex_words[index]:
            return word in self.words[index]
        else:
            return self._match_regex([word], self.regex_words[index])\
                   or word in self.words[index]
        
    def _is_valid(self, w1, w2, freq):
        """ Validate bigram. Discard stopwords and those which
        do not match with the word of interest lists """
        if freq >= self.freq_threshold and self.word_freqs[w2] >= self.freq_threshold_collocate:    
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
    Stopword functions
    ================================================================ """

    """ tf_idf() returns a stopword list based on TF-IDF. Argument
    ´threshold´ defines the size of the stopword list. If no threshold
    is given, a list relative to corpus size is returned.

    This list can be passed as ´stopwords´ argument.
    """
    
    def tf_idf(self, threshold=0):
        print('Making TF-IDF stopword list...)')
        tf_idfs = {}
        w = []
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
            w.append([sum(scores), word])

        return [x[1] for x in sorted(w, reverse=True)[0:threshold]]

            
    """ ================================================================
    Bigram counting

    *_symmetric()         Use symmetrical window
    *_symmetric_dist()    Keep track of distances

    Uses separate functions to avoid complex conditional statements.
    (there are already too many conditionals)
    
    ================================================================ """
    
    def score_bigrams(self, measure):
        """ Score bigrams by using class ´measure´ """
        print('Counting bigrams ...')
        
        self.measure = measure.__name__
        
        """ Set has_meta if metadata is available. """
        has_meta = len(self.metalist) > 0

        if HIDE_MIN_SCORE:
            self.minimum = ''
        else:
            self.minimum = measure.minimum
        
        if not self.text:
            print('Input text not loaded.')
            sys.exit()
        
        def _check_formulaic(bigram, window, index):
            """ Store windows only if formulaic_measures are used,
            otherwise skip this to save memory and time. Remove
            index of the collocate from the window to preserve only
            context of the bigram """
            if self.formulaic_measure is not None:
                #window.pop(index)
                window[index] = '_'
                self.WINS.setdefault(bigram, []).append(window)
            return bigram

        def _gather_meta(bigram, meta):
            """ Gather and count metadata for each bigram:
            force to increase performance """
            try:
                self.metadata[bigram][meta] += 1
            except:
                try:
                    self.metadata[bigram].update({meta: 1})
                except:
                    self.metadata[bigram] = {meta: 1}
                finally:
                    pass
            finally:
                pass
        
        def scale(bf, distance):
            """ Scale bigram frequency with window size. Makes the
            scores comparable with NLTK/Collocations PMI measure """
            if WINDOW_SCALING:
                if self.symmetry:
                    return bf / (self.windowsize - 1)
                else:
                    return bf / (self.windowsize - 1)
            else:
                return bf
        
        def count_bigrams_symmetric():
            """ Symmetric window """
            wz = self.windowsize - 1                
            for w in zip(*[self.text[i:] for i in range(1+wz*2)]):
                if w[0] == LINEBREAK:
                    if has_meta:
                        meta = tuple(self.metalist.pop(0))
                if self._is_wordofinterest(w[wz], 1) and \
                   self._has_condition(w[0:wz]+w[wz+1:]):
                    for index, bigram in enumerate(itertools.product([w[wz]],w[0:wz]+w[wz+1:])):
                        if has_meta:
                            _gather_meta(bigram, meta)
                        yield _check_formulaic(bigram, list(w[0:wz]+w[wz+1:]), index)

        def count_bigrams_symmetric_dist():
            """ Symmetric window and distance tracking. """

            def chain(w1, w2):
                """ Make a zip/convolution chain of two lists.
                [a, b], [c, d] -> [a, c, b, d] """
                chain = [' '] * len(w1+w2)
                chain[::2] = w1
                chain[1::2] = w2
                return chain

            wz = self.windowsize - 1
            for w in zip(*[self.text[i:] for i in range(1+wz*2)]):
                left = list(w[0:wz])
                right = list(w[wz+1:])
                if w[0] == LINEBREAK:
                    if has_meta:
                        meta = tuple(self.metalist.pop(0))
                if self._is_wordofinterest(w[wz], 1) and \
                   self._has_condition(left+right):
                    for index, bigram in enumerate(itertools.product([w[wz]], left+right)):
                        bigram = _check_formulaic(bigram, left+right, index)
                        context = chain(left[::-1], right)
                        min_dist = math.floor(context.index(bigram[1])/2) + 1
                        if has_meta:
                            _gather_meta(bigram, meta)
                        self.distances.setdefault(bigram, []).append(min_dist)
                        yield bigram

        def count_bigrams_forward():
            """ Calculate bigrams within each forward-looking window """
            for w in zip(*[self.text[i:] for i in range(self.windowsize)]):
                if w[0] == LINEBREAK:
                    """ Keep track of lines and their metadata """
                    if has_meta:
                        meta = tuple(self.metalist.pop(0))
                if w[0] in self.words[1] and self._has_condition(w[1:]):
                    for index, bigram in enumerate(itertools.product([w[0]], w[1:])):
                        if has_meta:
                            """ If metadata is available, store it """
                            _gather_meta(bigram, meta)
                        yield _check_formulaic(bigram, list(w[1:]), index)

        def count_bigrams_forward_dist():
            """ Calculate bigrams within each forward-looking window,
            calculate also average distance between words. Distance
            tracking is not included into count_bigrams_forward()
            for better efficiency """
            for w in zip(*[self.text[i:] for i in range(self.windowsize)]):
                if w[0] == LINEBREAK:
                    if has_meta:
                        meta = tuple(self.metalist.pop(0))
                if w[0] in self.words[1] and self._has_condition(w[1:]):
                    for index, bigram in enumerate(itertools.product([w[0]], w[1:])):
                        bg = _check_formulaic(bigram, list(w[1:]), index)
                        if has_meta:
                            _gather_meta(bg, meta)
                        d = index + 1
                        self.distances.setdefault(bg, []).append(d)
                        yield bg

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

        """ Make dictionary for JSON """
        print('Calculating scores ...')
        w1list, w2list = [], []
        F_MEASURE = self.formulaic_measure
        for bigram in bigram_freqs.keys():
            w1, w2 = bigram[0], bigram[1]
            if self._is_valid(w1, w2, bigram_freqs[bigram]):
                if self.formulaic_measure is not None:
                    formulaic_measure = F_MEASURE.score(self.WINS[bigram],
                                                        self.formulaic_forced,
                                                        bigram[-1])
                else:
                    formulaic_measure = 0
                fm = max(1-formulaic_measure, 00000.1)
                distance = self._get_distance(bigram)
                freq_w1 = self.word_freqs[w1]
                freq_w2 = self.word_freqs[w2]               
                score = measure.score(scale(bigram_freqs[bigram], distance),
                                      freq_w1, freq_w2, self.corpus_size)
                #score = (_score[0] / _score[1]) * fm
                #score = _log(score)
                data = {'score': score * fm,
                        'distance': distance,
                        'frequency': bigram_freqs[bigram],
                        'similarity': formulaic_measure}
                self.scored['translations'][w1] = self._get_translation(w1)
                self.scored['translations'][w2] = self._get_translation(w2)
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
        #print(self.metadata[('zakāru', DEBUG)])
        #print(len(self.metadata[('zakāru', DEBUG)]))
        #print(bigram_freqs[('zakāru', DEBUG)])
        #print(Counter(X))
        
    """ ================================================================
    Score table processing =============================================
    ================================================================ """

    def _stringify(self, array):
        """ Convert all values in table into strings and
        localize decimal markers. """
        def _format_decimal(item):
            if isinstance(item, float):
                return str(item).replace('.', DECIMAL)
            else:
                return str(item)
        return [_format_decimal(x) for x in array]

    def _sort_by_index(self, table, indices):
        """ Sort table by given two indices, i.e. [0, 4] sorts
        the table by 1st and 5th values """
        i = indices[0]
        j = indices[-1]
        return sorted(table, key=lambda item: (item[i], -item[j]))
        
    def _filter_json(self, words, index):
        """ Validate and return words of interest """
        return [w for w in sorted(words) if self._is_wordofinterest(w, index)\
                and not self._meets_anycondition('stopwords', [w])]

    def _check_table(self, table):
        """ Check if table is imported from JSON """
        if table is None:
            from_json = True
            table = self.scored
            words1 = sorted(table['words1'])
            words2 = sorted(table['words2'])
        else:
            from_json = True
            words1 = self._filter_json(table['words1'], 1)
            words2 = self._filter_json(table['words2'], 2)
        return table, from_json, words1, words2

    def print_matrix(self, value, table=None):
        """ Make a collocation matrix of two sets of words of
        interest. Argument ´value´ must be ´score´, ´frequency´
        or ´distance´. """
        
        """ Use self.scored if imported JSON is not given.
        Apply word filters if JSON is loaded """

        print('Generating {} matrix ...'.format(value))
        self.output_format = 'matrix'
        table, from_json, words1, words2 = self._check_table(table)
        rows = [[value.upper() + ' MATRIX W1 ->']\
                + ['{}'.format(w + ' ' + table['translations'][w])\
                   for w in words2]]

        for w1 in words1:
            row = []
            for w2 in words2:
                score = self.minimum
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
            self.output.append('\t'.join([str(m) for m in r]))

    def print_scores(self, limit=10000, table=None, gephi=False):
        def _add_prefix(key):
            ''' Add Mylly-prefixes '''
            if not MYLLY:
                return key
            else:
                if 'freq' in key:
                    return 'cM' + key
                if 'score' in key or 'distance' in key:
                    return 'wM' + key
                else:
                    return key

        def _get_diff(bigram):
            diff = []
            for word in zip(*self.WINS[bigram][::-1]):
                diff.append(len(set(word)) / len(word))
            return sum(diff) / len(diff)

        def _beautify_meta(bigram):
            """ Pretty print metadata; calculate distribution for
            each keyword : collocate pair (1) period-wise,
            (2) genre-wise and (3) period-genre-wise """
            
            def _sort(d):
                """ Sort frequency data by frequency """
                return sorted([[d[x], x] for x in d.keys()], reverse=True) 

            def _join(key):
                """ Join tuple keys """
                if isinstance(key, tuple):
                    return ' '.join(key)
                else:
                    return key
            
            metas = [] 
            formatted = {'period': {}, 'genre' : {}, 'combo' : {}}

            if not self.metadata:
                return ['', '', '']
            else:
                for k, v in self.metadata[bigram].items():
                    m = {'period': k[0], 'genre': k[1], 'combo': k}
                    for metatype, val in m.items():                        
                        if val not in formatted[metatype].keys():
                            formatted[metatype][val] = v
                        else:
                            formatted[metatype][val] += v
                            
                for key in sorted(m.keys()):
                    metas.append(', '.join(['%s (%s)' % (_join(x[1]), x[0])\
                                    for x in _sort(formatted[key])]))
                return metas

        
        print('Generating score table ...')
        #table, from_json, words1, words2 = self._check_table(table)
        if table is None:
            table = self.scored
        else:
            pass
        
        """ Define score table header """
        if MYLLY:
            header = ['word1', 'attr1', 'word2', 'attr2',
                      'word1 freq', 'word2 freq',
                      'bigram freq', 'score trimmed', 'score']
            sort_indices = [0, -1]
        else:
            """ For Oracc """
            header = ['word1', 'attr1', 'word2', 'attr2',
                      'word1 freq', 'word2 freq', 'bigram freq',
                      'score trimmed', 'distance', 'similarity',
                      'period', 'genre', 'combined', 'url']
            
            sort_indices = [0, -7]

        self.output_format = 'scores'
        if gephi:
            self.output.append(';'.join(['source', 'target', 'weight']))
        else:
            self.output.append('\t'.join([_add_prefix(x) for x in header]))
            
        rows = []
        for w1 in table['collocations'].keys():
            for w2 in table['collocations'][w1].keys():
                combined, genre, period = _beautify_meta((w1, w2))
                bigram = table['collocations'][w1][w2]
                freqs = table['freqs']
                items = {'word1': w1,
                         'word2': w2,
                         'attr1': table['translations'][w1],
                         'attr2': table['translations'][w2],
                         'word1 freq': freqs[w1],
                         'word2 freq': freqs[w2],
                         'bigram freq': bigram['frequency'],
                         'score trimmed':
                             float(self._trim_float(bigram['score'])),
                         'score': '{0:.16f}'.format(bigram['score']),#float(bigram['score']),
                         'score2': float('{0:.16f}'.format(bigram['score'])),#float(bigram['score']),
                         'distance': bigram['distance'],
                         'similarity': self._trim_float(bigram['similarity']),
                         'period': period,
                         'genre': genre,
                         'combined': combined,
                         'url': _make_korp_oracc_url(w1, w2, self.windowsize-2)}
                rows.append([items[key] for key in header])
                
        lastword = ''
        for line in self._sort_by_index(rows, sort_indices):
            if not gephi:
                if lastword != line[0]:
                    i = 0
                if i < limit:
                    self.output.append('\t'.join(self._stringify(line)))
                lastword = line[0]
                i += 1
            if gephi:
                if lastword != line[0]:
                    i = 0
                if i < limit:
                    data = [line[0]+' '+line[1]] + [line[2]+' '+line[3]] + [line[7]]
                    self.output.append(';'.join(self._stringify(data)))
                lastword = line[0]
                i += 1

    """ ================================================================
    Dictionary tools ===================================================
    ================================================================ """

    def get_translation(self, lemma):
        if lemma in self.translations.keys():
            return self.translations[lemma]
        else:
            return '[?]'
        
    def _search_dict(self, translations):
        """ General method for seaching the dictionary by
        given translations """
        if not self.text:
            print('text not loaded')
            sys.exit()
            
        freqlist = []
        def _get_freqs(k, v):
            if k in self.word_freqs.keys():
                freqlist.append([k, '[%s]' % v, self.word_freqs[k]])

        for k, v in self.translations.items():
            for t in translations:
                if isinstance(t, str):
                    if t == v:
                        _get_freqs(k, v)
                else:
                    if re.match(t, v):
                        _get_freqs(k, v)
        return freqlist

    def get_freqs_by_lemma(self, lemmas):
        """ Count lemma frequencies in corpus. Print absolute and per million
        freqs """
        freqlist = []
        for lemma in lemmas:
            if lemma in self.word_freqs.keys():
                freqlist.append([self.word_freqs[lemma],
                        1000000*(self.word_freqs[lemma]/self.corpus_size),
                        lemma,
                        self.get_translation(lemma)])
            else:
                freqlist.append([0, 0, lemma, self.get_translation(lemma)])

        print('\n'.join([str(x[0]) + '\t' \
            + str(self._trim_float(x[1])).replace('.', ',') + '\t' + '\t'.join(x[2:])\
                         for x in sorted(freqlist, reverse=True)]))

    def get_freqs_by_translation(self, translations, sort_by=0):
        """ Search words and their frequencies by their translation
        from the loaded text. """          
        freqlist = self._search_dict(translations)     
        for item in self._sort_by_index(freqlist, [sort_by]):
            print('\t'.join(self._stringify(item)))

    def has_translation(self, translations):
        """ Return all words that have the given translation """
        wordlist = self._search_dict(translations)
        return [word[0] for word in wordlist]

    def has_postag(self, postag):
        return self.has_translation(postag)

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




