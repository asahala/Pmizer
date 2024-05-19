import re
from pmizer2 import Text, Associations, Lazy, PMIDELTA


""" ==============================================================

                                          Aleksi Sahala 2019-2024
                                               github.com/asahala

This is the part of the script that can be safely modified.

:: DEFINING KEYWORDS:

  words1=['sisû[horse]N', 'parû[mule]N']

         Would find all collocates for sisû and parû. You can
         also use regular expressions but they must be
         precompiled, for example.

   words1=[re.compile('banû.+')]

         Will find collocates for banû regarldess of their sense
         or part-of-speech. If you want to only find banûs that
         are adjectives, you can do

   words1=[re.compile('banû.+AJ')]


:: DEFINING CLOSED COLLOCATE SETS

         You can limit your possible collocates by defining words2.
         For example, if you only want to find collocates that
         are divine names, use

   words2=[re.compile('.*DN.*')]


:: DEFINING STOPWORDS

         Stopwords can be used to filter out groups of words
         from your collocate lists. For example, if you want
         to filter out all proper names and royal names, you
         can use the regular expression [RD]N, that is (RN|DN).

    stopwords=[re.compile('.*([RD]N).*')]

         In general its faster to define stopwords in inverse
         by using words2, e.g.

    words2=[re.compile('.*\](N|AJ|V).*')]

         disallows ALL other collocates than nouns, adjectives
         and verbs. 

:: DEFINING CONDITIONS (ADVANCED FEATURE)

         To define conditions for your collocates, you can
         use arguments `conditions` and `positive_condition`.
         For example, the following arguments

    words1=['kakku[weapon]N'],
    conditions=[re.compile('.*naparšudu.*')],
    positive_condition=False,

         will calculate collocates for kakku[weapon]N ONLY
         if the word naparšudu does not exist within the window.
         On the contrary, the following would find collocates
         for kakku[weapon]N only if naparšudu exists in the
         window.

    words1=['kakku[weapon]N'],
    conditions=[re.compile('.*naparšudu.*')],
    positive_condition=True,


:: AUTO-GENERATING STOPWORD LISTS (ADVANCED FEATURE)
   
    stopwords=z.tf_idf(threshold=1000)

          Will generate a list of 1000 most uninteresting words
          in the text by using TF-IDF. Note that this is a
          part of the Text class.           

============================================================== """

z = Text('dataset.txt') # Dataset in TPL format
wz = 5                  # Window size
x = Associations(z,
                 words1=['kakku[weapon]N'],      # keyword of interest
                 #conditions=[re.compile('.*naparšudu.*')],
                 #positive_condition=False,
                 #stopwords=[re.compile('.*([RD]N).*')],
                 formulaic_measure=Lazy,   # use CSW
                 minfreq_b = 2,            # min collocate freq
                 minfreq_ab = 2,           # min bigram freq
                 symmetry=True,            # use symmetric window
                 windowsize=wz,            # window size 
                 factorpower=3)            # CSW k-value

A = x.score(PMIDELTA)              # Select measure (e.g, PMI, PMI2, Jaccard...)

# Save results
x.print_scores(A, limit=20, gephi=False, filename='oracc.tsv')
