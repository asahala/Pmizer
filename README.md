# pmizer.py

A tool for calculating word association measures. 

      _____   __    __   __   _____    ______   ______
     |     \ |  \  /  | |  | |___  |  |  ____| |      \
     |     | |   \/   | |  |    /  /  |  |___  |       |
     |  ___/ |        | |  |   /  /   |  ____| |      /
     | |     |  |\/|  | |  |  /  /__  |  |____ |  |\  \
     |_|     |__|  |__| |__| |______| |______| |__| \__\
     
       2019-12-31
     
     
     ========================================================================
     ************************************************************************
                   G E N E R A L   D E S C R I P T I O N 
     ========================================================================
     ************************************************************************
     
     This script calculates different word association measures derived from
     PMI (Pointwise Mutual Information). In Language Technology, the PMI is
     used to find collocations and associations between words.
     
     By its basic definition, PMI is the ratio of p(w1,w2), i.e. the actual
     co-occurrence probability of two words within a certain distance
     from each other to p(w1)p(w2), i.e. the expected chance of those words
     co-occurring independently.
     
                        p(w1,w2)          
       PMI(w1,w2) = ln ----------         
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
     
         WEAPON is something that is
           - LIFTED to intimidate and to show power
           - associated with FURY, ANGER and STRENGTH
           - used in BATTLES to DEFEAT ENEMIES and their CITIES between
             the SEAS in order to enforce the Assyrian KINGSHIP.
           - great weapons are granted by the GOD ASHUR
     
     
     
     ========================================================================
     ************************************************************************
                       H O W   T O   U S E   P M I Z E R 
     ========================================================================
     ************************************************************************
     
     
     The following lines of code will generate the top-20 NPMI results using
     a forward-looking window of 7, with a minimum bigram frequency of 10.
     
     1) Initialize Associations()        a = Associations()
     2) Set window properties:           a.window(size=7)
     3) Open your corpus file            a.read_raw('input.txt')
     4) Open dictionary file (OPTIONAL)  a.read_dictionary('dictionary.txt')
     5) Set constraints (optional)       a.set_constraints(freq_threshold=10)
     6) Calculate scores                 a.score_bigrams(NPMI)
     7) Make score table for the top-20  a.print_scores(20)
     8) Save scores                      a.write_tsv('output.tsv')
     
     See below for more detailed instructions.
     
     
     ========================================================================
     Associations.set_window(size, symmetry) ================================
     ========================================================================
     
      ´size´             (int) Collocational window that defines the
                         maximum mutual distance of the elements of a non-
                         contiguous bigram. Minimum distance is 2, which
                         means that the words are expected to be next to each
                         other.
     
                         SPECIAL CASE: Variable lenght windows
                         Sometimes it may be useful to use variable length windows, 
                         e.g. by using one line of cuneiform text as the window size.
                         This can be done by pre-processing the input text 
                         line-by-line and using a large window size, e.g. 40 that
                         spans over most of the standard-length lines. Using very
                         large windows (>100) may cause memory issues and crashes.
     
      ´symmetry´         (bool) Use symmetric window. If not used, the window
                         is forward-looking. For example, with a window size
                         of 3 and w4 being our word of interest, the windows
                         are calculated as:
     
                                      w1 w2 w3 w4 w5 w6 w7
                         symmetric       +--+--^--+--+
                         asymmetric            ^--+--+
     
     ========================================================================
     Associations.read_raw(filename) ========================================
     ========================================================================
     
      ´filename´         (str) name of the input file. Bigrams are NOT allowed 
                         to span from line to another. Thus, if you want to 
                         disallow bigrams spanning from sentence to another, the
                         text should contain one sentence per line.
     
                         Input file must be lemmatized and lemmas must be
                         separated from each other by using spaces, e.g.
     
                         "dogs eat bones" -->
                         dog eat bone
     
     NOTE: Window size must always be specified before reading the file
     for setting correct buffers (or paddings).
     
     ========================================================================
     Associations.read_dictionary(filename) =================================
     ========================================================================
     
      ´filename´         (str) Dictionary filename. This file should contain
                         all words in your corpus and their translations in
                         other language separated by tab. Using dictionaries
                         may help if you're working with a foreign language.
                         The translations will be shown for each word in the 
                         results. 
     
                         Using dictionaries is optional.    
     
     ========================================================================
     Associations.set_constraints(**kwargs) =================================
     ========================================================================
     
     Constraints and properties may be set by using the following kwargs:
     
      ´freq_threshold´   (int) minimum allowed bigram frequency. This can be
                         used to counter the low-frequency bias of certain
                         PMI variants. In other words, this defines how many
                         times to words must co-occur within the window to
                         be accepted.
                         
      ´freq_threshold_collocate´   (int) minimum allowed collocate frequency.
                         That is, how frequent word must be to be accepted.
                         It may be useful to discard words that occur only
                         once or twice in the corpus if bigram frequency
                         threshold is set very low.
     
      ´words1´ &         (list) words of interest AKA keywords, the white 
      ´words2´           list of words, which are allowed to exist in the
                         bigram (word1, word2) or (word1 ... word2).
                         Words of interest may also be expressed as compiled
                         regular expressions. See exampes in ´stopwords´.
     
                         NOTE: at least one words1 must be specified!
     
      ´stopwords´        (list) the black list of uninteresting words like
                         prepositions, numbers etc. May be expressed as
                         compiled regular expressions. For example
                         [re.compile('\d+?'), re.compile('^[A-ZŠṢṬĀĒĪŪ].+')]
                         will discard all numbers written as digits, as well
                         as all words that begin with a capital letter.
     
                         NOTE: It may be wise to express series of regular
                         expressions as disjunctions (regex1|...|regexn) as
                         it makes the matching significantly faster, e.g. 
                         [re.compile('^(\d+?|[A-ZŠṢṬĀĒĪŪ].+)'].
     
                         Stopwords and keywords may also be defined
                         by their translations, POS tag etc. See the section
                         ´Using dictionaries´ for more info.
     
      ´conditions´       (list) optional list of conditional words. 
                         If any of them matches, the condition will be True.
                         See below for setting polarity of the condition.
     
      ´positive_condition´ (bool) set True if conditional word must appear
                         in the window, or False if it is prohibited. E.g.
                         search collocates for ´queen´ ONLY of ´king´ is
                         mentioned (or not mentioned) in the same context.
      
      ´track_distance´   (bool) calculate and store the average minimum
                         distance between the words of the bigram. If the
                         same bigram can be found several times within the
                         window, only the closest distance is taken into
                         account.
     
                         This information is shown in the results as a
                         separate metric.
     
                         NOTE: Slows bigram counting 2-3 times depending
                         on the window size. With large symmetric windows
                         this can take several minutes or even hours.
     
      ´formulaic_measure´ Gives collocate an additional score depending on
                         how formulaic the context is. Possible choices are
                         (do not use " or ' with these):
     
                         None     (default) No measure
     
                         Greedy   Percentage of similar information
                         Lazy     Percentage of repeated information
                         Strict   formulaic : free (ratio)
     
                         See more precise documentation in the comments about
                         repetitiveness measures. Note that these measures
                         are generally useful only with window sizes of
                         3 or larger.
     
     
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
         NPMI2           Normalized Positive PMI^2 (Sahala 2019)
         PMISIG          TODO: add doc
         SCISIG          TODO: add doc
     
     
     ========================================================================
     Output formats =========================================================
     ========================================================================
     
     Associations.print_scores(limit) ---------------------------------------
     
      DESCRIPTION       Generates a score table that includes word and bigram
                        frequencies and the association measures.
     
      ´limit´           (int) Set the minimum rank that will be printed in
                        your score table. E.g. a limit of 20 will print only
                        the top-20 scores.
                        
     Associations.print_matrix(value) ---------------------------------------
     
      DESCRIPTION       Generates a matrix from two sets of words of
                        interest. NOTE: matrices should only be used with a
                        small pre-defined set of words of interest to avoid
                        huge unreadable outputs. For playing around with 
                        big matrices see section Word Embeddings.
     
      ´value´           Value that will be used in the matrix: ´score´,
                        bigram ´frequency´ or ´distance´.
     
     
     ========================================================================
     ************************************************************************
                   G E N E R A T I N G    W O R D L I S T S 
     ========================================================================
     ************************************************************************
     
     Argument ´stopwords´ in set_constraints() can be generated automatically
     by using function tf_idf(threshold), e.g.
     
     stopwords=a.tf_idf(threshold=50)
     
     Will automatically generate the 50 most uninteresting words in the input
     data by using term frequency - inverse document frequency metric.
     
     

# PMI2VEC/pmi2vec.py

Builds word vectors (embeddings) based on PMI scores by using singular value decomposition. Requires numpy and sklearn. 

Todo:
- Binary format
- Merge with pmizer
