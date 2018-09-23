# Collocations
Calculates different word association measures. The window splitting and n-gram finding algorithms are quite well optimized now. Designed specially for cuneiform languages (but not restricted to them). Takes input as a lemmatized text file with lacunae marked as '_'.

Example:

_ tuku _ palāhu _ sa dug _ kašādu _ nam kud ilu rabû arāru e gi
monkey eat coconut and the sun be shine

Features:
- Symmetric and asymmetric windows
- Word distance tracking
- Regex support for word/stopword filters
- High performance algorithms for score calculation and n-gram finding
- Measures for detecting formulaic expressions and repetitiveness
