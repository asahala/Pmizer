import os
import csv

""" ALP course CSV --> TPL converter
github.com/asahala """

POS_FILTER = {'V', 'AJ', 'N', 'AV'}
DATA_PATHS = ('data/rinap01', 'data/rinap05')

def read_csv(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for e, line in enumerate(csv.reader(f, delimiter=',')):
            if e == 0:
                keys = {a: b for a, b in zip(line, range(0, len(line)))}
                continue

            pos = line[keys['pos']]
            sense = line[keys['sense']].replace(' ', '_')
            cf = line[keys['cf']]

            if len(pos) == 2 and pos.endswith('N') or pos in POS_FILTER:
                yield f'{cf}[{sense}]{pos}'
            else:
                yield '_'

""" Batch process all CSV files in DATA_PATHS and write to dataset.txt """
lemmalist = set()
with open('dataset_nostops.txt', 'w', encoding='utf-8') as o:
    for path in DATA_PATHS:
        for file in (x for x in os.listdir(path) if x.endswith('.csv')):
            print(f'processing {file}...')
            for word in read_csv(os.path.join(path, file)):
                o.write(word + ' ')
                lemmalist.add(word)
            o.write('\n')

with open('lemmalist.txt', 'w', encoding='utf-8') as o:
    for lemma in sorted(lemmalist):
        o.write(lemma + '\n')

