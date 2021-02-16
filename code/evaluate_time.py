
import os
import re
import copy
import time
import pickle
import numpy as np
import pandas as pd

# from spmf import Spmf
from functools import reduce
from itertools import groupby
from nltk.corpus import stopwords
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from TRASE_v1 import *
from pygapbide import *


'''
        =================================================================
                                Main Program
        =================================================================
'''

threshold = 0.2
min_support = 0.2
min_method = 5
max_gap = 3

time_record = []

# Test time on different number of sequences
folder = 'components/synthetic/n_seq'
for rep in os.listdir(folder):
    if rep.startswith('.'):
        continue

    rep = int(rep)
    current_folder = '%s/%d' % (folder, rep)
    traces = sorted(os.listdir(current_folder))

    data = []
    trace_idx = 0
    unique_components = set()
    for trace in traces:
        if trace.startswith('.'):
            continue

        components = pickle.load(open('%s/%s' % (current_folder, trace), "rb"))
        # print('Number of Raw Preliminary Phrases: %d' % len(components))

        # Remove consecutive duplicate items
        components = [i[0] for i in groupby(components)]
        # print('Number of Cleaned Preliminary Phrases: %d' % len(components))

        for component in components:
            unique_components.add(frozenset(component))
        data.append(components)
    unique_components = [set(x) for x in unique_components]

    # Convert the trace only preserve the first occuring phase
    sequence_db = []
    for trace in data:
        sequence = []
        for i in range(len(trace)):
            if trace[i] not in sequence:
                sequence.append(trace[i])
        sequence_db.append(sequence)

    '''
        =================================================================
                                TRASE Algorithm Begin
        =================================================================
    '''

    # starting time
    start = time.time()

    # Build ID List
    id_list = IdList()
    id_list.build_list(sequence_db, max_gap, min_support)

    # Find Closed Sequential Pattern
    Z = []  # Maximum Sequential Pattern

    # Generate and sort search_space by support
    search_space = np.argsort(list(id_list.phase_support.values()))[::-1]

    # Ignore Ids less than min_support support
    search_space = list(search_space[:np.sum(np.array(list(id_list.phase_support.values())) >= min_support)])

    while len(search_space) > 0:
        que_idx = search_space.pop(0)
        patterns = id_list.extend_pattern(que_idx, Z)

        # Check if pattern satisfy minimum number of methods
        for i in range(len(patterns) - 1, -1, -1):
            no_of_methods = sum([len(id_list.ids[x]) for x in patterns[i].pattern])
            if no_of_methods < min_method:
                del patterns[i]

        # Keep closed pattern only
        for i in range(len(patterns) - 1, -1, -1):
            for j in range(len(patterns)):
                if i == j:
                    continue
                # p[i] and p[j] have the same support and p[i] is a subsequence of p[j]
                if (patterns[i].support == patterns[j].support) & (is_subsequence(patterns[i].pattern, patterns[j].pattern)):
                    # print('p[%d] is a subsequence of p[%d]: (%.2f: %s) and (%.2f: %s)' % (
                    #     i, j, patterns[i].support, patterns[i].pattern, patterns[j].support, patterns[j].pattern))
                    del patterns[i]
                    break

        # Pattern is Valid and added to Z
        Z += patterns

    # end time
    end = time.time()
    TRASE_time = end - start

    print('\nRuntime of TRASE is %.2fs' % TRASE_time)
    time_record.append(('TRASE', 'REP', '%.3f' % TRASE_time))


    '''
        =================================================================
                            Gap-Bide Algorithm Begin
        =================================================================
    '''

    # Represent Data by ID-List IDs
    temp_data = []
    for trace in sequence_db:
        temp_trace = []
        for event in trace:
            temp_trace.append(id_list.ids.index(event))
        temp_data.append(temp_trace)

    start = time.time()

    # All Sequential Patterns
    # spmf = Spmf("SPAM", input_filename="%s.txt" % app, output_filename="output.txt", arguments=[0.6, 5, 500, max_gap, False])
    # Closed Sequential Patterns
    sdb = []
    for trace in temp_data:
        s = []
        for i in range(len(trace)):
            s.append(trace[i])
        sdb.append(s)
    gb = Gapbide(sdb, int(min_support * id_list.n_traces), 0, max_gap-1)
    temp = gb.run()
    # Maximal Sequential Patterns
    # spmf = Spmf("VMSP", input_filename="%s.txt" % app, output_filename="output.txt", arguments=['%d%%' % (min_support * 100), 500, max_gap, False])
    # spmf.run()
    # print(spmf.to_pandas_dataframe(pickle=True))

    end = time.time()
    GB_time = end - start

    print('\nRuntime of Gap-Bide is %.2fs' % GB_time)
    time_record.append(('GAP-BIDE', 'REP', '%.3f' % GB_time))

    '''
        =================================================================
                            Gap-Bide Algorithm Begin
        =================================================================
    '''


# Print Pattern Result
for p in patterns:
    print('%.2f\t%s' % (p[2], p[0]))
    print('Positions:')
    for i in range(len(p[1])):
        print('  Trace %d: %s' % (i, p[1][i]))

import random
sorted(random.sample(range(1,50), 6))
