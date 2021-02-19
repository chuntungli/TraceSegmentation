
import os
import time
import pickle
import numpy as np
import pandas as pd

# from spmf import Spmf
from itertools import groupby

from TRASE_v1 import *
from pygapbide import *


def build_seqDB(folder_path):
    traces = sorted(os.listdir(folder_path))

    data = []
    trace_idx = 0
    unique_components = set()
    for trace in traces:
        if trace.startswith('.'):
            continue

        components = pickle.load(open('%s/%s' % (folder_path, trace), "rb"))
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

    return sequence_db

# Convert traces to list of phase ids
def convertTrace(id_list):
    # Convert the trace to other format
    sdb_phaseIDs = []
    for trace in sequence_db:
        s = []
        for event in trace:
            s.append(id_list.ids.index(event))
        sdb_phaseIDs.append(s)

    # Write data to file
    # f = open("%s.txt" % app, "w+")
    # for trace in temp_data:
    #     for i in range(len(trace)):
    #         f.write('%d -1 ' % trace[i])
    #     f.write('-2\r\n')
    # f.close()

    return sdb_phaseIDs

'''
        =================================================================
                        Evaluate Different Length of Patterns
        =================================================================
'''

time_record = []
min_sup = 0.2
min_size = 1
max_gap = 3

# Test time on different number of sequences
folder = 'components/synthetic/pat_len'
for pat_len in os.listdir(folder):
    if not pat_len.isdigit():
        continue

    pat_len = int(pat_len)
    folds = sorted(os.listdir('%s/%d' % (folder, pat_len)))

    for fold in folds:
        fold = int(fold)

        sequence_db = build_seqDB('%s/%d/%d' % (folder, pat_len, fold))

        '''
            =================================================================
                                    TRASE Algorithm
            =================================================================
        '''

        start = time.time()
        # seq_db, min_sup, min_size, max_gap
        id_list, patterns = TRASE(sequence_db, min_sup, min_size, max_gap)
        TRASE_time = time.time() - start

        print('\nRuntime of TRASE: %.2fs\tNo. of Patterns: %d' % (TRASE_time, len(patterns)))
        time_record.append(('TRASE', pat_len, fold, '%.3f' % TRASE_time))


        '''
            =================================================================
                                Gap-Bide Algorithm
            =================================================================
        '''

        # Convert the trace to other format
        sdb = convertTrace(id_list)

        start = time.time()

        # All Sequential Patterns
        # spmf = Spmf("SPAM", input_filename="%s.txt" % app, output_filename="output.txt", arguments=[0.6, 5, 500, max_gap, False])

        # Closed Sequential Patterns
        gb = Gapbide(sdb, int(min_sup * id_list.n_traces), 0, max_gap-1)
        patterns = gb.run()

        # Maximal Sequential Patterns
        # spmf = Spmf("VMSP", input_filename="%s.txt" % app, output_filename="output.txt", arguments=['%d%%' % (min_support * 100), 500, max_gap, False])
        # spmf.run()
        # print(spmf.to_pandas_dataframe(pickle=True))

        GB_time = time.time() - start

        print('Runtime of Gap-Bide: %.2fs\tNo. of Patterns: %d' % (GB_time, len(patterns)))
        time_record.append(('GAP-BIDE', pat_len, fold, '%.3f' % GB_time))

time_df = pd.DataFrame(time_record, columns=('method', 'pat_len', 'fold', 'time'))
time_df = time_df.astype({'time': 'double'})
time_df.to_csv('%s/result.csv' % folder, index=False)
print(time_df.groupby(by=['method', 'pat_len']).agg({'time': ['mean', 'std']}))



'''
        =================================================================
                        Evaluate Different No. of Patterns
        =================================================================
'''

time_record = []
min_sup = 0.2
min_size = 1
max_gap = 1

# Test time on different number of sequences
folder = 'components/synthetic/seq_len'
for seq_len in os.listdir(folder):
    if not seq_len.isdigit():
        continue

    seq_len = int(seq_len)
    folds = sorted(os.listdir('%s/%d' % (folder, seq_len)))

    for fold in folds:
        fold = int(fold)

        sequence_db = build_seqDB('%s/%d/%d' % (folder, seq_len, fold))

        '''
            =================================================================
                                    TRASE Algorithm
            =================================================================
        '''

        start = time.time()
        # seq_db, min_sup, min_size, max_gap
        id_list, patterns = TRASE(sequence_db, min_sup, min_size, max_gap)
        TRASE_time = time.time() - start

        print('\nRuntime of TRASE: %.2fs\tNo. of Patterns: %d' % (TRASE_time, len(patterns)))
        time_record.append(('TRASE', seq_len, fold, '%.3f' % TRASE_time))


        '''
            =================================================================
                                Gap-Bide Algorithm
            =================================================================
        '''

        # Convert the trace to other format
        sdb = convertTrace(id_list)

        start = time.time()

        # All Sequential Patterns
        # spmf = Spmf("SPAM", input_filename="%s.txt" % app, output_filename="output.txt", arguments=[0.6, 5, 500, max_gap, False])

        # Closed Sequential Patterns
        gb = Gapbide(sdb, int(min_sup * id_list.n_traces), 0, max_gap-1)
        patterns = gb.run()

        # Maximal Sequential Patterns
        # spmf = Spmf("VMSP", input_filename="%s.txt" % app, output_filename="output.txt", arguments=['%d%%' % (min_support * 100), 500, max_gap, False])
        # spmf.run()
        # print(spmf.to_pandas_dataframe(pickle=True))

        GB_time = time.time() - start

        print('Runtime of Gap-Bide: %.2fs\tNo. of Patterns: %d' % (GB_time, len(patterns)))
        time_record.append(('GAP-BIDE', seq_len, fold, '%.3f' % GB_time))

time_df = pd.DataFrame(time_record, columns=('method', 'seq_len', 'fold', 'time'))
time_df = time_df.astype({'time': 'double'})
time_df.to_csv('%s/result.csv' % folder, index=False)
print(time_df.groupby(by=['method', 'seq_len']).agg({'time': ['mean', 'std']}))




# # Print Pattern Result
# for p in patterns:
#     print('%.2f\t%s' % (p[2], p[0]))
#     print('Positions:')
#     for i in range(len(p[1])):
#         print('  Trace %d: %s' % (i, p[1][i]))
#
# import random
# sorted(random.sample(range(1,50), 6))
