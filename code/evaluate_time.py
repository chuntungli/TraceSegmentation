


import os
import time
import pickle
import shutil
import numpy as np
import pandas as pd
import multiprocessing as mp

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

def evaluate(folder):
    for value in sorted(os.listdir(folder)):
        if not value.isdigit():
            continue

        folds = sorted(os.listdir('%s/%s' % (folder, value)))

        for fold in folds:
            fold = int(fold)

            sequence_db = build_seqDB('%s/%s/%d' % (folder, value, fold))

            '''
                =================================================================
                                        TRASE Algorithm
                =================================================================
            '''

            # Build ID List
            id_list = IdList()
            id_list.build_list(sequence_db, min_sup, max_gap)

            start = time.time()
            # seq_db, min_sup, min_size, max_gap
            q = mp.Queue()
            p = mp.Process(target=TRASE, args=(q, sequence_db, min_sup, min_size, max_gap,))
            p.start()

            try:
                patterns = q.get(timeout=time_out)
            except Exception:
                patterns = []

            p.join(timeout=time_out)
            if p.is_alive():
                p.kill()
                TRASE_time = np.inf
            else:
                TRASE_time = time.time() - start

            print('\nRuntime of TRASE: %.2fs\tNo. of Patterns: %d' % (TRASE_time, len(patterns)))
            time_record.append(('TRASE', int(value), fold, '%.3f' % TRASE_time))


            '''
                =================================================================
                                    Gap-Bide Algorithm
                =================================================================
            '''

            # Convert the trace to other format
            sdb = []
            for trace in sequence_db:
                s = []
                for event in trace:
                    s.append(id_list.ids.index(event))
                sdb.append(s)

            # Write data to file
            # f = open("%s.txt" % app, "w+")
            # for trace in temp_data:
            #     for i in range(len(trace)):
            #         f.write('%d -1 ' % trace[i])
            #     f.write('-2\r\n')
            # f.close()

            # All Sequential Patterns
            # spmf = Spmf("SPAM", input_filename="%s.txt" % app, output_filename="output.txt", arguments=[0.6, 5, 500, max_gap, False])

            # Closed Sequential Patterns
            start = time.time()

            gb = Gapbide(sdb, int(min_sup * id_list.n_traces), 0, max_gap - 1)

            q = mp.Queue()
            p = mp.Process(target=gb.run, args=(q,))
            p.start()

            try:
                patterns = q.get(timeout=time_out)
            except Exception:
                patterns = []

            p.join(timeout=time_out)

            if p.is_alive():
                p.kill()
                GB_time = np.inf
            else:
                GB_time = time.time() - start

            print('\nRuntime of Gap-Bide: %.2fs\tNo. of Patterns: %d' % (GB_time, len(patterns)))
            time_record.append(('GAP-BIDE', int(value), fold, '%.3f' % GB_time))

            # Maximal Sequential Patterns
            # spmf = Spmf("VMSP", input_filename="%s.txt" % app, output_filename="output.txt", arguments=['%d%%' % (min_support * 100), 500, max_gap, False])
            # spmf.run()
            # print(spmf.to_pandas_dataframe(pickle=True))

    return time_record

if __name__ == '__main__':

    result_folder = 'result'
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    os.makedirs(result_folder)

    '''
        =================================================================
                        Evaluate Different Length of Patterns
        =================================================================
    '''

    time_record = []
    min_sup = 0.6
    min_size = 5
    max_gap = 2
    time_out = 500

    # Test time on different number of sequences
    evaluate('components/synthetic/pat_len')

    time_df = pd.DataFrame(time_record, columns=('method', 'pat_len', 'fold', 'time'))
    time_df = time_df.astype({'time': 'double'})
    time_df.to_csv('%s/result_pat_len.csv' % result_folder, index=False)
    print(time_df.groupby(by=['method', 'pat_len']).agg({'time': ['mean', 'std']}))

    '''
        =================================================================
                        Evaluate Different No. of Sequences
        =================================================================
    '''

    # Test time on different number of sequences
    evaluate('components/synthetic/n_seq')

    time_df = pd.DataFrame(time_record, columns=('method', 'n_seq', 'fold', 'time'))
    time_df = time_df.astype({'time': 'double'})
    time_df.to_csv('%s/result_n_seq.csv' % result_folder, index=False)
    print(time_df.groupby(by=['method', 'n_seq']).agg({'time': ['mean', 'std']}))

    '''
        =================================================================
                        Evaluate Different No. of Sequences
        =================================================================
    '''

    # Test time on different number of sequences
    evaluate('components/synthetic/seq_len')

    time_df = pd.DataFrame(time_record, columns=('method', 'seq_len', 'fold', 'time'))
    time_df = time_df.astype({'time': 'double'})
    time_df.to_csv('%s/result_seq_len.csv' % result_folder, index=False)
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

