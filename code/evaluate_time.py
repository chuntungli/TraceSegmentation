


import os
import copy
import time
import pickle
import shutil
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

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

def safe_div(a,b):
    if a == 0 or b == 0:
        return 0
    else:
        return a/b

def evaluate(data_folder, gt_folder):
    performance_record = []
    time_record = []
    skip_gb = skip_spam = skip_vmsp = False

    for value in sorted(os.listdir(data_folder)):
        if not value.isdigit():
            continue

        folds = sorted(os.listdir('%s/%s' % (data_folder, value)))

        for fold in folds:
            fold = int(fold)

            groundtruth = pickle.load(open('%s/%s/%d/groundtruth.p' % (gt_folder, value, fold), 'rb'))
            sequence_db = build_seqDB('%s/%s/%d' % (data_folder, value, fold))

            '''
                =================================================================
                                        TRASE Algorithm
                =================================================================
            '''

            start = time.time()
            # seq_db, min_sup, min_size, max_gap
            id_list, Z = TRASE(sequence_db, min_sup, min_size, max_gap)
            TRASE_time = time.time() - start

            print('\nRuntime of TRASE: %.2fs\tNo. of Patterns: %d' % (TRASE_time, len(Z)))
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

            if not skip_gb:
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
                    p.terminate()
                    GB_time = np.inf
                    skip_gb = True
                    print('GAP-BIDE time out at value: %s' % value)
                    break
                else:
                    GB_time = time.time() - start

                print('\nRuntime of Gap-Bide: %.2fs\tNo. of Patterns: %d' % (GB_time, len(patterns)))
                time_record.append(('GAP-BIDE', int(value), fold, '%.3f' % GB_time))

            '''
                =================================================================
                                        VMSP Algorithm
                =================================================================
            '''

            # Write data to file
            f = open("%s.txt" % app, "w+")
            for trace in temp_data:
                for i in range(len(trace)):
                    f.write('%d -1 ' % trace[i])
                f.write('-2\r\n')
            f.close()

            if not skip_vmsp:

                start = time.time()

                # Maximal Sequential Patterns
                spmf = Spmf("VMSP", input_filename="%s.txt" % app, output_filename="output.txt",
                            arguments=['%d%%' % (min_support * 100), 100, max_gap, False])

                p = mp.Process(target=spmf.run)
                p.start()

                p.join(timeout=time_out)

                if p.is_alive():
                    p.terminate()
                    vmsp_time = np.inf
                    skip_vmsp = True
                    print('VMSP time out at value: %s' % value)
                    break
                else:
                    vmsp_time = time.time() - start

                print('\nRuntime of VMSP: %.2fs\tNo. of Patterns: %d' % (vmsp_time, len(spmf.to_pandas_dataframe(pickle=True))))
                time_record.append(('VMSP', int(value), fold, '%.3f' % vmsp_time))

            '''
                =================================================================
                                        SPAM Algorithm
                =================================================================
            '''

            # Write data to file
            f = open("%s.txt" % app, "w+")
            for trace in temp_data:
                for i in range(len(trace)):
                    f.write('%d -1 ' % trace[i])
                f.write('-2\r\n')
            f.close()

            if not skip_spam:

                start = time.time()

                # Maximal Sequential Patterns
                spmf = Spmf("SPAM", input_filename="%s.txt" % app, output_filename="output.txt",
                            arguments=['%d%%' % (min_support * 100), 10, 100, max_gap, False])

                p = mp.Process(target=spmf.run)
                p.start()

                p.join(timeout=time_out)

                if p.is_alive():
                    p.terminate()
                    spam_time = np.inf
                    skip_spam = True
                    print('SPAM time out at value: %s' % value)
                    break
                else:
                    spam_time = time.time() - start

                print('\nRuntime of SPAM: %.2fs\tNo. of Patterns: %d' % (
                spam_time, len(spmf.to_pandas_dataframe(pickle=True))))
                time_record.append(('SPAM', int(value), fold, '%.3f' % spam_time))


    return (time_record, performance_record)

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

    min_sup = 0.5
    min_size = 100
    max_gap = 2
    time_out = 500

    # Test time on different number of sequences
    time_record, performance_record = evaluate('components/synthetic/pat_len', 'groundtruth/synthetic/pat_len')
    time_df = pd.DataFrame(time_record, columns=('method', 'pat_len', 'fold', 'time'))
    time_df = time_df.astype({'time': 'double'})
    performance_df = pd.DataFrame(performance_record, columns=('pat_len', 'fold', 'label', 'seg_count', 'precision', 'recall'))
    time_df.to_csv('%s/result_pat_len.csv' % result_folder, index=False)
    performance_df.to_csv('%s/performance_result_pat_len.csv' % result_folder, index=False)

    print(time_df.groupby(by=['method', 'pat_len']).agg({'time': ['mean', 'std']}))

    '''
        =================================================================
                        Evaluate Different No. of Sequences
        =================================================================
    '''

    # Test time on different number of sequences
    time_record, performance_record = evaluate('components/synthetic/n_seq', 'groundtruth/synthetic/n_seq')

    time_df = pd.DataFrame(time_record, columns=('method', 'n_seq', 'fold', 'time'))
    time_df = time_df.astype({'time': 'double'})
    time_df.to_csv('%s/result_n_seq.csv' % result_folder, index=False)
    performance_df = pd.DataFrame(performance_record, columns=('n_seq', 'fold', 'label', 'seg_count', 'precision', 'recall'))
    performance_df.to_csv('%s/performance_result_n_seq.csv' % result_folder, index=False)
    print(time_df.groupby(by=['method', 'n_seq']).agg({'time': ['mean', 'std']}))

    '''
        =================================================================
                        Evaluate Different No. of Sequences
        =================================================================
    '''

    # Test time on different number of sequences
    time_record, performance_record = evaluate('components/synthetic/seq_len', 'groundtruth/synthetic/seq_len')

    time_df = pd.DataFrame(time_record, columns=('method', 'seq_len', 'fold', 'time'))
    time_df = time_df.astype({'time': 'double'})
    time_df.to_csv('%s/result_seq_len.csv' % result_folder, index=False)
    performance_df = pd.DataFrame(performance_record, columns=('seq_len', 'fold', 'label', 'seg_count', 'precision', 'recall'))
    performance_df.to_csv('%s/performance_result_seq_len.csv' % result_folder, index=False)
    print(time_df.groupby(by=['method', 'seq_len']).agg({'time': ['mean', 'std']}))


result_folder = 'result'

'''
============================================================
                Read data for Pattern Length
============================================================
'''

df = pd.read_csv('%s/result_pat_len.csv' % result_folder)
aggResult = df.groupby(['pat_len']).agg({'time': ['median']})

fig = plt.figure(figsize=(4,3), dpi=120)
plt.plot(aggResult.index,
         aggResult.time,
         linestyle = '--', marker = 's', fillstyle = 'none', label = 'TRASE')
# plt.xlim((-0.05,1.05))
# plt.ylim((-5,105))
plt.xlabel('Pattern Length', fontsize=12)
plt.ylabel("Execution Time(sec)", fontsize=12)
plt.gca().yaxis.grid(True, linestyle='--')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol = 3)
plt.legend(loc='right', bbox_to_anchor=(0.4,0.9))
fig.tight_layout()
plt.show()
fig.savefig('fig_result_pat_len.pdf', format='pdf')
plt.close(fig)

'''
============================================================
                Read data for Number of Sequences
============================================================
'''

df = pd.read_csv('%s/result_n_seq.csv' % result_folder)
aggResult = df.groupby(['n_seq']).agg({'time': ['median']})


fig = plt.figure(figsize=(4,3), dpi=120)
plt.plot(aggResult.index,
         aggResult.time,
         linestyle = '--', marker = 's', fillstyle = 'none', label = 'TRASE')
# plt.xlim((-0.05,1.05))
# plt.ylim((-5,105))
plt.xlabel('No. of Sequences', fontsize=12)
plt.ylabel("Execution Time(sec)", fontsize=12)
plt.gca().yaxis.grid(True, linestyle='--')
plt.legend(loc='right', bbox_to_anchor=(0.4,0.9))
fig.tight_layout()
plt.show()
fig.savefig('fig_result_n_seq.pdf', format='pdf')
plt.close(fig)

'''
============================================================
                Read data for Sequence Length
============================================================
'''

df = pd.read_csv('%s/result_seq_len.csv' % result_folder)
aggResult = df.groupby(['seq_len']).agg({'time': ['median']})


fig = plt.figure(figsize=(4,3), dpi=120)
plt.plot(aggResult.index,
         aggResult.time,
         linestyle = '--', marker = 's', fillstyle = 'none', label = 'TRASE')
# plt.xlim((-0.05,1.05))
plt.ylim((0,11))
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel("Execution Time(sec)", fontsize=12)
plt.gca().yaxis.grid(True, linestyle='--')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol = 3)
plt.legend(loc='right', bbox_to_anchor=(0.4,0.9))
fig.tight_layout()
plt.show()
fig.savefig('fig_result_seq_len.pdf', format='pdf')
plt.close(fig)

# # Print Pattern Result
# for p in patterns:
#     print('%.2f\t%s' % (p[2], p[0]))
#     print('Positions:')
#     for i in range(len(p[1])):
#         print('  Trace %d: %s' % (i, p[1][i]))
#
# import random
# sorted(random.sample(range(1,50), 6))

