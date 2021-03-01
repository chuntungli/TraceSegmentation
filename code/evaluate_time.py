


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

            for i in range(len(Z) - 1, 0, -1):
                for j in range(len(Z)):
                    if i == j:
                        continue
                    # Z[i] and Z[j] have the same support and Z[i] is a subsequence of Z[j]
                    if (Z[i].support == Z[j].support) & (is_subsequence(Z[i].pattern, Z[j].pattern)):
                        # print('Z[%d] is a subsequence of Z[%d]: (%s) and (%s)' % (i, j, Z[i].pattern, Z[j].pattern))
                        # Delete Z[i]
                        del Z[i]
                        break

            Z = np.array(sorted(Z, key=lambda pattern: pattern.support, reverse=True))

            # Vertex list contains the weight of the phase
            # Edge list contains relationship among phases if two phases are overlapped
            vertex_list = []
            edge_list = []
            for i in range(len(Z)):
                # Weight is defined as the number_of_method * support_of_phase
                vertex_list.append(Z[i].support * sum([len(id_list.ids[x]) for x in Z[i].pattern]))
                for j in range(i + 1, len(Z)):
                    if is_intersect(Z[i], Z[j]):
                        # print('%d and %d are overlapped' % (i, j))
                        edge_list.append((i, j))
                        edge_list.append((j, i))
            vertex_list = np.array(vertex_list)
            edge_list = np.array(edge_list)

            adjacency_list = [[] for __ in vertex_list]
            for edge in edge_list:
                adjacency_list[edge[0]].append(edge[1])
            adjacency_list = np.array(adjacency_list)

            subgraphs = generateSubgraphs(vertex_list, adjacency_list)

            solution = np.zeros(len(vertex_list), dtype=bool)
            for subgraph in subgraphs:
                vl = np.array(copy.deepcopy(vertex_list[subgraph]))
                al = np.array(copy.deepcopy(adjacency_list[subgraph]))
                for i in range(len(al)):
                    for j in range(len(al[i])):
                        al[i][j] = np.where(subgraph == al[i][j])[0][0]
                OPT_X = MWIS(vl, al)
                solution[subgraph] = OPT_X

            patterns = Z[solution]

            # Print Pattern Result
            # for pattern in patterns:
            #     print('%.2f\t%s' % (pattern.support, pattern.pattern))

            # Construct list of groundtruth
            labels = list(groundtruth.keys())

            gt_dict = dict()
            pt_dict = dict()
            pt_count = dict()

            # Construct gt_dict
            for label in labels:
                gt_dict[label] = set().union(*groundtruth[label]['pattern'])
                pt_dict[label] = set()
                pt_count[label] = 0

            for pattern in patterns:
                ml_pattern = []
                for pid in pattern.pattern:
                    ml_pattern.append(id_list.ids[pid])
                ml_pattern = set().union(*ml_pattern)

                # Find the best matching groundtruth
                accuracy = dict()
                for label in labels:
                    accuracy[label] = len(gt_dict[label].intersection(ml_pattern)) / len(ml_pattern)

                prediction = max(accuracy, key=accuracy.get)
                pt_dict[prediction] = pt_dict[prediction].union(ml_pattern)
                pt_count[prediction] += 1

            # Construct Confusion Matrix
            for label in labels:
                pt_set = pt_dict[label]
                gt_set = gt_dict[label]

                TP = len(gt_set.intersection(pt_set))
                FP = len(pt_set.difference(gt_set))
                FN = len(gt_set.difference(pt_set))

                precision = safe_div(TP, (TP + FP))
                recall = safe_div(TP, (TP + FN))

                performance_record.append((int(value), fold, label, pt_count[label], precision, recall))

            '''
                =================================================================
                                    Gap-Bide Algorithm
                =================================================================
            '''

            # # Convert the trace to other format
            # sdb = []
            # for trace in sequence_db:
            #     s = []
            #     for event in trace:
            #         s.append(id_list.ids.index(event))
            #     sdb.append(s)
            #
            # # Write data to file
            # # f = open("%s.txt" % app, "w+")
            # # for trace in temp_data:
            # #     for i in range(len(trace)):
            # #         f.write('%d -1 ' % trace[i])
            # #     f.write('-2\r\n')
            # # f.close()
            #
            # # All Sequential Patterns
            # # spmf = Spmf("SPAM", input_filename="%s.txt" % app, output_filename="output.txt", arguments=[0.6, 5, 500, max_gap, False])
            #
            # # Closed Sequential Patterns
            # start = time.time()
            #
            # gb = Gapbide(sdb, int(min_sup * id_list.n_traces), 0, max_gap - 1)
            #
            # q = mp.Queue()
            # p = mp.Process(target=gb.run, args=(q,))
            # p.start()
            #
            # try:
            #     patterns = q.get(timeout=time_out)
            # except Exception:
            #     patterns = []
            #
            # p.join(timeout=time_out)
            #
            # if p.is_alive():
            #     p.kill()
            #     GB_time = np.inf
            # else:
            #     GB_time = time.time() - start
            #
            # print('\nRuntime of Gap-Bide: %.2fs\tNo. of Patterns: %d' % (GB_time, len(patterns)))
            # time_record.append(('GAP-BIDE', int(value), fold, '%.3f' % GB_time))

            # Maximal Sequential Patterns
            # spmf = Spmf("VMSP", input_filename="%s.txt" % app, output_filename="output.txt", arguments=['%d%%' % (min_support * 100), 500, max_gap, False])
            # spmf.run()
            # print(spmf.to_pandas_dataframe(pickle=True))

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

    min_sup = 0.6
    min_size = 100
    max_gap = 2
    time_out = 500

    # Test time on different number of sequences
    time_record, performance_record = evaluate('components/synthetic/pat_len', 'groundtruth/synthetic/pat_len')
    time_df = pd.DataFrame(time_record, columns=('method', 'pat_len', 'fold', 'time'))
    time_df = time_df.astype({'time': 'double'})
    performance_df = pd.DataFrame(performance_record, columns=('pat_len', 'fold', 'label', 'seg_count', 'precision', 'recall'))
    time_df.to_csv('%s/time_result_pat_len.csv' % result_folder, index=False)
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


# Read data for Pattern Length
df = pd.read_csv('%s/time_pat_len.csv' % result_folder)
df['pat_size'] = df['pat_len'] * 20
aggResult = df.groupby(['pat_size']).agg({'time': ['mean']})


fig = plt.figure(figsize=(4,3), dpi=120)
plt.plot(aggResult.index,
         aggResult.time,
         linestyle = '--', marker = 's', fillstyle = 'none', label = 'TRASE')
# plt.xlim((-0.05,1.05))
# plt.ylim((-5,105))
plt.xlabel('Pattern Size', fontsize=12)
plt.ylabel("Execution Time(sec)", fontsize=12)
plt.gca().yaxis.grid(True, linestyle='--')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol = 3)
plt.legend(loc='right', bbox_to_anchor=(0.93,0.55))
fig.tight_layout()
plt.show()
# fig.savefig('fig_tau_effect.pdf', format='pdf')
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

