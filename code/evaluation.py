
import os
import copy
import time
import pickle
import shutil
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

def safe_div(a,b):
    if a == 0 or b == 0:
        return 0
    else:
        return a/b

def evaluate(data_folder, gt_folder, min_sup=0.3, min_size=1, max_gap=1):
    results = []
    for fold in sorted(os.listdir(data_folder)):
        if not fold.isdigit():
            continue

        fold = int(fold)

        groundtruth = pickle.load(open('%s/%d/groundtruth.p' % (gt_folder, fold), 'rb'))
        sequence_db = build_seqDB('%s/%d' % (data_folder, fold))

        '''
            =================================================================
                                    TRASE Algorithm
            =================================================================
        '''

        # seq_db, min_sup, min_size, max_gap
        id_list, Z = TRASE(sequence_db, min_sup, min_size, max_gap)

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

            precision = safe_div(TP, (TP + FP)) * 100
            recall = safe_div(TP, (TP + FN)) * 100
            f1 = 2 * safe_div((precision * recall), (precision + recall))

            results.append((fold, label, pt_count[label], precision, recall, f1))

    return results
    #
    #     '''
    #         =================================================================
    #                             Gap-Bide Algorithm
    #         =================================================================
    #     '''
    #
    #     # Convert the trace to other format
    #     sdb = []
    #     for trace in sequence_db:
    #         s = []
    #         for event in trace:
    #             s.append(id_list.ids.index(event))
    #         sdb.append(s)
    #
    #     # Write data to file
    #     # f = open("%s.txt" % app, "w+")
    #     # for trace in temp_data:
    #     #     for i in range(len(trace)):
    #     #         f.write('%d -1 ' % trace[i])
    #     #     f.write('-2\r\n')
    #     # f.close()
    #
    #     # All Sequential Patterns
    #     # spmf = Spmf("SPAM", input_filename="%s.txt" % app, output_filename="output.txt", arguments=[0.6, 5, 500, max_gap, False])
    #
    #     # Closed Sequential Patterns
    #     start = time.time()
    #
    #     gb = Gapbide(sdb, int(min_sup * id_list.n_traces), 0, max_gap - 1)
    #
    #     q = mp.Queue()
    #     p = mp.Process(target=gb.run, args=(q,))
    #     p.start()
    #
    #     try:
    #         patterns = q.get(timeout=time_out)
    #     except Exception:
    #         patterns = []
    #
    #     p.join(timeout=time_out)
    #
    #     if p.is_alive():
    #         p.kill()
    #         GB_time = np.inf
    #     else:
    #         GB_time = time.time() - start
    #
    #     print('\nRuntime of Gap-Bide: %.2fs\tNo. of Patterns: %d' % (GB_time, len(patterns)))
    #     time_record.append(('GAP-BIDE', int(value), fold, '%.3f' % GB_time))
    #
    #     # Maximal Sequential Patterns
    #     # spmf = Spmf("VMSP", input_filename="%s.txt" % app, output_filename="output.txt", arguments=['%d%%' % (min_support * 100), 500, max_gap, False])
    #     # spmf.run()
    #     # print(spmf.to_pandas_dataframe(pickle=True))
    #
    # return time_record

data_folder = 'components/synthetic/performance'
gt_folder = 'groundtruth/synthetic/performance'

min_sup = 0.5
min_size = 100

for max_gap in np.arange(1,6):

    results = evaluate(data_folder, gt_folder, min_sup, min_size, max_gap)
    results = pd.DataFrame(results, columns=('fold', 'label', 'seg_count', 'precision', 'recall', 'f1'))

    # Aggregate the results
    agg_result = []
    for fold in results.fold.unique():
        fold_result = results[results.fold == fold]
        rmse = np.sqrt(np.sum((fold_result.seg_count - 1) ** 2) / len(fold_result))
        agg_result.append([rmse] + list(fold_result[['precision', 'recall', 'f1']].mean()))

    agg_result = pd.DataFrame(agg_result, columns=('rmse', 'precision', 'recall', 'f1'))
    # Print the aggregated Result
    mean_result = agg_result.mean()
    std_result = agg_result.std()
    # print('\nMaxGap: %d\tRMSE: %.2f (± %.2f)\tPrecision: %.2f (± %.2f)\tRecall: %.2f (± %.2f)\tF1: %.2f (± %.2f)' % (max_gap, mean_result[0], std_result[0], mean_result[1], std_result[1], mean_result[2], std_result[2], mean_result[3], std_result[3]))
    # print('\nMaxGap: %d\tRMSE: %.2f\tPrecision: %.2f\tRecall: %.2f\tF1: %.2f' % (max_gap, mean_result[0], mean_result[1], mean_result[2], mean_result[3]))
    print('%d\t & %.2f\t & %.2f\t & %.2f\t & %.2f' % (max_gap, mean_result[0], mean_result[1], mean_result[2], mean_result[3]))
