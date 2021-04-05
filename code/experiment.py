
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

from TRASE_v2 import *
from pygapbide import *

def safe_div(a,b):
    if a == 0 or b == 0:
        return 0
    else:
        return a/b

'''
        =================================================================
                                Main Program
        =================================================================
'''


archive_url = os.getcwd() + '/components'
apps = os.listdir(archive_url)
# apps = ['synthetic']

threshold = 0.05
min_sup = 0.3

all_result = []

for min_len in [1,2,3,4,5]:
    for max_gap in [1,2,3,4,5]:

        # apps = ['chensi', 'ogden', 'abhi', 'colornote']
        apps = ['chensi', 'ogden', 'abhi']
        for app in apps:
            if app.startswith(','):
                continue

            app_folder = '%s/%s' % (archive_url, app)
            traces = sorted(os.listdir(app_folder))

            data = []
            trace_idx = 0
            unique_components = set()
            for trace in traces:
                if trace.startswith('.'):
                    continue

                components = pickle.load(open('%s/%s' % (app_folder, trace), "rb"))
                # print('Number of Raw Preliminary Phrases: %d' % len(components))

                # Remove consecutive duplicate items
                components = [i[0] for i in groupby(components)]
                # print('Number of Cleaned Preliminary Phrases: %d' % len(components))

                for component in components:
                    unique_components.add(frozenset(component))
                data.append(components)
            unique_components = [set(x) for x in unique_components]
            # unique_components = list(unique_components)

            # Clustering IDs
            dist_mat = compute_distance_matrix(unique_components)
            model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete', distance_threshold=threshold)
            clustering = model.fit(dist_mat)
            (unique, counts) = np.unique(clustering.labels_, return_counts=True)
            unique = unique[counts > 1]
            counts = counts[counts > 1]
            for label in unique:
                # Find intersection within cluster
                indices = np.where(clustering.labels_ == label)[0]
                cluster_instances = [unique_components[i] for i in indices]
                cluster_head = set.intersection(*cluster_instances)
                # Update data with the intersection
                for trace in data:
                    for i in range(len(trace)):
                        if trace[i] in cluster_instances:
                            trace[i] = cluster_head

            sequence_db = []
            for trace in data:
                sequence = []
                for i in range(len(trace)):
                    # Convert the trace only preserve the first occuring phase
                    if trace[i] not in sequence:
                        sequence.append(trace[i])
                    # sequence.append(trace[i])
                sequence_db.append(sequence)

            phase_sizes = []
            for seq in sequence_db:
                temp = []
                for phase in seq:
                    temp.append(len(phase))
                phase_sizes.append(np.mean(temp))

            min_size = np.round(min_len * np.mean(phase_sizes))

            # Construct groundtruth and sequence_db
            remove_indices = []
            if app == 'chensi':
                labels = {
                    'add': [1,2,5,7],
                    'edit': [3,6],
                    'delete': [4,6,9]}
                remove_indices += [32,33]
            elif app == 'ogden':
                labels = {
                    'add': [7,22,23],
                    'edit': [8,24,25],
                    'delete': [9,26,27]}
                remove_indices += [1,2,3,4,5,6]
            elif app == 'abhi':
                labels = {
                    'add': [3,41,42],
                    'edit': [4,43,44],
                    'delete': [5,45,46]}
                remove_indices += [1,2,39,40]
            elif app == 'colornote':
                labels = {
                    'add': [0,1,2,3,4,5],
                    'edit': [3,4,5],
                    'delete': [6,7,8]}

            groundtruth = {}
            for label in labels:
                indices = labels[label]
                groundtruth[label] = []
                for idx in indices:
                    remove_indices.append(idx)
                    groundtruth[label].append(sequence_db[idx-1])
            remove_indices = sorted(remove_indices)[::-1]
            for idx in remove_indices:
                del sequence_db[idx-1]

            '''
                =================================================================
                                        Algorithm Begin
                =================================================================
            '''

            # starting time
            start = time.time()

            id_list, Z = TRASE(sequence_db, min_sup, min_size, max_gap, print_status=False)

            # # TEMP
            # temp = []
            # for phase in sequence_db[2]:
            #     temp.append(id_list.ids.index(phase))
            # print(temp)
            # # END TEMP

            # end time
            end = time.time()
            TRASE_time = end - start

            # print('\nRuntime of TRASE is %.2fs' % TRASE_time)

            # print('Number of Raw Patterns: %d' % len(Z))

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

            # print('Number of Closed Patterns: %d' % len(Z))

            Z = np.array(sorted(Z, key=lambda pattern: pattern.support, reverse=True))

            # Prnt closed patterns
            # for pattern in Z:
            #     print('%.2f\t%.100s' % (pattern.support, pattern.pattern))

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
            adjacency_list = np.array(adjacency_list, dtype=object)

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

            # Generate Human Readable labels for each class prediction

            # Read method list and group names by patterns
            methods_df = pd.read_csv('method list/%s.csv' % app, header=None, names=('index', 'method_id'))
            methods = []
            for i, row in methods_df.iterrows():
                words = []
                # Remove text in bracket
                string = re.sub(r'\([^)]*\)', '', row.method_id)
                # Get method name and activity/fragment name only
                terms = re.findall(r"[\w']+", string)[-2:]
                for term in terms:
                    words += [x.lower() for x in re.findall('.[^A-Z]*', term)]
                methods.append(words)

            # Build bag of word for each pattern
            pattern_methods = []
            for pattern in patterns:
                documents = []
                for pid in pattern.pattern:
                    for method_ids in id_list.ids[pid]:
                        documents.append(' '.join(methods[method_ids]))
                pattern_methods.append(documents)

            gt_dict = {}
            pt_dict = {}
            pt_methods = {}
            pt_count = {}

            # Construct gt_dict
            for label in labels:
                gt_patterns = []
                # Phase Level
                for gt_pattern in groundtruth[label]:
                    for phase in gt_pattern:
                        try:
                            gt_patterns.append(id_list.ids.index(phase))
                        except Exception as e:
                            id_list.add_phase(phase)
                            gt_patterns.append(id_list.ids.index(phase))
                gt_dict[label] = set(gt_patterns)
                # Method Level
                # for gt_pattern in groundtruth[label]:
                #     gt_patterns.append(set.union(*gt_pattern))
                # gt_dict[label] = set.intersection(*gt_patterns)

            # Find utility methods
            common_methods = set.intersection(* [gt_dict[label] for label in gt_dict])
            for label in labels:
                gt_dict[label] = gt_dict[label].difference(common_methods)
            gt_dict['utility'] = common_methods
            labels['utility'] = []

            # Find utility methods among add and edit
            # common_methods = gt_dict['add'].intersection(gt_dict['edit'])
            # gt_dict['add'] = gt_dict['add'].difference(common_methods)
            # gt_dict['edit'] = gt_dict['edit'].difference(common_methods)
            # gt_dict['add_edit'] = common_methods
            # labels['add_edit'] = []

            for label in labels:
                pt_dict[label] = set()
                pt_methods[label] = []
                pt_count[label] = 0

            for i in range(len(patterns)):

                # Phase Level
                ml_pattern = set(patterns[i].pattern)

                # Method Level
                # ml_pattern = []
                # for pid in patterns[i].pattern:
                #     ml_pattern.append(id_list.ids[pid])
                # ml_pattern = set().union(*ml_pattern)

                # Find the best matching groundtruth
                accuracy = dict()
                for label in labels:
                    accuracy[label] = len(gt_dict[label].intersection(ml_pattern)) / len(ml_pattern)

                prediction = max(accuracy, key=accuracy.get)
                pt_dict[prediction] = pt_dict[prediction].union(ml_pattern)
                pt_count[prediction] += 1

                vectorizer = TfidfVectorizer(use_idf=True)
                tfIdf = vectorizer.fit_transform(pattern_methods[i])
                df = pd.DataFrame(tfIdf[0].T.todense(), index=vectorizer.get_feature_names(), columns=["TF-IDF"])
                df = df.sort_values('TF-IDF', ascending=False)
                pt_methods[prediction].append(set(df.index[:5]))

            # Construct Confusion Matrix
            results = []
            for label in labels:

                # Phase Level
                pt_set = []
                gt_set = []
                for pid in pt_dict[label]:
                    pt_set += id_list.ids[pid]
                for pid in gt_dict[label]:
                    gt_set += id_list.ids[pid]
                pt_set = set(pt_set)
                gt_set = set(gt_set)

                # # Method Level
                # pt_set = pt_dict[label]
                # gt_set = gt_dict[label]

                TP = len(gt_set.intersection(pt_set))
                FP = len(pt_set.difference(gt_set))
                FN = len(gt_set.difference(pt_set))

                precision = safe_div(TP, (TP + FP)) * 100
                recall = safe_div(TP, (TP + FN)) * 100
                f1 = 2 * safe_div((precision * recall), (precision + recall))

                results.append((label, pt_count[label], precision, recall, f1))
            result_df = pd.DataFrame(results, columns=['class', 'count', 'precision', 'recall', 'f1'])

            # Aggregated Result
            RMSE = np.sqrt(np.sum(np.square(result_df['count'] - 1)) / len(result_df))
            avg_prec = np.mean(result_df['precision'])
            avg_recall = np.mean(result_df['recall'])
            avg_f1 = np.mean(result_df['f1'])
            print('RMSE: %.2f\tPrec: %.2f\tRecall: %.2f\tF1: %.2f' % (RMSE, avg_prec, avg_recall, avg_f1))

            all_result.append((app, min_len, max_gap, RMSE, avg_prec, avg_recall, avg_f1))

            # print(result_df)
            # print(pt_methods)

            # # Print keywords from groundtruth
            # gt_methods = {}
            # gt_keywords = {}
            # for label in gt_dict:
            #     gt_methods[label] = []
            #     for mid in gt_dict[label]:
            #         gt_methods[label].append(methods[mid])
            #
            #     vectorizer = TfidfVectorizer(use_idf=True)
            #     tfIdf = vectorizer.fit_transform(gt_methods[label])
            #     df = pd.DataFrame(tfIdf[0].T.todense(), index=vectorizer.get_feature_names(), columns=["TF-IDF"])
            #     df = df.sort_values('TF-IDF', ascending=False)
            #     gt_keywords[label].append(set(df.index[:5]))
            # print(gt_keywords)

            # '''
            #         Testing Simple Approach With VMSP
            # '''
            # # Represent Data by ID-List IDs
            # temp_data = []
            # for trace in sequence_db:
            #     temp_trace = []
            #     for event in trace:
            #         temp_trace.append(id_list.ids.index(event))
            #     temp_data.append(temp_trace)
            #
            # # Write data to file
            # # f = open("%s.txt" % app, "w+")
            # # for trace in temp_data:
            # #     for i in range(len(trace)):
            # #         f.write('%d -1 ' % trace[i])
            # #     f.write('-2\r\n')
            # # f.close()
            #
            # start = time.time()
            #
            # # All Sequential Patterns
            # # spmf = Spmf("SPAM", input_filename="%s.txt" % app, output_filename="output.txt", arguments=[0.6, 5, 500, max_gap, False])
            # # Closed Sequential Patterns
            # sdb = []
            # for trace in temp_data:
            #     s = []
            #     for i in range(len(trace)):
            #         s.append(trace[i])
            #     sdb.append(s)
            # gb = Gapbide(sdb, int(min_support * id_list.n_traces), 0, max_gap-1)
            # temp = gb.run()
            # # Maximal Sequential Patterns
            # # spmf = Spmf("VMSP", input_filename="%s.txt" % app, output_filename="output.txt", arguments=['%d%%' % (min_support * 100), 500, max_gap, False])
            # # spmf.run()
            # # print(spmf.to_pandas_dataframe(pickle=True))
            #
            # end = time.time()
            # GB_time = end - start
            #
            # print('\nRuntime of Gap-Bide is %.2fs' % GB_time)
            #
            # '''
            #         End Testing Simple Approach With VMSP
            # '''


all_result = pd.DataFrame(all_result, columns=['app', 'min_len', 'max_gap', 'RMSE', 'PREC', 'RECALL', 'F1'])
pd.set_option('display.max_rows', 80)
print(all_result.groupby(['app', 'min_len', 'max_gap']).agg(['mean']))

# # Print Pattern Result
# for p in patterns:
#     print('%.2f\t%s' % (p[2], p[0]))
#     print('Positions:')
#     for i in range(len(p[1])):
#         print('  Trace %d: %s' % (i, p[1][i]))
#
# import random
# sorted(random.sample(range(1,50), 6))
