
import os
import re
import copy
import time
import pickle
import numpy as np
import pandas as pd

from spmf import Spmf
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


archive_url = os.getcwd() + '/components'
# apps = os.listdir(archive_url)
apps = ['synthetic']

threshold = 0.2
min_support = 0.2
min_method = 5
max_gap = 1

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
        print('Number of Raw Preliminary Phrases: %d' % len(components))

        # Remove consecutive duplicate items
        components = [i[0] for i in groupby(components)]
        print('Number of Cleaned Preliminary Phrases: %d' % len(components))

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
                                Algorithm Begin
        =================================================================
    '''

    # starting time
    start = time.time()

    # Build ID List
    id_list = IdList()
    id_list.build_list(sequence_db, max_gap, min_support)

    # TEMP
    temp = []
    for phase in sequence_db[2]:
        temp.append(id_list.ids.index(phase))
    print(temp)
    # END TEMP

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

    print('\nRuntime of TRASE is %.2fs' % (end - start))

    # # print('Number of Raw Patterns: %d' % len(Z))
    #
    # for i in range(len(Z) - 1, 0, -1):
    #     for j in range(len(Z)):
    #         if i == j:
    #             continue
    #         # Z[i] and Z[j] have the same support and Z[i] is a subsequence of Z[j]
    #         if (Z[i].support == Z[j].support) & (is_subsequence(Z[i].pattern, Z[j].pattern)):
    #             print('Z[%d] is a subsequence of Z[%d]: (%s) and (%s)' % (i, j, Z[i].pattern, Z[j].pattern))
    #             # Delete Z[i]
    #             del Z[i]
    #             break
    #
    # print('Number of Closed Patterns: %d' % len(Z))
    #
    # Z = np.array(sorted(Z, key=lambda pattern: pattern.support, reverse=True))
    #
    # # Prnt closed patterns
    # for pattern in Z:
    #     print('%.2f\t%.100s' % (pattern.support, pattern.pattern))
    #
    # # Vertex list contains the weight of the phase
    # # Edge list contains relationship among phases if two phases are overlapped
    # vertex_list = []
    # edge_list = []
    # for i in range(len(Z)):
    #     # Weight is defined as the number_of_method * support_of_phase
    #     vertex_list.append(Z[i].support * sum([len(id_list.ids[x]) for x in Z[i].pattern]))
    #     for j in range(i + 1, len(Z)):
    #         if is_intersect(Z[i], Z[j]):
    #             print('%d and %d are overlapped' % (i, j))
    #             edge_list.append((i, j))
    #             edge_list.append((j, i))
    # vertex_list = np.array(vertex_list)
    # edge_list = np.array(edge_list)
    #
    # adjacency_list = [[] for __ in vertex_list]
    # for edge in edge_list:
    #     adjacency_list[edge[0]].append(edge[1])
    # adjacency_list = np.array(adjacency_list)
    #
    # subgraphs = _generateSubgraphs(vertex_list, adjacency_list)
    #
    # solution = np.zeros(len(vertex_list), dtype=bool)
    # for subgraph in subgraphs:
    #     vl = np.array(copy.deepcopy(vertex_list[subgraph]))
    #     al = np.array(copy.deepcopy(adjacency_list[subgraph]))
    #     for i in range(len(al)):
    #         for j in range(len(al[i])):
    #             al[i][j] = np.where(subgraph == al[i][j])[0][0]
    #     OPT_X = MWIS(vl, al)
    #     solution[subgraph] = OPT_X
    #
    # patterns = Z[solution]
    #
    # # Print Pattern Result
    # for pattern in patterns:
    #     print('%.2f\t%s' % (pattern.support, pattern.pattern))
    #
    #
    # # for pattern in patterns:
    # #     phases = []
    # #     for phase in pattern[0]:
    # #         phases.append(id_list.ids[phase])
    # #     print('%.2f\t%s' % (pattern[2], phases))
    #
    # # Read method list and group names by patterns
    # methods_df = pd.read_csv('method list/%s.csv' % app, header=None, names=('index', 'method_id'))
    # methods = []
    # for i,row in methods_df.iterrows():
    #     words = []
    #     # Remove text in bracket
    #     string = re.sub(r'\([^)]*\)', '', row.method_id)
    #     # Get method name and activity/fragment name only
    #     terms = re.findall(r"[\w']+", string)[-2:]
    #     for term in terms:
    #         words += [x.lower() for x in re.findall('.[^A-Z]*', term)]
    #     methods.append(words)
    #
    # # Build bag of word for each pattern
    # pattern_methods = []
    # for pattern in patterns:
    #     documents = []
    #     for phase in pattern.pattern:
    #         documents.append(' '.join(methods[phase]))
    #     pattern_methods.append(documents)
    #
    # # Perform TF-IDF
    # for i in range(len(patterns)):
    #     phases = []
    #     for phase in patterns[i].pattern:
    #         phases.append(id_list.ids[phase])
    #     print('%.2f\t%s' % (patterns[i].support, phases))
    #
    #     vectorizer = TfidfVectorizer(use_idf=True)
    #     tfIdf = vectorizer.fit_transform(pattern_methods[i])
    #     df = pd.DataFrame(tfIdf[0].T.todense(), index=vectorizer.get_feature_names(), columns=["TF-IDF"])
    #     df = df.sort_values('TF-IDF', ascending=False)
    #     print(df.head(10))



    '''
            Testing Simple Approach With VMSP
    '''
    # Represent Data by ID-List IDs
    temp_data = []
    for trace in sequence_db:
        temp_trace = []
        for event in trace:
            temp_trace.append(id_list.ids.index(event))
        temp_data.append(temp_trace)

    # Write data to file
    # f = open("%s.txt" % app, "w+")
    # for trace in temp_data:
    #     for i in range(len(trace)):
    #         f.write('%d -1 ' % trace[i])
    #     f.write('-2\r\n')
    # f.close()

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

    print('\nRuntime of Gap-Bide is %.2fs' % (end - start))

    '''
            End Testing Simple Approach With VMSP
    '''


# Print Pattern Result
for p in patterns:
    print('%.2f\t%s' % (p[2], p[0]))
    print('Positions:')
    for i in range(len(p[1])):
        print('  Trace %d: %s' % (i, p[1][i]))

import random
sorted(random.sample(range(1,50), 6))
