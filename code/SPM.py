
import os
import copy
import pickle
import numpy as np
from functools import reduce
from itertools import groupby
from sklearn.cluster import AgglomerativeClustering

archive_url = os.getcwd() + '/components'
apps = os.listdir(archive_url)

# ====================================================================
#                              ID List
# ====================================================================
class IdList:

    def __init__(self):
        self.max_gap = 1  # Gap is 1 by default
        self.__reset()

    def __reset(self):
        self.n_traces = 0           # Number of traces
        self.ids = []               # Preliminary Phase IDs
        self.phase_support = {}     # Support of each phase
        self.idList = {}            # Actual Data Structure
                                    #           ID
                                    # tid   |   Occurence
        self.RXMap = {}             # Right-eXtention-Map indicate pairwise extention relationship
        self.LXMap = {}             # Left-eXtention-Map indicate pairwise extention relationship

    def __add_phase(self, phase):
        if phase in self.ids:
            print('Phase already exists, do nothing.')

        self.ids.append(phase)
        idx = self.ids.index(phase)
        self.phase_support[idx] = 0
        self.idList[idx] = [[] for _ in range(self.n_traces)]
        self.LXMap[idx] = set()
        self.RXMap[idx] = set()

    # Construct the Id List given the list of preliminary phases as list of sets: [{phase_1}, {phase_2}, ..., {phase_n}]
    def build_list(self, traces):
        self.__reset()

        self.n_traces = len(traces)

        if not isinstance(traces, list):
            raise TypeError('Input Traces must be list object.')

        # Itterate Each Trace
        for trace_idx in range(self.n_traces):
            trace = traces[trace_idx]

            for event_idx in range(len(trace)):
                phase = trace[event_idx]

                # Check if phase already exists
                if phase not in self.ids:
                    self.__add_phase(phase)

                phase_idx = self.ids.index(phase)

                # Insert to IdList
                self.idList[phase_idx][trace_idx].append(event_idx)
                self.phase_support[phase_idx] += 1

                # Update Extention Map
                for gap in range(1, self.max_gap + 1):
                    # Left Extention
                    idx = event_idx - gap
                    if idx >= 0:
                        self.LXMap[phase_idx].add(self.ids.index(trace[idx]))
                    # Righ Extention
                    idx = event_idx + gap
                    if idx < len(trace):
                        # Check if phase already exists in ids
                        if trace[idx] not in self.ids:
                            self.__add_phase(trace[idx])
                        self.RXMap[phase_idx].add(self.ids.index(trace[idx]))
            trace_idx += 1

        # Clean Extention Map from removing self-extention
        for i in range(len(self.ids)):
            if i in self.LXMap[i]:
                self.LXMap[i].remove(i)
            if i in self.RXMap[i]:
                self.RXMap[i].remove(i)

    # Matching between a and b where elements of a should greater than b
    def __matching(self, l_loc, r_loc):
        matches = []
        l_idx = r_idx = 0
        while (l_idx < len(l_loc)) & (r_idx < len(r_loc)):
            diff = r_loc[r_idx] - l_loc[l_idx]
            if diff <= 0:                    # r_loc < l_loc
                r_idx += 1
            elif diff <= self.max_gap:      # r_loc > l_loc & diff(r_loc, l_loc) <= max_gap
                matches.append((l_loc[l_idx], r_loc[r_idx]))
                l_idx += 1
                r_idx += 1
            else:                           # r_loc > l_loc & diff(r_loc, l_loc) > max_gap
                l_idx += 1
        return matches

    # Extend pattern by depth-first-search manner
    def __extend_pattern(self, pattern, Z, l_loc, r_loc, que_support):
        print('Extend Pattern: %d - %s - %s' % (que_support, r_loc, pattern))
        patterns = [(pattern, l_loc, r_loc, que_support)]

        # Recursive extention to the right
        for candidate in self.RXMap[pattern[-1]]:

            # Check if candidate is a closed pattern with higher support
            skip_candidate = False
            for z in Z:
                if (z[3] > que_support) & (z[0][0] == candidate):
                    skip_candidate = True
                    break
            if skip_candidate:
                # Skip the candidate as it is contained in another closed pattern with higher support
                continue

            support = 0
            c_loc = []
            for i in range(self.n_traces):
                matches = self.__matching(r_loc[i], self.idList[candidate][i])
                if len(matches) > 0:
                    c_loc.append([x[1] for x in matches])
                else:
                    c_loc.append([])
                support += len(matches)
            if support == que_support:
                patterns.append(
                    self.__extend_pattern(pattern + [candidate], Z, l_loc, c_loc, que_support)
                )

        pattern_sizes = [len(x[0]) for x in patterns]
        closed_pattern = patterns[np.argmax(pattern_sizes)]

        return closed_pattern

    # Find maximum sequential pattern given the starting element
    # Returns: Pattern, l_loc, r_loc, support
    def extend_pattern(self, que_idx, Z):
        que_idlist = self.idList[que_idx]
        que_support = self.phase_support[que_idx]
        return self.__extend_pattern([que_idx], Z, que_idlist, que_idlist, que_support)


def distance(com1, com2):
    return (len(com1 - com2) + len(com2 - com1)) / (len(com1) + len(com2))

def compute_distance_matrix(seq):
    n_seq = len(seq)
    dist_mat = np.zeros((n_seq, n_seq))
    for i in range(n_seq-1):
        que = seq[i]
        for j in range(i+1, n_seq):
            dist_mat[i,j] = distance(que, seq[j])
    dist_mat += dist_mat.T
    return dist_mat

def is_subsequence(query, base):
    l_q = len(query)
    l_b = len(base)
    if l_q > l_b:
        return False
    for i in range(l_b):
        if base[i:i + l_q] == query:
            return True
    return False

def __is_intersect(a, b):
    return (a[1] >= b[0]) and (b[1] >= a[0])

def is_intersect(A, B):
    for i in range(len(A)):
        for a in A[i]:
            for b in B[i]:
                if __is_intersect(a, b):
                    print('%s & %s is intersect' % (a,b))
                    return True
    return False

def _generateSubgraphs(vertex_list, adjacency_list):
    subgraphs = []
    freeVertices = list(np.arange(len(vertex_list)))
    while freeVertices:
        freeVertex = freeVertices.pop()
        subgraph = _constructSubgraph(freeVertex, adjacency_list, [freeVertex])
        freeVertices = [vertex for vertex in freeVertices if vertex not in subgraph]
        subgraphs.append(subgraph)
    return subgraphs

def _constructSubgraph(vertex, adjacencyList, subgraph):
    neighbors = [vertex for vertex in adjacencyList[vertex] if vertex not in subgraph]
    if (len(neighbors) == 0):
        return subgraph
    else:
        subgraph = subgraph + neighbors
        for vertex in neighbors:
            subgraph = _constructSubgraph(vertex, adjacencyList, subgraph)
        return subgraph

def _incumb(vertexWeight, adjacencyList):
    N = len(vertexWeight)

    X = np.zeros(N, dtype=bool)
    for i in range(N):
        if (len(adjacencyList[i]) == 0):
            X[i] = True

    Z = np.zeros(N)
    for i in range(N):
        Z[i] = vertexWeight[i] - np.sum(vertexWeight[list(adjacencyList[i])])

    freeVertices = np.where(X == 0)[0]
    while True:
        if len(freeVertices) == 0:
            break;
        imin = freeVertices[np.argmax(Z[freeVertices])]
        X[imin] = True
        freeVertices = freeVertices[freeVertices != imin]
        X[adjacencyList[imin]] = False
        freeVertices = freeVertices[~np.isin(freeVertices, adjacencyList[imin])]
        for i in freeVertices:
            Z[i] = vertexWeight[i] - np.sum(vertexWeight[np.intersect1d(freeVertices, adjacencyList[i])])
    return X

def _calculateLB(X, vertexWeight, adjacencyList, visitedVertices=[]):
    neighbors = np.array([], dtype=int)
    if (len(adjacencyList[np.where(X == 1)[0]]) > 0):
        neighbors = reduce(np.union1d, adjacencyList[np.where(X == 1)[0]])
    if (len(visitedVertices) > 0):
        neighbors = np.append(neighbors, visitedVertices[np.where(X[visitedVertices] == False)])
    neighbors = np.unique(neighbors)
    neighbors = np.array(neighbors, dtype=int)
    wj = np.sum(vertexWeight[neighbors])
    return -1 * (np.sum(vertexWeight) - wj)

def _BBND(vertexWeight, adjacencyList, LB, OPT_X):
    N = len(vertexWeight)
    X = np.zeros(N)
    X[:] = np.nan
    visitedVertices = np.array([], dtype=int)
    OPT = np.sum(vertexWeight[OPT_X == 1])
    prob = {'X': [], 'visitedVertices': []}
    sub_probs = []

    while True:
        if (np.sum(np.isnan(X)) == 0):
            if (np.sum(vertexWeight[np.where(X == 1)[0]]) > OPT):
                OPT = np.sum(vertexWeight[np.where(X == 1)[0]])
                OPT_X = X
            if (len(sub_probs) > 0):
                prob = sub_probs.pop()
                X = prob['X']
                visitedVertices = prob['visitedVertices']
            else:
                break

        for i in range(N):
            if (~np.any(X[list(adjacencyList[i])])):
                X[i] = 1
                if (not i in visitedVertices):
                    visitedVertices = np.append(visitedVertices, i)

        Z = np.zeros(N)
        for i in range(N):
            Z[i] = vertexWeight[i] - np.sum(vertexWeight[list(adjacencyList[i])])
        if (len(visitedVertices) > 0):
            Z[visitedVertices] = np.inf
        imin = np.argmin(Z)

        visitedVertices = np.append(visitedVertices, imin)

        X[imin] = 0
        LB0 = _calculateLB(X, vertexWeight, adjacencyList, visitedVertices)

        X[imin] = 1
        LB1 = _calculateLB(X, vertexWeight, adjacencyList, visitedVertices)

        if (LB0 < LB1):
            if (LB1 < LB):
                X[imin] = 1
                prob['X'] = X.copy()
                prob['visitedVertices'] = visitedVertices.copy()

                prob['X'][list(adjacencyList[imin])] = 0
                neighbors = adjacencyList[imin]
                for i in neighbors:
                    if (not i in prob['visitedVertices']):
                        prob['visitedVertices'] = np.append(prob['visitedVertices'], i)
                if (np.sum(np.isnan(prob['X'])) < 0):
                    sub_probs.append(prob.copy())

            X[imin] = 0
        else:
            if (LB0 < LB):
                X[imin] = 0
                prob['X'] = X.copy()
                prob['visitedVertices'] = visitedVertices.copy()
                if (np.sum(np.isnan(prob['X'])) < 0):
                    sub_probs.append(prob.copy())
            X[imin] = 1
            X[list(adjacencyList[imin])] = 0
            neighbors = adjacencyList[imin]
            for i in neighbors:
                if (not i in visitedVertices):
                    visitedVertices = np.append(visitedVertices, i)
    return OPT_X


def MWIS(vertexWeight, adjacencyList):
    '''
    :param vertexWeight: List of real-valued vertex weight
    :param adjacencyList: List of adjacency vertices
    :return: Maximum sum of weights of the independent set
    :Note:
        This is the implementation of the follow publication:
        Pardalos, P. M., & Desai, N. (1991). An algorithm for finding a maximum weighted independent set in an arbitrary graph.
        International Journal of Computer Mathematics, 38(3-4), 163-175.
    '''
    X = _incumb(vertexWeight, adjacencyList)
    LB = _calculateLB(X, vertexWeight, adjacencyList)
    return _BBND(vertexWeight, adjacencyList, LB, X)

'''
        =================================================================
                                Main Program
        =================================================================
'''

threshold = 0.2
min_support = 2
min_method = 15
max_gap = 2

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

    # Build ID List
    id_list = IdList()
    id_list.max_gap = max_gap
    id_list.build_list(data)

    # Find Closed Sequential Pattern
    Z = []  # Maximum Sequential Pattern

    # Generate and sort search_space by support
    search_space = np.argsort(list(id_list.phase_support.values()))[::-1]

    # Ignore Ids less than min_support support
    search_space = list(search_space[:np.sum(np.array(list(id_list.phase_support.values())) >= min_support)])

    while len(search_space) > 0:
        que_idx = search_space.pop(0)
        pattern = id_list.extend_pattern(que_idx, Z)

        # Check if pattern satisfy minimum number of methods
        no_of_methods = np.sum([len(id_list.ids[x]) for x in pattern[0]])
        if no_of_methods < min_method:
            continue

        # Pattern is Valid and added to Z
        Z.append(pattern)

        # Check if search space can be reduced
        pattern_support = pattern[3]
        for candidate in pattern[0]:
            if pattern_support == id_list.phase_support[candidate]:
                if candidate in search_space:
                    search_space.remove(candidate)
            else:
                break


    print('Number of Raw Patterns: %d' % len(Z))
    # Keep closed pattern only
    for i in range(len(Z)-1, 0, -1):
        for j in range(len(Z)):
            if i == j:
                continue
            # Z[i] and Z[j] have the same support and Z[i] is a subsequence of Z[j]
            if (Z[i][3] == Z[j][3]) & (is_subsequence(Z[i][0], Z[j][0])):
                print('Z[%d] is a subsequence of Z[%d]: (%s) and (%s)' % (i, j, Z[i][0], Z[j][0]))
                # Delete Z[i]
                del Z[i]
                break

    print('Number of Closed Patterns: %d' % len(Z))

    # Format Closed Pattern as (Pattern, Positions, Support)
    closed_patterns = []
    for z in Z:
        positions = []
        # For each trace
        for i in range(id_list.n_traces):
            positions.append(list(zip(z[1][i], z[2][i])))
        closed_patterns.append((z[0], positions, z[3]))
    closed_patterns = np.array(closed_patterns, dtype=object)

    # Vertex list contains the weight of the phase
    # Edge list contains relationship among phases if two phases are overlapped
    vertex_list = []
    edge_list = []
    for i in range(len(closed_patterns)):
        # Weight is defined as the number_of_method * support_of_phase
        vertex_list.append(len(closed_patterns[i][0]) * closed_patterns[i][2])
        for j in range(i+1, len(closed_patterns)):
            if is_intersect(closed_patterns[i][1], closed_patterns[j][1]):
                print('%d and %d are overlapped' % (i, j))
                edge_list.append((i,j))
                edge_list.append((j,i))
                break
    vertex_list = np.array(vertex_list)
    edge_list = np.array(edge_list)

    adjacency_list = [[] for __ in vertex_list]
    for edge in edge_list:
        adjacency_list[edge[0]].append(edge[1])
    adjacency_list = np.array(adjacency_list, dtype=object)

    subgraphs = _generateSubgraphs(vertex_list, adjacency_list)

    solution = np.zeros(len(vertex_list), dtype=bool)
    for subgraph in subgraphs:
        vl = np.array(copy.deepcopy(vertex_list[subgraph]))
        al = np.array(copy.deepcopy(adjacency_list[subgraph]))
        for i in range(len(al)):
            for j in range(len(al[i])):
                al[i][j] = np.where(subgraph == al[i][j])[0][0]
        OPT_X = MWIS(vl, al)
        solution[subgraph] = OPT_X

    patterns = closed_patterns[solution]

# Print Pattern Result
for p in patterns:
    print('Pattern: %s' % p[0])
    print('Support: %d' % p[2])
    print('Locations:')
    for i in range(len(p[1])):
        print('  Trace %d: %s' % (i, p[1][i]))

import random
sorted(random.sample(range(1,50), 6))