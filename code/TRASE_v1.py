
import numpy as np
from functools import reduce

# ====================================================================
#                              ID List
# ====================================================================
class IdList:

    class Pattern:
        def __init__(self):
            self.pattern = np.empty(0)
            self.l_loc = np.empty(0)
            self.r_loc = np.empty(0)
            self.support = 0

        def __init__(self, pattern, l_loc, r_loc, support):
            self.pattern = pattern
            self.l_loc = l_loc
            self.r_loc = r_loc
            self.support = support

        def __eq__(self, other):
            return self.pattern == other.pattern

        def __hash__(self):
            return hash(str(self.pattern))

    def __init__(self):
        self.__reset()

    def __reset(self):
        self.max_gap = 1
        self.min_support = 0.3
        self.n_traces = 0           # Number of traces
        self.ids = []               # Preliminary Phase IDs
        self.phase_support = {}     # Support of each phase
        self.phase_size = {}        # Size of each phase
        self.idList = {}            # Actual Data Structure
                                    #           ID
                                    # tid   |   Occurence
        self.XMap = {}              # eXtention-Map indicate pairwise extention relationship

        self.pruned = 0
        self.explored = 0

    def __add_phase(self, phase):
        if phase in self.ids:
            return

        self.ids.append(phase)
        idx = self.ids.index(phase)
        self.phase_support[idx] = 0
        self.phase_size[idx] = len(phase)
        self.idList[idx] = np.full(self.n_traces, np.nan)
        self.XMap[idx] = set()

    # Construct the Id List given the list of preliminary phases as list of sets: [{phase_1}, {phase_2}, ..., {phase_n}]
    def build_list(self, traces, max_gap = 1, min_sup = 0.3):
        self.__reset()

        self.max_gap = max_gap
        self.min_support = min_sup

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
                self.idList[phase_idx][trace_idx] = event_idx
                # self.phase_support[phase_idx] += 1

                # Update Extention Map
                for gap in range(1, self.max_gap + 1):
                    # Extention
                    idx = event_idx + gap
                    if idx < len(trace):
                        # Check if phase already exists in ids
                        if trace[idx] not in self.ids:
                            self.__add_phase(trace[idx])
                        self.XMap[phase_idx].add(self.ids.index(trace[idx]))
            trace_idx += 1

        # Clean Extention Map from removing self-extention
        # for i in range(len(self.ids)):
        #     if i in self.XMap[i]:
        #         self.XMap[i].remove(i)

        # Compute Support of phases
        for i in range(len(self.ids)):
            self.phase_support[i] = np.sum(~np.isnan(self.idList[i])) / self.n_traces

    # Extend pattern by depth-first-search manner
    def __extend_pattern(self, closed_patterns, pattern, l_loc, r_loc, que_support, best_score):
        print('\rExplored: %d\tPruned: %d\tExploring: %.2f - %s' % (self.explored, self.pruned, que_support, pattern), end='')
        self.explored += 1

        is_closed = True
        candidates = list(self.XMap[pattern[-1]])

        # Compute matches for each
        cand_supports = {}
        cand_matches = {}
        for candidate in candidates:
            # Matches
            diff = self.idList[candidate] - r_loc
            with np.errstate(invalid='ignore'):
                matches = (diff > 0) & (diff <= self.max_gap)
            cand_matches[candidate] = matches

            # Support is defined as sup_(q) * |q|
            support = (sum(matches) / self.n_traces)
            # support = (sum(matches) / self.n_traces) * np.log(pattern_size + self.phase_size[candidate])
            # support = (sum(matches) / self.n_traces) * np.log(len(pattern) + 1)
            cand_supports[candidate] = support

            if support >= que_support:
                is_closed = False

        if is_closed:
            # score = que_support * len(pattern)
            score = que_support * np.log(sum([len(self.ids[x]) for x in pattern]))
            if score > best_score:
                best_score = score
            else:
                self.pruned += 1
                return

        # Sort search space by support
        ordered_candidates = sorted(cand_supports, key=cand_supports.get)
        for candidate in ordered_candidates:
            extended_pattern = pattern + [candidate]
            matches = cand_matches[candidate]
            support = cand_supports[candidate]

            if support <= self.min_support:
                continue

            # Check if candidate sequence is subsequence of found patterns
            skip_candidate = False
            for p in closed_patterns:
                if (p.support >= que_support) & is_subsequence(extended_pattern, p.pattern):
                    skip_candidate = True
                    break
            if skip_candidate:
                continue

            xl_loc = np.full(self.n_traces, np.nan)
            xr_loc = np.full(self.n_traces, np.nan)
            xl_loc[matches] = l_loc[matches]
            xr_loc[matches] = self.idList[candidate][matches]
            self.__extend_pattern(closed_patterns, extended_pattern, xl_loc, xr_loc, support, best_score)

        if is_closed:
            should_add = True
            for p in closed_patterns:
                if (p.support >= que_support) & (is_subsequence(pattern, p.pattern)):
                    should_add = False
                    break
            if should_add:
                closed_patterns.add(self.Pattern(pattern, l_loc, r_loc, que_support))

    # Find maximum sequential pattern given the starting element
    # Returns: Pattern, l_loc, r_loc, support
    def extend_pattern(self, que_idx, Z):
        closed_patterns = set()
        que_idlist = self.idList[que_idx]
        que_support = self.phase_support[que_idx]

        for z in Z:
            if (z.support >= que_support) & (que_idx in z.pattern):
                return []

        self.__extend_pattern(closed_patterns, [que_idx], que_idlist, que_idlist, que_support, 0)
        return list(closed_patterns)

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
    # For strictly subsequences (a_i = b_j, a_i+1 = b_j+1, ...)
    # l_q = len(query)
    # l_b = len(base)
    # if l_q > l_b:
    #     return False
    # for i in range(l_b):
    #     if base[i:i + l_q] == query:
    #         return True
    # return False

    # For normal subsequences
    m = len(query)
    n = len(base)
    i = j = 0

    while j < m and i < n:
        if query[j] == base[i]:
            j = j + 1
        i = i + 1

    # If all characters of str1 matched, then j is equal to m
    return j == m

def is_intersect(A, B):
    with np.errstate(invalid='ignore'):
        return any((A.r_loc >= B.l_loc) & (B.r_loc >= A.l_loc))

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