
import os
import pickle
import numpy as np
from itertools import groupby
from sklearn.cluster import AgglomerativeClustering

threshold = 0.3
min_support = 4

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
    def __extend_pattern(self, pattern, l_loc, r_loc, que_support):
        print('Extend Pattern: %d - %s - %s' % (que_support, r_loc, pattern))
        patterns = [(pattern, l_loc, r_loc, que_support)]

        # Check Left Extention
        # for candidate in self.LXMap[pattern[0][0]]:
        #     support = 0
        #     c_loc = []
        #     for i in range(self.n_traces):
        #         matches = self.__matching(self.idList[candidate][i], l_loc[i])
        #         if len(matches) > 0:
        #             c_loc.append([x[0] for x in matches])
        #         else:
        #             c_loc.append([])
        #         support += len(matches)
        #     if support == que_support:
        #         patterns.append(
        #             self.__extend_pattern([candidate] + pattern, c_loc, r_loc, que_support)
        #         )

        # Check Right Extention
        for candidate in self.RXMap[pattern[-1]]:
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
                    self.__extend_pattern(pattern + [candidate], l_loc, c_loc, que_support)
                )

        pattern_sizes = [len(x[0]) for x in patterns]
        return patterns[np.argmax(pattern_sizes)]

    # Find maximum sequential pattern given the starting element
    # Returns: Pattern, l_loc, r_loc, support
    def extend_pattern(self, que_idx):
        que_idlist = self.idList[que_idx]
        que_support = self.phase_support[que_idx]
        return self.__extend_pattern([que_idx], que_idlist, que_idlist, que_support)


def distance(com1, com2):
    return (len(com1 - com2) + len(com2 - com1)) / (len(com1) + len(com2))

def computeDistMat(seq):
    n_seq = len(seq)
    dist_mat = np.zeros((n_seq, n_seq))
    for i in range(n_seq-1):
        que = seq[i]
        for j in range(i+1, n_seq):
            dist_mat[i,j] = distance(que, seq[j])
    dist_mat += dist_mat.T
    return dist_mat

for app in apps:
    if app.startswith(','):
        continue

    app_folder = '%s/%s' % (archive_url, app)
    traces = os.listdir(app_folder)

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
    dist_mat = computeDistMat(unique_components)
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
    id_list.max_gap = 2
    id_list.build_list(data)

    # Find Closed Sequential Pattern
    Z = []  # Maximum Sequential Pattern

    # Generate and sort search_space by support
    search_space = np.argsort(list(id_list.phase_support.values()))[::-1]

    # Ignore Ids with only one support
    n_one = np.sum(np.array(list(id_list.phase_support.values())) <= 1)
    search_space = list(search_space[:-n_one])

    while len(search_space) > 0:
        que_idx = search_space.pop(0)
        pattern = id_list.extend_pattern(que_idx)

        Z.append(pattern)
        # Check if pattern is subsequence of another pattern
        # is_subsequence = False
        # for z in Z:
        #
        # if not is_subsequence:
        #     Z.append(pattern)

        # # Check if search space can be reduced
        pattern_support = pattern[3]
        for candidate in pattern[0]:
            if pattern_support == id_list.phase_support[candidate]:
                if candidate in search_space:
                    search_space.remove(candidate)
            else:
                break

    len(Z)