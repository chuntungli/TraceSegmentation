
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scrimp import scrimp_plus_plus as scrimp
from spmf import Spmf

from itertools import groupby

def distance(com1, com2):
    return (len(com1 - com2) + len(com2 - com1)) / (len(com1) + len(com2))

def computeDP(que, seq):
    _seq_len = len(seq)
    _dp = np.zeros(_seq_len)
    for _j in range(_seq_len):
        _dp[_j] = distance(que, seq[_j])
    return _dp

def computeMP(seq):
    _n = len(seq)
    _mp = np.empty(_n)
    _pi = np.empty(_n, dtype=int)
    _mp[:] = _pi[:] = np.inf
    for _j in range(_n):
        # Compute DP
        _que = seq[_j]
        _dp = computeDP(_que, seq)
        _dp[_j] = np.inf

        # Update MP
        _update_pos = (_dp < _mp)
        _mp[_update_pos] = _dp[_update_pos]
        _pi[_update_pos] = _j
    return (_mp, _pi)

threshold = 0.05

archive_url = os.getcwd() + '/components'
traces = os.listdir(archive_url)
for trace in traces:
    if trace.startswith('.'):
        continue
    components = pickle.load(open('%s/%s' % (archive_url, trace), "rb"))

    # Number of Raw Preliminary Phrases
    print('Number of Raw Preliminary Phrases: %d' % len(components))

    # Remove consecutive duplicate items (Run-length Encoding)
    components = [i[0] for i in groupby(components)]
    print('Number of Preliminary Phrases after Run-length Encoding: %d' % len(components))

    # Compute Phase Sizes
    phase_sizes = np.array([len(e) for e in components], dtype=float)
    # plt.plot(phase_sizes);plt.show()

    #
    # mp, ip = scrimp(phase_sizes, 5, random_state=42)
    # plt.plot(mp);plt.show()

    # Write data to file
    f = open("%s.txt" % trace, "w+")
    for event in components:
        for item in event:
            f.write("%s " % item)
        f.write("-1 ")
    f.write("-2\r\n")
    f.close()

    spmf = Spmf("VMSP", input_filename="com.angkorworld.memo.txt", output_filename="output.txt", arguments=['50%', 10, 1, False])
    # spmf = Spmf("SPAM", input_filename="com.angkorworld.memo.txt", output_filename="output.txt", arguments=[0.5, 2, 50, 1, False])
    # spmf = Spmf("VMSP", input_filename="contextPrefixSpan.txt", output_filename="output.txt", arguments=[0.5])
    spmf.run()
    print(spmf.to_pandas_dataframe(pickle=True))

    import random
    random.sample(range(1,50), 6)

    # Comput Matrix Profile
    mp, pi = computeMP(components)

    # Remove obvious Outliers (e.g. distance of nearest neighbor = 1)
    outliers = np.where(mp >= 0.5)[0][::-1]
    components = [i for j, i in enumerate(components) if j not in outliers]
    mp = np.delete(mp, outliers)
    pi = np.delete(pi, outliers)
    for outlier in outliers:
        pi[pi >= outlier] -= 1

    # Algorithm 1
    while not all(np.isinf(mp)):
        # Find candidate components (minimum distance & maximum phase size)
        min_mp = np.nanmin(mp)
        phase_sizes = np.array([len(e) for e in components], dtype=float)
        valid_mask = (mp == min_mp)
        if not all(valid_mask):
            phase_sizes[~valid_mask] = np.nan
        can_idx = np.nanargmax(phase_sizes)
        nei_idx = pi[can_idx]

        # Check expanding in both directions
        left_dist = np.inf
        right_dist = np.inf
        if (can_idx-1 >= 0) & (nei_idx-1 >= 0):
            left_dist = distance(components[can_idx-1], components[nei_idx-1])
        if (can_idx+1 < len(components)) & (nei_idx+1 < len(components)):
            right_dist = distance(components[can_idx+1], components[nei_idx+1])

        # Expand to the direction with minimum distance
        min_idx = min(can_idx, nei_idx)
        min_idx2 = max(can_idx, nei_idx)
        if left_dist <= right_dist:
            # Expand leftside
            min_idx -= 1
            min_idx2 -= 1

        c1 = components[min_idx] | components[min_idx + 1]
        c2 = components[min_idx2] | components[min_idx2 + 1]

        if distance(c1, c2) > threshold:
            mp[can_idx] = np.inf
            continue

        # =============================================
        #             Perform the merge
        # =============================================

        components[min_idx] = c1
        components[min_idx2] = c2

        # Delete the two elements
        # components = [j for i, j in enumerate(components) if i not in [min_idx + 1, min_idx2 + 1]]
        del components[min_idx2+1]
        del components[min_idx+1]
        # Remove MP and PI
        mp = np.delete(mp, [min_idx+1, min_idx2+1])
        pi = np.delete(pi, [min_idx+1, min_idx2+1])
        min_idx2 -= 1

        # Update PI
        pi[pi > min_idx] -= 1
        pi[pi > min_idx2] -= 1
        # Update MP
        update_indices = list(np.where((pi == min_idx+1) | (pi == min_idx2+1))[0]) + [min_idx, min_idx2]
        mp[update_indices] = np.inf
        for idx in update_indices:
            dp = computeDP(components[idx], components)
            dp[idx] = np.inf
            ui = (dp < mp)
            mp[ui] = dp[ui]
            pi[ui] = idx
            nn = np.nanargmin(dp)
            mp[idx] = dp[nn]
            pi[idx] = nn

    # Identify Features by clustering identified components
    distance_matrix = np.empty((len(components), len(components)))
    distance_matrix[:] = np.inf
    for i in range(len(components)):


    mp, pi = computeMP(components)
