

import os
import copy
import math
import random
import pickle
import shutil
import numpy as np
import itertools


# Total number of sequence required in DB
# pattern number divided by pattern in sequence
## Random create method pool using elements from 1 to 10000 with specified amount
def method_pool(methodUniqueAmount, seed):
    random.seed(seed)
    np.random.seed(seed)
    methodPool = random.sample(range(1,methodUniqueAmount+1),methodUniqueAmount)
    return methodPool

def phase_pool(methodPool, methodNumInPhase_mu, methodNumInPhase_sigma, phaseUniqueAmount,seed):
    np.random.seed(seed)
    methodNormNum = np.random.normal(methodNumInPhase_mu,methodNumInPhase_sigma,phaseUniqueAmount)
    methodNormInt = np.round(methodNormNum).astype(int)
    ## Create phaseSet with all the unique phase
    phaseSet = []
    for i in range(0,len(methodNormInt)):
        if i == 0:
            methodBegin = 0
            methodEnd = methodNormInt[i]
        if i > 0:
            methodBegin = methodSum
            methodEnd = methodSum + methodNormInt[i] ## the ith elements

        methodSum = methodEnd
        phase_i = set(methodPool[methodBegin:methodEnd])
        phaseSet.append(phase_i)

    ## Random create the phase repeated times and number
    phaseNormNum = np.random.normal(phaseRep_mu,phaseRep_sigma,len(methodNormInt))
    phaseNormInt = np.round(phaseNormNum).astype(int)
    ## Create phasePool with repeated unique phases
    phasePool = []
    for j in range(0,len(phaseNormInt)):
        for k in range(0,phaseNormInt[j]):
            phasePool.append(phaseSet[j])
    return (phasePool, phaseNormInt)

def pattern_pool_phase_id(phasePool, phaseNormInt, patternNum_mu, patternNum_sigma, patternUniqueAmount,seed):
    random.seed(seed)
    phaseOrder = random.sample(range(0,phaseNormInt.sum()),phaseNormInt.sum())
    np.random.seed(seed)
    patternNormNum = np.random.normal(patternNum_mu,patternNum_sigma,patternUniqueAmount)
    patternNormInt = np.round(patternNormNum).astype(int)

    # Create pattern set in terms of phase id
    patternSet_phaseId = []
    # pattern = []
    patternSum = 0
    for i in range(0,len(patternNormInt)):
        if i == 0:
            patternBegin = 0
            patternEnd = patternNormInt[i]

        if i > 0:
            patternBegin = patternSum
            patternEnd = patternSum + patternNormInt[i] ## the ith elements

        patternSum = patternEnd
        pattern_i = phaseOrder[patternBegin:patternEnd]
        patternSet_phaseId.append(pattern_i)
    return patternSet_phaseId


def pattern_pool(patternSet_phaseID, phasePool, patternRep_mu, patternRep_sigma, length_of_patternSet, seed):
    ## Represent pattern set in terms of phase
    patternSet = copy.deepcopy(patternSet_phaseID)
    # patternSet = patternSet_phaseID
    for i, row in enumerate(patternSet):
        for j,cell in enumerate(row):
            location = patternSet[i][j]
            patternSet[i][j] = phasePool[location]

    ## Create pattern pool by repeating phases
    patternPool = []
    pattern_temp = []
    np.random.seed(seed)
    patternRepNum = np.random.normal(patternRep_mu,patternRep_sigma,len(patternSet))
    patternRepInt = np.round(patternRepNum).astype(int)
    for j in range(0,len(patternRepInt)):
        for k in range(0,patternRepInt[j]):
            patternPool.append(patternSet[j])
            pattern_temp.append(j)
    return (patternSet, pattern_temp)

# pattern_set,pattern_no,patternInSeq_mu,patternInSeq_sigma,phase,sequence_number_in_db,seed
def sequenceDB_and_dict(patternSet,pattern_temp,patternInSeq_mu,patternInSeq_sigma,phase_set,sequence_in_db,seed,noise_ratio=0):
    ## Randomize pattern pool order
    random.seed(seed)
    # patternOrder = random.sample(range(0,patternRepInt.sum()),patternRepInt.sum())

    ## Randomize number of patterns in sequence by Norm dist
    np.random.seed(seed)
    seqNormNum = np.random.normal(patternInSeq_mu,patternInSeq_sigma,sequence_in_db-1)
    seqNormInt = list(np.round(seqNormNum).astype(int))
    diff = len(pattern_temp) - sum(seqNormInt)
    if diff < 0:
        pattern_temp += random.choices(range(len(patternSet)), k=-diff)
    if diff > 0:
        seqNormInt.append(len(pattern_temp) - sum(seqNormInt))


    ## Create dictionary for groundtruth
    pattern_index = list(range(0, len(patternSet)))
    pattern_index_str = map(str, pattern_index)

    pattern_key = []
    for i in range(0,len(pattern_index)):
        pattern_key.append("pattern")

    pattern_dict = {index:{key:value} for (index, key, value) in zip(pattern_index_str, pattern_key, patternSet)}
    # Create position list
    position_init = []
    for j in range(0,len(seqNormInt)):
        position_init.append([])

    for i in pattern_dict:
        pattern_dict[i]['position'] = copy.deepcopy(position_init)

    ##  Create sequence DB and append the corresponding location to dict
    seqDB_patternId = []
    sequence = []
    seq_temp = []
    seqSum = 0
    # for i in range(0,len(patternNormInt)):
    for i, row in enumerate(seqNormInt):
        candidates = [pattern_temp.pop(random.randrange(len(pattern_temp))) for _ in range(seqNormInt[i])]
        seqDB_patternId.append(candidates)
    #     print('====Sequence: ',i,'====')
    #     print('row: ',row)
    #     print(seqDB_patternId[i])
        for j in range(0,row):
            seqPattern = seqDB_patternId[i][j]
    #         print(seqPattern)
    #         print('--item: ',j,'--')
            if j == 0:
                begin = 0
                end = len(patternSet[seqPattern])
                coordinate = (begin,end)

            if j > 0:
                begin = accum_sum
                end = len(patternSet[seqPattern]) + begin
                coordinate = (begin,end)

            accum_sum = end
            pattern_dict[str(seqPattern)]['position'][i].append(coordinate)

    ## Create the sequence DB in terms of methods
    seqDB_list = copy.deepcopy(seqDB_patternId)
    for i, row in enumerate(seqDB_list):
        for j,cell in enumerate(row):
            location = seqDB_list[i][j]
            seqDB_list[i][j] = patternSet[location]

    seqDB = []
    for i, row in enumerate(seqDB_list):
        seq_pattern = seqDB_list[i]
        seq = list(itertools.chain.from_iterable(seq_pattern))
        # Add Noise to seq
        noise_flag = np.random.rand(len(seq)) < noise_ratio
        for idx in sorted(np.where(noise_flag == True)[0], reverse=True):
            seq.insert(idx, np.random.choice(phase_set))
        seqDB.append(seq)

    return (pattern_dict, seqDB, seqDB_list)

def noise_factor():
    return 0.05 + (np.random.rand() * 0.1)


'''
    ================================================================================
                                Generate DB Logics
    ================================================================================
'''

seed = 42

folder = 'components/synthetic'
if os.path.exists(folder):
    shutil.rmtree(folder)
folder = 'groundtruth/synthetic'
if os.path.exists(folder):
    shutil.rmtree(folder)

# Global parameters

# Number of unique methods
methodUniqueAmount = 9999999

# Distribution of method in phases ( Number of Methods in Phases)
methodNumInPhase_mu = 20
methodNumInPhase_sigma = methodNumInPhase_mu * 0.1

# Distribution of repeating Phases
phaseRep_mu = 1
phaseRep_sigma = 0

n_fold = 5

'''
    =====================================================================
            Generate sequences with different length of patterns
    =====================================================================
'''

# Number of patterns
patternUniqueAmount = 20

# Distribution of number of patterns in sequences
patternInSeq_mu = 20
patternInSeq_sigma = patternInSeq_mu * 0.1

# Distribution of number of patterns repeated
patternRep_mu = 20
patternRep_sigma = patternRep_mu * 0.1

# for pat_len in [5, 10, 25, 50, 100, 250, 500, 1000]:
for pat_len in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:

    # Distribution of number of phases in patterns
    patternNum_mu = pat_len
    patternNum_sigma = pat_len * 0.1

    # Number of unique phases
    phaseUniqueAmount = int(patternUniqueAmount * (patternNum_mu + 1))

    method = method_pool(methodUniqueAmount, seed)
    phase, phase_norm_int = phase_pool(method, methodNumInPhase_mu, methodNumInPhase_sigma, phaseUniqueAmount, seed)
    pattern = pattern_pool_phase_id(phase, phase_norm_int, patternNum_mu, patternNum_sigma, patternUniqueAmount, seed)

    for fold in range(n_fold):

        '''
            Parameters for generating sequences
        '''
        pattern_set, pattern_no = pattern_pool(pattern, phase, patternRep_mu, patternRep_sigma, len(pattern), fold)
        sequence_number_in_db = math.floor(len(pattern_no) / patternInSeq_mu)

        pattern_dictionary, sequence_database, sequence_database_list = \
            sequenceDB_and_dict(pattern_set, pattern_no, patternInSeq_mu, patternInSeq_sigma, phase, sequence_number_in_db, fold, noise_factor())

        # Write with pickle
        folder = 'components/synthetic/pat_len/%03d/%d' % (pat_len, fold)
        os.makedirs(folder)
        for i in range(len(sequence_database)):
            pickle.dump(sequence_database[i], open('%s/seq_%d.p' % (folder, i), "wb"))
        folder = 'groundtruth/synthetic/pat_len/%03d/%d' % (pat_len, fold)
        os.makedirs(folder)
        pickle.dump(pattern_dictionary, open('%s/groundtruth.p' % folder, 'wb'))



'''
    =====================================================================
        Generate sequences with different number of patterns (seq_len)
    =====================================================================
'''

# Distribution of number of phases in patterns
patternNum_mu = 20
patternNum_sigma = patternNum_mu * 0.1

# Distribution of number of patterns repeated
patternRep_mu = 20
patternRep_sigma = patternRep_mu * 0.1

method = method_pool(methodUniqueAmount, seed)
phase, phase_norm_int = phase_pool(method, methodNumInPhase_mu, methodNumInPhase_sigma, phaseUniqueAmount, seed)
pattern = pattern_pool_phase_id(phase, phase_norm_int, patternNum_mu, patternNum_sigma, patternUniqueAmount, seed)

for seq_len in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:

    # Number of patterns
    patternUniqueAmount = seq_len

    # Distribution of number of patterns in sequences
    patternInSeq_mu = patternUniqueAmount
    patternInSeq_sigma = 0.1 * patternInSeq_mu

    for fold in range(n_fold):

        '''
            Parameters for generating sequences
        '''
        pattern_set, pattern_no = pattern_pool(pattern, phase, patternRep_mu, patternRep_sigma, len(pattern), fold)
        sequence_number_in_db = math.floor(len(pattern_no) / patternInSeq_mu)
        pattern_dictionary, sequence_database, sequence_database_list = \
            sequenceDB_and_dict(pattern_set, pattern_no, patternInSeq_mu, patternInSeq_sigma, phase, sequence_number_in_db, fold, noise_factor())

        # Write with pickle
        folder = 'components/synthetic/seq_len/%03d/%d' % (seq_len, fold)
        os.makedirs(folder)
        for i in range(len(sequence_database)):
            pickle.dump(sequence_database[i], open('%s/seq_%d.p' % (folder, i), "wb"))
        folder = 'groundtruth/synthetic/seq_len/%03d/%d' % (seq_len, fold)
        os.makedirs(folder)
        pickle.dump(pattern_dictionary, open('%s/groundtruth.p' % folder, 'wb'))



'''
    =====================================================================
            Generate sequences with different number of sequences
    =====================================================================
'''

# Number of patterns
patternUniqueAmount = 20

# Distribution of number of phases in patterns
patternNum_mu = 20
patternNum_sigma = patternNum_mu * 0.1

# Number of unique phases
phaseUniqueAmount = int(patternUniqueAmount * (patternNum_mu + 1))

# Distribution of number of patterns in sequences
patternInSeq_mu = 20
patternInSeq_sigma = patternInSeq_mu * 0.1

method = method_pool(methodUniqueAmount, seed)
phase, phase_norm_int = phase_pool(method, methodNumInPhase_mu, methodNumInPhase_sigma, phaseUniqueAmount, seed)
pattern = pattern_pool_phase_id(phase, phase_norm_int, patternNum_mu, patternNum_sigma, patternUniqueAmount, seed)

for n_seq in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:

    # Distribution of number of patterns repeated
    patternRep_mu = n_seq
    patternRep_sigma = n_seq * 0.1

    for fold in range(n_fold):

        '''
            Parameters for generating sequences
        '''
        pattern_set, pattern_no = pattern_pool(pattern, phase, patternRep_mu, patternRep_sigma, len(pattern), fold)
        sequence_number_in_db = math.floor(len(pattern_no) / patternInSeq_mu)
        pattern_dictionary, sequence_database, sequence_database_list = \
            sequenceDB_and_dict(pattern_set, pattern_no, patternInSeq_mu, patternInSeq_sigma, phase, sequence_number_in_db, fold, noise_factor())

        # Write with pickle
        folder = 'components/synthetic/n_seq/%03d/%d' % (n_seq, fold)
        os.makedirs(folder)
        for i in range(len(sequence_database)):
            pickle.dump(sequence_database[i], open('%s/seq_%d.p' % (folder, i), "wb"))
        folder = 'groundtruth/synthetic/n_seq/%03d/%d' % (n_seq, fold)
        os.makedirs(folder)
        pickle.dump(pattern_dictionary, open('%s/groundtruth.p' % folder, 'wb'))



'''
    =====================================================================
            Generate sequences for performance evaluation
    =====================================================================
'''

# # Number of patterns
# patternUniqueAmount = 20
#
# # Distribution of number of phases in patterns
# patternNum_mu = 20
# patternNum_sigma = patternNum_mu * 0.1
#
# # Number of unique phases
# phaseUniqueAmount = int(patternUniqueAmount * (patternNum_mu + 1))
#
# # Distribution of number of patterns in sequences
# patternInSeq_mu = 20
# patternInSeq_sigma = patternInSeq_mu * 0.1
#
# # Distribution of number of patterns repeated
# patternRep_mu = 20
# patternRep_sigma = patternRep_mu * 0.1
#
#
# for fold in range(n_fold):
#
#     method = method_pool(methodUniqueAmount, fold)
#     phase, phase_norm_int = phase_pool(method, methodNumInPhase_mu, methodNumInPhase_sigma, phaseUniqueAmount, fold)
#     pattern = pattern_pool_phase_id(phase, phase_norm_int, patternNum_mu, patternNum_sigma, patternUniqueAmount, fold)
#
#     '''
#         Parameters for generating sequences
#     '''
#     pattern_set, pattern_no = pattern_pool(pattern, phase, patternRep_mu, patternRep_sigma, len(pattern), fold)
#     sequence_number_in_db = math.floor(len(pattern_no) / patternInSeq_mu)
#     pattern_dictionary, sequence_database, sequence_database_list = \
#         sequenceDB_and_dict(pattern_set, pattern_no, patternInSeq_mu, patternInSeq_sigma, phase, sequence_number_in_db, fold, noise_factor())
#
#     # Write with pickle
#     folder = 'components/synthetic/performance/%d' % fold
#     os.makedirs(folder)
#     for i in range(len(sequence_database)):
#         pickle.dump(sequence_database[i], open('%s/seq_%d.p' % (folder, i), "wb"))
#     folder = 'groundtruth/synthetic/performance/%d' % fold
#     os.makedirs(folder)
#     pickle.dump(pattern_dictionary, open('%s/groundtruth.p' % folder, 'wb'))




# # Print seqDB
# for i in range(len(sequence_database)):
#     print('Trace %d: %s' % (i, len(sequence_database[i])))
#
# # Print Pattern Result
# for key in pattern_dictionary:
#     print('Pattern: %s' % pattern_dictionary[key]['pattern'])
#     print('Positions:')
#     for i in range(len(pattern_dictionary[key]['position'])):
#         print('  Trace %d: %s' % (i, pattern_dictionary[key]['position'][i]))
