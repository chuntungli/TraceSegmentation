import os
import copy
import math
import random
import pickle
import numpy as np
import itertools

seed = 42

'''
    Parameters for generating patterns
'''

# Number of patterns
patternUniqueAmount = 20

# Distribution of number of phases in patterns
patternNum_mu = 10
patternNum_sigma = 3

# Number of unique methods
methodUniqueAmount = 9999

# Distribution of method in phases ( Number of Methods in Phases)
methodNumInPhase_mu = 10
methodNumInPhase_sigma = 3

# Number of unique phases
phaseUniqueAmount = (methodUniqueAmount // methodNumInPhase_mu) - 1

# Distribution of repeating Phases
phaseRep_mu = 1
phaseRep_sigma = 0

'''
    Parameters for generating sequences
'''

# Distribution of number of patterns repeated
patternRep_mu = 6
patternRep_sigma = 2

# Distribution of number of patterns in sequences
patternInSeq_mu = 6
patternInSeq_sigma = 2

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
#     patternSet = copy.deepcopy(patternSet_phaseId)
    patternSet = patternSet_phaseID
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

def sequenceDB_and_dict(patternSet,pattern_temp,patternInSeq_mu,patternInSeq_sigma,sequence_in_db,seed):
    ## Randomize pattern pool order
    random.seed(seed)
    # patternOrder = random.sample(range(0,patternRepInt.sum()),patternRepInt.sum())

    ## Randomize number of patterns in sequence by Norm dist
    np.random.seed(seed)
#     seqNormNum = np.random.normal(patternInSeq_mu,patternInSeq_sigma,patternUniqueAmount)
    seqNormNum = np.random.normal(patternInSeq_mu,patternInSeq_sigma,sequence_in_db)
    seqNormInt = np.round(seqNormNum).astype(int)

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
        seqDB.append(list(itertools.chain.from_iterable(seq_pattern)))
    return (pattern_dict, seqDB, seqDB_list)

method = method_pool(methodUniqueAmount, seed)
phase, phase_norm_int = phase_pool(method, methodNumInPhase_mu, methodNumInPhase_sigma, phaseUniqueAmount, seed)
pattern = pattern_pool_phase_id(phase, phase_norm_int, patternNum_mu, patternNum_sigma, patternUniqueAmount, seed)
pattern_set, pattern_no  = pattern_pool(pattern, phase, patternRep_mu, patternRep_sigma, len(pattern), seed)
sequence_number_in_db = math.floor(len(pattern_no)/patternInSeq_mu)
pattern_dictionary, sequence_database, sequence_database_list =  sequenceDB_and_dict(pattern_set,pattern_no,patternInSeq_mu,patternInSeq_sigma,sequence_number_in_db,seed)

# Print seqDB
for i in range(len(sequence_database)):
    print('Trace %d: %s' % (i, sequence_database[i]))

# Print Pattern Result
for key in pattern_dictionary:
    print('Pattern: %s' % pattern_dictionary[key]['pattern'])
    print('Positions:')
    for i in range(len(pattern_dictionary[key]['position'])):
        print('  Trace %d: %s' % (i, pattern_dictionary[key]['position'][i]))

# Write with pickle
folder = 'components/synthetic'
if not os.path.exists(folder):
    os.makedirs(folder)
for i in range(len(sequence_database)):
    pickle.dump(sequence_database[i], open('%s/seq_%d.p' % (folder, i), "wb"))

# Write with pickle
groundtruths = list(pattern_dictionary.values())
folder = 'data/synthetic'
if not os.path.exists(folder):
    os.makedirs(folder)
pickle.dump(groundtruths, open('%s/groundtruths.p' % folder, "wb"))