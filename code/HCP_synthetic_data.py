import random
import numpy as np
import copy
import itertools

## Param
seed = 4

methodUniqueAmount = 999

methodNumInPhase_mu = 40
methodNumInPhase_sigma = 10

phaseUniqueAmount = 5

phaseRep_mu = 18
phaseRep_sigma = 3

patternUniqueAmount = 10

patternNum_mu = 8
patternNum_sigma = 2

patternRep_mu = 8
patternRep_sigma = 3 

patternInSeq_mu = 6
patternInSeq_sigma = 2

## Random create method pool using elements from 1 to 10000 with specified amount
random.seed(seed)
np.random.seed(seed)
methodPool = random.sample(range(1,10000),methodUniqueAmount)
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
np.random.seed(seed)
phaseNormNum = np.random.normal(phaseRep_mu,phaseRep_sigma,len(methodNormInt))
phaseNormInt = np.round(phaseNormNum).astype(int)
## Create phasePool with repeated unique phases
phasePool = []
for j in range(0,len(phaseNormInt)):
    for k in range(0,phaseNormInt[j]):
        phasePool.append(phaseSet[j])

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

## Represent pattern set in terms of phase
patternSet = copy.deepcopy(patternSet_phaseId)
for i, row in enumerate(patternSet):
#     print(i, row)
    for j,cell in enumerate(row):
        location = patternSet[i][j]
        patternSet[i][j] = phasePool[location]
        
## Create pattern pool by repeating phases
patternPool = []
np.random.seed(seed)
patternRepNum = np.random.normal(patternRep_mu,patternRep_sigma,len(patternSet))
patternRepInt = np.round(patternRepNum).astype(int)
for j in range(0,len(patternRepInt)):
    for k in range(0,patternRepInt[j]):
        patternPool.append(patternSet[j])
        
## Randomize pattern pool order
random.seed(seed)
patternOrder = random.sample(range(0,patternRepInt.sum()),patternRepInt.sum())

## Randomize pattern repeated times by Norm dist
np.random.seed(seed)
patternNormNum = np.random.normal(patternInSeq_mu,patternInSeq_sigma,10)
patternNormInt = np.round(patternNormNum).astype(int)

## Temp list for repeating pattern
temp = []
temp_list = list(range(len(patternNormInt)))
for i in range(0,len(patternNormInt)):
    temp_list[i] = np.repeat(temp_list[i], patternNormInt[i])
#     temp.append(temp_list[i])
temp.append(list(itertools.chain.from_iterable(temp_list)))
pattern_temp = temp[0]

## Create dictionary for groundtruth
pattern_index = list(range(0,len(patternSet)))
pattern_index_str = map(str, pattern_index)  

pattern_key = []
for i in range(0,len(pattern_index)):
    pattern_key.append("pattern")

pattern_dict = {index:{key:value} for (index, key, value) in zip(pattern_index_str, pattern_key, patternSet)}
# Create position list
position_init = []
for j in range(0,len(patternNormInt)):
    position_init.append([])

for i in pattern_dict:    
    pattern_dict[i]['position'] = copy.deepcopy(position_init)
    
##  Create sequence DB and append the corresponding location to dict 
random.seed(seed)
seqDB_patternId = []
sequence = []
seq_temp = []
seqSum = 0
# for i in range(0,len(patternNormInt)):
for i, row in enumerate(patternNormInt):
    seqDB_patternId.append(random.sample(pattern_temp,patternNormInt[i]))
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

print(seqDB)
# print(pattern_dict)