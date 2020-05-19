#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:01:30 2020

@author: mzp06256
"""

import sys
from pyspark import SparkContext
import random
import time


num_hash = 100


primes = [11273,
 11279,
 11287,
 11299,
 11311,
 11317,
 11321,
 11329,
 11351,
 11353,
 11369,
 11383,
 11393,
 11399,
 11411,
 11423,
 11437,
 11443,
 11447,
 11467,
 11471,
 11483,
 11489,
 11491,
 11497,
 11503,
 11519,
 11527,
 11549,
 11551,
 11579,
 11587,
 11593,
 11597,
 11617,
 11621,
 11633,
 11657,
 11677,
 11681,
 11689,
 11699,
 11701,
 11717,
 11719,
 11731,
 11743,
 11777,
 11779,
 11783,
 11789,
 11801,
 11807,
 11813,
 11821,
 11827,
 11831,
 11833,
 11839,
 11863,
 11867,
 11887,
 11897,
 11903,
 11909,
 11923,
 11927,
 11933,
 11939,
 11941,
 11953,
 11959,
 11969,
 11971,
 11981,
 11987,
 12007,
 12011,
 12037,
 12041,
 12043,
 12049,
 12071,
 12073,
 12097,
 12101,
 12107,
 12109,
 12113,
 12119,
 12143,
 12149,
 12157,
 12161,
 12163,
 12197,
 12203,
 12211,
 12227,
 12239]

#a = [1,3,5,7,9,11,19,27,29,31,33,37,39,41,43,47,51,53,57,59]
a = []
b = []
while len(a) != num_hash:
    num = random.randrange(1,10000) 
    if num not in a:
        a.append(num)
        
while len(b) != num_hash:
    num = random.randrange(-10000,10000)
    if num not in a and num not in b:
        b.append(num)
    
def minHash(x, m):
    # x -> list of users row ids
    global a
    ans = []
    for j in range(num_hash):
        minSoFar = sys.maxsize
#        a = random.randrange(1,m)
#        print(str(a), '\n')
#        b = random.randrange(-sys.maxsize, sys.maxsize)
        for i in range(len(x)):

#        b = random.randrange(1,m)
            minSoFar = min(minSoFar, (((a[j]*x[i] + b[j] ) % primes[j]) % m))
        ans.append(minSoFar)
    return ans
    

def bandMap(business, num_bands, num_rows):
    business_id = business[0]
    signatures = business[1]
    
    bands_sig = []
    
    for i in range(num_bands):
        rows_sig = []
        for j in range(num_rows):
            rows_sig.append(signatures[i*num_rows + j])
        bands_sig.append(((i, frozenset(rows_sig)), [business_id]))
    
    return bands_sig

def jaccardSim(x):
    cand_list = []
    jaccard_sim = []
    for cand in x[1]:
        cand_list.append(cand)
    
    
    for i in range(len(cand_list)):
        for j in range(i+1, len(cand_list)):
            pair = (cand_list[i], cand_list[j])
            users_1 = set(business_dict[cand_list[i]])
            users_2 = set(business_dict[cand_list[j]])
            
            similarity = len(users_1 & users_2)/len(users_1 | users_2)
            jaccard_sim.append((pair, similarity))
    
    return jaccard_sim

if __name__ == "__main__":
    
    start = time.time()

    args = sys.argv
    
    if len(args) == 1:
        # testing
        path_in = "data/yelp_train.csv"
        path_out = "output1.csv"
    elif len(args) != 3:
        print("Usage: ./spark-submit task1.py <input_file_name> <output_file_name>")
    else:
        path_in = args[1]
        path_out = args[2]
        
        
    sc = SparkContext('local[*]','mzma_task2')
    
    rdd = sc.textFile(path_in) # 455855 entries
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header).map(lambda x: x.split(','))

    # characteristic matrix representing sets
    users = rdd.map(lambda x: x[0]).distinct().collect()
        
    Lookup = dict()
    for i in range(len(users)):
        Lookup[users[i]] = i

    
    # business to user_id mapping    
    char_mat = rdd. \
                map(lambda x: (x[1], [Lookup[x[0]]])). \
                reduceByKey(lambda x,y: x + y)
    business_dict = char_mat.collectAsMap()
    
    m = len(users)
    
#    with open('primes.txt') as file:
#        data = file.read()
#    data.replace('\n', '')
#    primes = data.split()[:100]
#    for i in range(len(primes)):
#        primes[i] = int(primes[i])

    sig_mat = char_mat.\
                map(lambda x: (x[0], minHash(x[1], m)))
    
    num_bands = 50
    num_rows = int(num_hash/num_bands)
    
    sig_mat = sig_mat.flatMap(lambda x: bandMap(x, num_bands, num_rows))
    
    candidates = sig_mat. \
                reduceByKey(lambda x,y: x+y). \
                filter(lambda x: len(x[1])>1)
    
    sim_pairs = candidates. \
                flatMap(lambda x: jaccardSim(x)). \
                filter(lambda x: x[1] >= 0.5). \
                distinct()
    
    similar_pairs = sim_pairs. \
                    map(lambda x: (tuple(sorted(x[0])), x[1])). \
                    sortByKey(lambda x: (x[0][0], x[0][1], x[1])). \
                    collect()
    
    sc.stop()
    
    end = time.time()
    
    with open(path_out, 'w+') as file:
        file.write('business_id_1,business_id_2,similarity\n')
        for row in similar_pairs:
            pair = row[0]
            sim = row[1]
            file.write(pair[0] + ',' + pair[1] + ',' + str(sim) + '\n')
        
    print('total time elapsed: ' + str(end - start) + ' seconds')