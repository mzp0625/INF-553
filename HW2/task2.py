#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:21:38 2020

@author: mzp06256
"""

import sys
from pyspark import SparkContext
from collections import defaultdict
from itertools import combinations
import time

def APriori(s, chunk):
    k = 1
    data = list(chunk)
    
    # get frequent singleton
    candidates = set() # Lk
    counts = defaultdict(int) 
    for basket in data:
        for item in basket:
            if counts[item] < s:
                counts[item] += 1
                if counts[item] >= s:

                    candidates.add(frozenset([item])) # these are truly frequent singletons
                    yield (item, 1)
                    
    next_candidates = set()
    
    cand_list = list(candidates)
    for i in range(len(cand_list)):
        for j in range(i+1, len(cand_list)):
            next_candidates.add(frozenset((cand_list[i] | cand_list[j]))) # these are doubleton candidates
                
                
    while next_candidates:
        k += 1
        candidates = set()
        count = defaultdict(int)
        
        for basket in data:
            for candidate in next_candidates:
                if candidate.issubset(basket):
                    if count[candidate] < s:
                        count[candidate] += 1
                        if count[candidate] >= s:
                            candidates.add(candidate)
                            yield (candidate, 1)
        
        next_candidates = set()
        cand_list = list(candidates)
        
        for i in range(len(cand_list)):
            for j in range(i+1, len(cand_list)):
                cand_set = cand_list[i] | cand_list[j]
                if len(cand_set) == k+1:
                    if(all([set(i) in candidates for i in list(combinations(cand_set, k))])):
                        next_candidates.add(frozenset(cand_set))
            
    
def CountSupport(chunk, candidates):
    data = list(chunk)
    
    for basket in data:
        for cand_set in candidates:
            if type(cand_set) == str:
                cand_set = {cand_set}
            else:
                cand_set = set(cand_set)
            if cand_set.issubset(basket):
                if len(cand_set) == 1:
                    yield (list(cand_set)[0], 1)
                else:
                    yield (tuple(cand_set), 1)

        
if __name__ == "__main__":
    
    start = time.time()
    
    argc, argv = len(sys.argv), sys.argv
    threshold, support, path_in, path_out = None, None, None, None
    
    if argc == 5:
        threshold = int(argv[1])
        support = int(argv[2])
        path_in = argv[3]
        path_out = argv[4]
    elif argc == 1: # for testing
        threshold = 20
        support = 50
        path_in = "ta_feng_all_months_merged.csv"
        path_out = "output_task2.txt"
        
    sc = SparkContext(appName = 'mzma_task1')
    sc.setLogLevel("WARN") 
    
    rdd = sc.textFile(path_in)
    rdd = rdd.map(lambda line: line.split(",")).map(lambda x: (x[0][1:-1] + '-' + x[1][1:-1], x[5][1:-1]));
    header = rdd.first()

    rdd = rdd.filter(lambda x: x!= header)
    rdd = rdd.map(lambda x: ((x[0]), {str(int(x[1]))})) \
            .reduceByKey(lambda x,y: x.union(y))


    rdd = rdd.values().filter(lambda x: len(x) > threshold).repartition(40)

    p = rdd.getNumPartitions()
    s = support//p
    
    candidates = rdd.mapPartitions(lambda x: APriori(s,x)).reduceByKey(lambda x,y: x).keys().collect()
    
    freq_sets = rdd.mapPartitions(lambda x: CountSupport(x, candidates)) \
            .reduceByKey(lambda x,y : x+y) \
            .filter(lambda x: x[1] >= support) \
            .keys()\
            .collect()
            
    sc.stop()

    
    for i in range(len(candidates)):
        if type(candidates[i]) == str:
            candidates[i] = [candidates[i]]
        else:
            candidates[i] = sorted(list(candidates[i]))
            
    for i in range(len(freq_sets)):
        if type(freq_sets[i]) == str:
            freq_sets[i] = [freq_sets[i]]
        else:
            freq_sets[i] = sorted(list(freq_sets[i])) 
    
    candidates.sort(key = lambda x: (len(x), x))
    freq_sets.sort(key = lambda x: (len(x), x))
    
    with open(path_out,'w') as file:
        file.write('Candidates:\n')
        for i, cand_set in enumerate(candidates):
            file.write('(' +str(cand_set)[1:-1] + ')')
            if i < len(candidates)-1:
            
                if len(candidates[i]) == len(candidates[i+1]):
                    file.write(',')
                else:
                    file.write('\n\n')
        
        file.write('\n\n')
            
        file.write('Frequent Itemsets:\n')
        for i, item_set in enumerate(freq_sets):
            file.write('(' +str(item_set)[1:-1] + ')')
            if i < len(freq_sets)-1:
            
                if len(freq_sets[i]) == len(freq_sets[i+1]):
                    file.write(',')
                else:
                    file.write('\n\n')
        
    
    
    
    stop = time.time()
    print("Duration: " + str(stop - start) + " seconds")