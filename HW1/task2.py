#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:53:22 2020

@author: mzp06256
"""

import pyspark, json
import sys
from time import time 


class Task2:
    
    def __init__(self, path_in, path_out, partition_count):
        self.path_in = path_in
        self.path_out = path_out
        self.partition_count = partition_count
        self.output = dict()
        self.output['default'] = dict()
        self.output['customized'] = dict()
    
    def top10Default(self,reviewRDD):
        
        start = time()
        
        
        rdd = reviewRDD.reduceByKey(lambda x,y: x + y) 
        
        counts = rdd \
                .sortBy(keyfunc = lambda x: (-x[1], x[0])) \
                .take(10)

        n_items = rdd.glom().map(len).collect()
        n_partition = reviewRDD.getNumPartitions()
        
        
        stop = time()
        
        self.output['default']['n_partition'] = n_partition
        self.output['default']['n_items'] = n_items
        self.output['default']['exe_time'] = stop - start
        
        return counts
    
    def top10Customized(self,reviewRDD):
        
        start = time()
        
        
        rdd = reviewRDD \
                .partitionBy(self.partition_count, lambda x: hash(x) % self.partition_count) \
                .reduceByKey(lambda x,y: x + y)
        
        counts = rdd \
                .sortBy(keyfunc = lambda x: (-x[1], x[0])) \
                .take(10)
                
        n_items = rdd.glom().map(len).collect()

        stop = time()
        
        self.output['customized']['n_partition'] = self.partition_count
        self.output['customized']['n_items'] = n_items
        self.output['customized']['exe_time'] = stop - start
        
        return counts
    
    
if __name__ == "__main__": 
    
    start = time()
    
    argv = sys.argv
    argc = len(sys.argv)
    
    if argc == 4:
        print("Number of arguments: " + str(argc) + '\n \n')
        print("Argument list: " + str(argv) + '\n')
    
        path_in = argv[1]
        path_out = argv[2]
        partition_count = int(argv[3])
        
        sc = pyspark.SparkContext('local[*]','mzma_task1')
        sc.setLogLevel("ERROR")
        reviewRDD = sc.textFile(path_in) \
                .flatMap(lambda x: x.split('\n')) \
                .map(lambda line: json.loads(line)) \
                .map(lambda review: (review['business_id'],1)) \

        
        
        task2 = Task2(path_in, path_out, partition_count)
        
        
        task2.top10Default(reviewRDD)
        task2.top10Customized(reviewRDD)
        
        sc.stop()
        
        with open(task2.path_out, 'w') as fp:
            json.dump(task2.output, fp)
            
        
    stop = time()
    
    print("Total time elapsed for task 2 : " + str(stop - start) + ' seconds')