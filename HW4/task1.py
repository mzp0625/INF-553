#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:12:04 2020

@author: mzp06256
"""

import os
import sys
from functools import reduce
from pyspark.sql.functions import col, lit, when
from pyspark import SparkContext
from pyspark.sql import SQLContext
from graphframes import *
import time
from collections import defaultdict

if __name__ == "__main__":
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.6.0-spark2.3- s_2.11"
    if len(sys.argv) == 1:
        input_file_path = "power_input.txt"
        community_output_file_path = "output1.txt"
    elif len(sys.argv) != 3:
        print("Usage: spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py <input_file_path> <community_output_file_path>")
        exit(1)
    else:
        input_file_path = sys.argv[1]
        community_output_file_path = sys.argv[2]

    start = time.time()
    
    sc = SparkContext('local[*]', 'mzma_task1')
    sc.setLogLevel("WARN")
    
    sqlContext = SQLContext(sc)
    
    edge_rdd = sc.textFile(input_file_path).\
                map(lambda x: x.split()).\
                map(lambda x: (x[0], x[1]))
    temp = edge_rdd.collect()
    edges = []
    for i in range(len(temp)):
        edges.append(temp[i])
        edges.append(temp[i][::-1])
    
    vertices = set()
    for edge in edges:
        vertices.add((edge[0], edge[0]))
        vertices.add((edge[1], edge[1]))
    vertices = list(vertices)
    
    vertices = sqlContext.createDataFrame(vertices, ["id", "name"])

    edges = sqlContext.createDataFrame(edges, ["src", "dst"])

    
#    vertices = sqlContext.createDataFrame([
#      ("a", "Alice", 34),
#      ("b", "Bob", 36),
#      ("c", "Charlie", 30),
#      ("d", "David", 29),
#      ("e", "Esther", 32),
#      ("f", "Fanny", 36),
#      ("g", "Gabby", 60)], ["id", "name", "age"])
#    
#    edges = sqlContext.createDataFrame([
#      ("a", "b", "friend"),
#      ("b", "c", "follow"),
#      ("c", "b", "follow"),
#      ("f", "c", "follow"),
#      ("e", "f", "follow"),
#      ("e", "d", "friend"),
#      ("d", "a", "friend"),
#      ("a", "e", "friend")], ["src", "dst", "relationship"])
    

    g = GraphFrame(vertices, edges)
    
    result = g.labelPropagation(maxIter=5)
    result.show()   
    
    result = result.toLocalIterator()
    result_dict = defaultdict(list)
    for row in result:
        name, label = row[1], row[2]
        result_dict[label].append(name)
    
    ans = []
    for key in result_dict.keys():
        ans.append(sorted(result_dict[key]))
        ans.sort(key = lambda x: (len(x), x[0]))
        
        
    with open(community_output_file_path, 'w+') as file:
        for i in ans:
            file.write('\''+'\', \''.join(i)+'\'' + '\n')
    
    end = time.time()
    print("process exited successfully. Time elapsed = " + str(end - start) + " seconds")