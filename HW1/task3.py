#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:29:48 2020

@author: mzp06256
"""

import pyspark, json
import sys
from time import time


class Task3:
    def __init__(self, review_path, business_path, output_a, output_b):
        self.review_path = review_path
        self.business_path = business_path
        self.output_a = output_a
        self.output_b = output_b
        self.output = dict()
        self.partition_count = 60
    
    def topCities(self):
        
        # returns average stars for each city
        sc = pyspark.SparkContext('local[*]','mzma_task1')
        sc.setLogLevel("ERROR")
        reviewRDD = sc.textFile(self.review_path)
        businessRDD = sc.textFile(self.business_path)
        
        business_city= businessRDD \
                    .flatMap(lambda x: x.split('\n')) \
                    .map(lambda line : (json.loads(line)['business_id'], json.loads(line)['city']))        
        business_star = reviewRDD \
                    .flatMap(lambda x: x.split('\n')) \
                    .map(lambda line : (json.loads(line)['business_id'], json.loads(line)['stars']))
        
        city_stars_RDD = business_city.join(business_star) \
                    .map(lambda business : business[1]) \
                    .partitionBy(self.partition_count, lambda x: hash(x) % self.partition_count) \
                    .aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1), lambda a,b : (a[0] + b[0], a[1] + b[1])) \
                    .mapValues(lambda v: v[0]/v[1])
                    
        # m1
        start = time()
        city_stars = city_stars_RDD.collect() # collect 
        
        city_stars.sort(key = lambda x: (-x[1], x[0]))      # sort 
        print('city,stars') # print
        for pair in city_stars[:10]:
            print(pair)     
        stop = time()
        self.output['m1'] = (stop - start)
        
        
        with open(self.output_a, 'w') as fp:
            fp.write('city,stars\n')
            for pair in city_stars:
                fp.write(pair[0] + ',' + str(pair[1]) + '\n')
        

        # m2
        start = time()
        city_stars =city_stars_RDD.sortBy( keyfunc = lambda x: (-x[1], x[0])) \
                    .take(10) # sort in RDD
     
        print('city,stars')
        for pair in city_stars[:10]: # print
            print(pair)
        stop = time()
        self.output['m2'] = (stop - start)
        sc.stop()
        
if __name__ == "__main__": 
    
    start = time()
    
    argv = sys.argv
    argc = len(sys.argv)
    
    if argc == 5:
        review_path = argv[1]
        business_path = argv[2]
        output_a = argv[3]
        output_b = argv[4]
        
        task3 = Task3(review_path, business_path, output_a, output_b)
        task3.topCities()
        with open(output_b, 'w') as fp:
            json.dump(task3.output, fp)
            
    stop = time()
    print('Total time elapsed for task 3: ' + str(stop - start) + ' seconds')
        