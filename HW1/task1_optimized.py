# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pyspark, json
import sys
from time import time 

class Task1:
    def __init__(self,path_in, path_out):
        self.path_in = path_in
        self.path_out = path_out
        self.n_partitions = 60
        
    def reviewCount(self,rdd):
         

        count = rdd.count()
        
        
                    
        return count
    
    def reviewCount2018(self,rdd):
        
        
        count = rdd.filter(lambda review: review['date'].split('-')[0] == '2018').count()

        
        
        return count
    
    def distinctUserCount(self,rdd):
        

        userRDD = rdd \
                .map(lambda review: (review['user_id'],1)) \
                .reduceByKey(lambda x,y: x+y, self.n_partitions)

        
        
        return userRDD
    
    def top10Users(self,rdd):
        
        
        counts = rdd \
                .sortBy(keyfunc = lambda x:(-x[1], x[0]), numPartitions = self.n_partitions) \
                .take(10)
        
        
        
        return counts
        
    def distinctBusinessCount(self,rdd):
        businessRDD = rdd \
                .map(lambda review: (review['business_id'],1)) \
                .reduceByKey(lambda x,y: x + y, self.n_partitions)
                
        
        
        return businessRDD
    
    def top10Businesses(self,rdd):
        counts = rdd \
                .sortBy(keyfunc = lambda x:(-x[1], x[0]), numPartitions = self.n_partitions) \
                .take(10)

        
        
        return counts

if __name__ == "__main__": 
    
    
    start = time()
    
    argv = sys.argv
    argc = len(sys.argv)
    
    
    if argc == 3:
        print("Number of arguments: " + str(argc) + '\n \n')
        print("Argument list: " + str(argv) + '\n')
    
        path_in = argv[1]
        path_out = argv[2]
        
        task1 = Task1(path_in, path_out)
        
        sc = pyspark.SparkContext('local[*]','mzma_task1')
        sc.setLogLevel("ERROR")
        reviewRDD = sc.textFile(path_in) \
                    .flatMap(lambda x: x.split('\n')) \
                    .map(lambda x: json.loads(x))
        
        # a. total number of reviews
        output = dict()
        
        output["n_review"] = task1.reviewCount(reviewRDD)
        output["n_review_2018"] = task1.reviewCount2018(reviewRDD)
        
        userRDD = task1.distinctUserCount(reviewRDD)
        output["n_user"] = userRDD.count()
        output["top10_user"] = task1.top10Users(userRDD)
        
        businessRDD = task1.distinctBusinessCount(reviewRDD)
        output["n_business"] = businessRDD.count()

        output["top10_business"] = task1.top10Businesses(businessRDD)
        
        sc.stop()
        
        with open(task1.path_out, 'w') as fp:
            json.dump(output, fp)
            
    stop = time()
        
    print("Execution time elapsed for task 1: " + str(stop - start) + " seconds")
    
#    start = time()
#    task1.top10Users(userRDD)
#    stop = time()
#    print('Time elapsed: ' + str(stop - start))
    
    
