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
        self.n_partitions = 80
        
    def reviewCount(self,rdd):
         

        count = rdd.count()
                    
        return count
    
    def reviewCount2018(self,rdd):
    
        count = rdd \
            .map(lambda x: (1,1) if json.loads(x)['date'].split('-')[0] == '2018' else (1,0)) \
            .partitionBy(self.n_partitions, lambda x: hash(x) % self.n_partitions) \
            .reduceByKey(lambda x,y: x+y).collect()

        return count[0][1]
    
    def distinctUserCount(self,rdd):
    
        count = rdd \
                .map(lambda line: (json.loads(line)['user_id'],1)) \
                .partitionBy(self.n_partitions, lambda x: hash(x) % self.n_partitions) \
                .reduceByKey(lambda x,y: x+y).collect()

        return len(count)
    
    def top10Users(self,rdd):
        
            
        counts = rdd \
                .map(lambda line: (json.loads(line)['user_id'],1)) \
                .partitionBy(self.n_partitions, lambda x: hash(x) % self.n_partitions) \
                .reduceByKey(lambda x,y: x + y) \
                .sortBy(keyfunc = lambda x:(-x[1], x[0])) \
                .take(10)
        

        return counts
        
    def distinctBusinessCount(self,rdd):
        
        count = rdd \
                .map(lambda line: (json.loads(line)['business_id'],1)) \
                .partitionBy(self.n_partitions, lambda x: hash(x) % self.n_partitions) \
                .reduceByKey(lambda x,y: x + y).collect()
                
        return len(count)
    
    def top10Businesses(self,rdd):
        
        counts = rdd \
                .map(lambda line: (json.loads(line)['business_id'],1)) \
                .partitionBy(self.n_partitions, lambda x: hash(x) % self.n_partitions) \
                .reduceByKey(lambda x,y: x + y) \
                .sortBy(keyfunc = lambda x:(-x[1], x[0])) \
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
        
        sc = pyspark.SparkContext('local[*]','mzma_task1')
        sc.setLogLevel("ERROR")
        reviewRDD = sc.textFile(path_in).flatMap(lambda x: x.split('\n')) \
                                        .persist(pyspark.StorageLevel.MEMORY_ONLY)    
        
        # a. total number of reviews
        task1 = Task1(path_in, path_out)
        output = dict()
        
        output["n_review"] = task1.reviewCount(reviewRDD)
        output["n_review_2018"] = task1.reviewCount2018(reviewRDD)
        
        output["n_user"] = task1.distinctUserCount(reviewRDD)
        output["top10_user"] = task1.top10Users(reviewRDD)
        
        output["n_business"] = task1.distinctBusinessCount(reviewRDD)
        output["top10_business"] = task1.top10Businesses(reviewRDD)
        
        sc.stop()
        
        with open(task1.path_out, 'w') as fp:
            json.dump(output, fp)
            
    stop = time()
        
    print("Execution time elapsed for task 1: " + str(stop - start) + " seconds")
    
#    start = time()
#    task1.top10Users(userRDD)
#    stop = time()
#    print('Time elapsed: ' + str(stop - start))
    
    
