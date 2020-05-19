#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:10:07 2020

@author: mzp06256
"""
import sys
import time
from pyspark import SparkContext
from math import sqrt, log

# =============================================================================
# Item-based CF recommendation system with Pearson similarity
# =============================================================================
def pearson_correlation(ratings_i, ratings_j):
    # ratings_i = [rating from user1 for item i, rating from user2 for item i, ...]
    # ratings_j = [rating from user1 for item j, rating from user2 for item j, ...]
    # w_ij = ...
    average_i = sum(ratings_i)/len(ratings_i) if len(ratings_i) > 0 else 0
    average_j = sum(ratings_j)/len(ratings_j) if len(ratings_j) > 0 else 0
    normalized_i = [r_i - average_i for r_i in ratings_i]
    normalized_j = [r_j - average_j for r_j in ratings_j]
    numerator = sum([i*j for i,j in zip(normalized_i, normalized_j)])
    denominator= sqrt(sum([i**2 for i in normalized_i]) * sum([j**2 for j in normalized_j]))
    w_ij = numerator/denominator if denominator != 0 else 0.33
    
    return w_ij 

def item_based_prediction(x, all_users_all_ratings_dict, all_businesses_all_ratings_dict, avg):
    user, business = x[0],x[1]
    business_all_ratings = dict(all_businesses_all_ratings_dict[business]) if business in all_businesses_all_ratings_dict else dict()
    user_all_ratings = dict(all_users_all_ratings_dict[user]) if user in all_users_all_ratings_dict else dict()
    # will use all neighbors whose similarity w_ij is > 0
    neighbors = set(user_all_ratings.keys())
    
    P_ui = 0
    numerator = 0
    denominator = 0
    correlation_list = []
    for neighbor in neighbors:
        neighbor_all_ratings = dict(all_businesses_all_ratings_dict[neighbor]) if neighbor in all_businesses_all_ratings_dict else dict()
        neighbor_rating = user_all_ratings[neighbor]
        ratings_1 = []
        ratings_2 = []
        # look for users who have rated both neighbor and business
        for u in neighbor_all_ratings:
            if u in business_all_ratings:
                ratings_1.append(business_all_ratings[u])
                ratings_2.append(neighbor_all_ratings[u])
        w_ij = pearson_correlation(ratings_1, ratings_2)
        correlation_list.append((w_ij, neighbor_rating))

    for i in range(len(correlation_list)):
        w_ij = correlation_list[i][0] 
        neighbor_rating = correlation_list[i][1]
        if w_ij > 0:
            numerator += w_ij*neighbor_rating
            denominator += w_ij
    P_ui = numerator/denominator if denominator != 0 else avg
    return (user, business), P_ui

def inverse_user_frequency(n,n_j):
    return log(n/n_j)

def get_RMSE(test_pred_dict, test_actual_dict):
    SE = 0
    for key in test_actual_dict:
        SE += (test_actual_dict[key] - test_pred_dict[key])**2
    
    RMSE = sqrt(1/len(test_actual_dict) * SE)
    
    return RMSE
    


if __name__ == "__main__":
    start = time.time()
    
    args = sys.argv
    if len(args) == 1:
        train_file_path = "data/yelp_train.csv"
        test_file_path = "data/yelp_val.csv"
        output_file_path = "output21.csv"
    elif len(args) != 4:
        print("Usage: ./spark-submit task2_1.py <train_file_name> <test_file_name> <output_file_name>")
    else:
        train_file_path = args[1]
        test_file_path = args[2]
        output_file_path = args[3]
    
    sc = SparkContext('local[*]','mzma_task2')
    sc.setLogLevel('WARN')
    train_rdd = sc.textFile(train_file_path)
    header = train_rdd.first()
    train_rdd = train_rdd.\
                filter(lambda x: x != header).\
                map(lambda x: x.split(',')).\
                map(lambda x: [x[0], x[1], float(x[2])])
    
    
    test_rdd = sc.textFile(test_file_path)
    test_rdd = test_rdd.\
                filter(lambda x: x != header).\
                map(lambda x: x.split(',')).\
                map(lambda x: [x[0], x[1]])
#    test_actual_dict = test_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()
    
#    avg = sum([value for value in test_actual_dict.values()])/len(test_actual_dict)
    
    test_key_rdd = test_rdd.map(lambda x: [x[0],x[1]]) # user, business
    
    business_ratings_rdd = train_rdd.\
                            map(lambda x: (x[1], [(x[0], x[2])])).\
                            reduceByKey(lambda x,y : x+y)
    all_businesses_all_ratings_dict = business_ratings_rdd.collectAsMap()
    
    user_ratings_rdd = train_rdd.\
                        map(lambda x: (x[0], [(x[1], x[2])])).\
                        reduceByKey(lambda x,y : x+y)
    all_users_all_ratings_dict = user_ratings_rdd.collectAsMap()
        
    test_pred = test_key_rdd.\
                    map(lambda x: item_based_prediction(x, all_users_all_ratings_dict, all_businesses_all_ratings_dict, avg = 3.75))

    test_pred_dict = test_pred.collectAsMap()
        
    
#    RMSE = get_RMSE(test_pred_dict, test_actual_dict)
    
    with open(output_file_path, 'w+') as file:
        file.write('user_id,business_id,prediction\n')
        for key in test_pred_dict:
            user = key[0]
            business = key[1]
            rating = test_pred_dict[key]
            file.write(user + ',' + business + ',' + str(rating) + '\n')
    
    sc.stop()
    end = time.time()
    
    print('time elapsed: ' + str(end - start) + ' seconds')
#    print('RMSE is ' + str(RMSE))
    
    
    
    