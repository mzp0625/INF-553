#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:59:45 2020

@author: mzp06256
"""
import sys
import time
from pyspark import SparkContext
import xgboost as xgb
import numpy as np
import pandas as pd
from math import sqrt
import json


# =============================================================================
# Model-based recommendation system (1 point)
# =============================================================================
def getAverageUserRating(x, all_users_all_ratings_dict):
    user = x[0]
    user_all_ratings = all_users_all_ratings_dict[user] if user in all_users_all_ratings_dict else dict()
    user_average_rating = sum([x[1] for x in user_all_ratings])/len(user_all_ratings) if len(user_all_ratings) > 0 else 3.751
    x.append(user_average_rating)
    
    return x

def getAverageBusinessRating(x, business_dict):
    business = x[1]
    business_average_rating = business_dict[business]['stars']
    x.append(business_average_rating)
    
    return x

def getNumberUserReviews(x, all_users_all_ratings_dict):
    user = x[0]
    user_all_ratings = all_users_all_ratings_dict[user] if user in all_users_all_ratings_dict else dict()
    num_user_reviews = len(user_all_ratings)
    x.append(num_user_reviews)
    
    return x

def getNumberBusinessReviews(x, business_dict):
    business = x[1]
    num_business_reviews = business_dict[business]['review_count']
    x.append(num_business_reviews)    
    
    return x

def getUserSD(x, all_users_all_ratings_dict):
    user = x[0]
    user_all_ratings = all_users_all_ratings_dict[user] if user in all_users_all_ratings_dict else dict()
    user_SD = np.std([x[1] for x in user_all_ratings])
    x.append(user_SD)
    
    return x

def getBusinessSD(x, all_businesses_all_ratings_dict):
    business = x[1]
    business_all_ratings = all_businesses_all_ratings_dict[business] if business in all_businesses_all_ratings_dict else dict()
    business_SD = np.std([x[1] for x in business_all_ratings])
    x.append(business_SD)
    
    return x

def getUserBias(x, all_businesses_all_ratings_dict, all_users_all_ratings_dict):
    user = x[0]
    user_all_ratings = all_users_all_ratings_dict[user] if user in all_users_all_ratings_dict else dict()
    bias = 0
    for business, rating in user_all_ratings:
        business_avg = getAverageBusinessRating(['',business], business_dict)[-1]
        bias += (rating - business_avg)
    avg_bias = bias / len(user_all_ratings) if len(user_all_ratings) > 0 else 0
    x.append(avg_bias)
    
    return x
    
def getBusinessPictures(x, all_businesses_photo_count):
    business = x[1]
    photo_count = all_businesses_photo_count[business] if business in all_businesses_photo_count else 0
    x.append(photo_count)
    return x
    
def get_RMSE(Y_pred, Y_actual):
    
    SE = 0
    for i in range(len(Y_pred)):
        SE += (Y_pred[i] - Y_actual[i])**2
    RMSE = sqrt(1/(len(Y_pred)) * SE)
    return RMSE
    
if __name__ == "__main__":
    start = time.time()
    
    args = sys.argv
    if len(args) == 1:
        folder_path = 'data'
        train_file_path = folder_path + "/yelp_train.csv"
        business_json_path = folder_path + '/business.json'
        photo_json_path = folder_path + '/photo.json'
        checkin_path = folder_path + '/checkin.json'
        user_json_path = folder_path + '/user.json'
        validation_file_path = "data/yelp_val.csv"
        test_file_path = "data/yelp_val.csv"
        output_file_path = "output22.csv"
    elif len(args) != 4:
        print("Usage: ./spark-submit task2_1.py <train_file_name> <test_file_name> <output_file_name>")
    else:
        folder_path = args[1]
        train_file_path = folder_path + "/yelp_train.csv"
        business_json_path = folder_path + '/business.json'
        photo_json_path = folder_path + '/photo.json'
        checkin_path = folder_path + '/checkin.json'
        user_json_path = folder_path + '/user.json'
        validation_file_path = "data/yelp_val.csv"        
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
    
    test_key_rdd = test_rdd.map(lambda x: [x[0], x[1]])
    train_key_rdd = train_rdd.map(lambda x: [x[0], x[1]])

    business_dict = sc.textFile(business_json_path).\
                    map(lambda x: json.loads(x)).\
                    map(lambda x: (x['business_id'], x)).\
                    collectAsMap()
    
    train_business_ratings_rdd = train_rdd.\
                            map(lambda x: (x[1], [(x[0], x[2])])).\
                            reduceByKey(lambda x,y : x+y)
    train_all_businesses_all_ratings_dict = train_business_ratings_rdd.collectAsMap()
    
    train_user_ratings_rdd = train_rdd.\
                        map(lambda x: (x[0], [(x[1], x[2])])).\
                        reduceByKey(lambda x,y : x+y)
    train_all_users_all_ratings_dict = train_user_ratings_rdd.collectAsMap()
    
    all_businesses_photo_count = sc.textFile(photo_json_path).\
                    map(lambda x: (json.loads(x)['business_id'],1)).\
                    reduceByKey(lambda x,y: x+y).\
                    collectAsMap()
    
    X_train = train_key_rdd.\
            map(lambda x: getAverageUserRating(x, train_all_users_all_ratings_dict)).\
            map(lambda x: getAverageBusinessRating(x, business_dict)).\
            map(lambda x: getNumberUserReviews(x, train_all_users_all_ratings_dict)).\
            map(lambda x: getNumberBusinessReviews(x, business_dict)).\
            map(lambda x: getUserSD(x, train_all_users_all_ratings_dict)).\
            map(lambda x: getBusinessSD(x, train_all_businesses_all_ratings_dict)).\
            map(lambda x: getUserBias(x, train_all_businesses_all_ratings_dict, train_all_users_all_ratings_dict)).\
            map(lambda x: getBusinessPictures(x, all_businesses_photo_count)).\
            map(lambda x: x[2:]).\
            collect()

    test_keys = test_key_rdd.collect()
    
    test_business_ratings_rdd = train_rdd.\
                            map(lambda x: (x[1], [(x[0], x[2])])).\
                            reduceByKey(lambda x,y : x+y)
    test_all_businesses_all_ratings_dict = test_business_ratings_rdd.collectAsMap()
    
    test_user_ratings_rdd = train_rdd.\
                        map(lambda x: (x[0], [(x[1], x[2])])).\
                        reduceByKey(lambda x,y : x+y)
    test_all_users_all_ratings_dict = test_user_ratings_rdd.collectAsMap()
            
    X_test = test_key_rdd.\
            map(lambda x: getAverageUserRating(x, test_all_users_all_ratings_dict)).\
            map(lambda x: getAverageBusinessRating(x, business_dict)).\
            map(lambda x: getNumberUserReviews(x, test_all_users_all_ratings_dict)).\
            map(lambda x: getNumberBusinessReviews(x, business_dict)).\
            map(lambda x: getUserSD(x, test_all_users_all_ratings_dict)).\
            map(lambda x: getBusinessSD(x, test_all_businesses_all_ratings_dict)).\
            map(lambda x: getUserBias(x, test_all_businesses_all_ratings_dict, train_all_users_all_ratings_dict)).\
            map(lambda x: getBusinessPictures(x, all_businesses_photo_count)).\
            map(lambda x: x[2:]).\
            collect()
            
#    Y_test = sc.textFile(test_file_path).\
#            filter(lambda x: x!=header).\
#            map(lambda x: x.split(',')).\
#            map(lambda x: float(x[2])).\
#            collect()


    Y_train = train_rdd.\
            map(lambda x: x[2]).\
            collect()
    

    
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', \
                                       colsample_bytree = 0.3, \
                                       learning_rate = 0.1, \
                                       min_child_weight = 5,\
                                       max_depth = 8, \
                                       alpha = 1, \
                                       gamma = 0.1,\
                                       subsample = 0.8,\
                                       n_estimators = 130)
    
    xg_reg.fit(pd.DataFrame(X_train),pd.DataFrame(Y_train))
    Y_pred = xg_reg.predict(pd.DataFrame(X_test))
    
#    RMSE = get_RMSE(Y_pred, Y_test)
    
    with open(output_file_path, 'w+') as file:
        file.write('user_id, business_id, prediction\n')
        for i in range(len(test_keys)):
            user = test_keys[i][0]
            business = test_keys[i][1]
            prediction = Y_pred[i]
            file.write(user + ',' + business + ',' + str(prediction) + '\n')
    
    sc.stop()
    end = time.time()
#    print('RMSE is ' + str(RMSE))
    print('time elapsed: ' + str(end - start) + ' seconds')
