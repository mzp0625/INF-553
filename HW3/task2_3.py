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
from math import sqrt,log
import json
from collections import Counter


# =============================================================================
# Hybrid recommendation system (1 point)
# =============================================================================

def inverse_user_frequency(n,n_j):
    return log(n/n_j) if n_j != 0 else 1

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
    return [user, business, P_ui]

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

def getNumCheckIn(x, business_checkin_dict):
    business = x[1]
    numCheckIn = business_checkin_dict[business] if business in business_checkin_dict else 0
    x.append(numCheckIn)
    return x
    
def attributeLength(x, business_dict):
    business = x[1]
    attr_length = len(business_dict[business]['attributes']) if business_dict[business]['attributes'] else 0
    isOpen = business_dict[business]['is_open']
    x.append(attr_length)    
    x.append(isOpen)
    return x
    
def getCategory(x, business_dict):
    business = x[1]
    if business in business_dict and 'categories' in business_dict[business]:
        if business_dict[business]['categories'] and 'Restaurants' in business_dict[business]['categories']:
            x.append(1)
        else:
            x.append(0)
    else:
        x.append(0)
    if business in business_dict and 'categories' in business_dict[business]:
        if business_dict[business]['categories'] and 'Shopping' in business_dict[business]['categories']:
            x.append(1)
        else:
            x.append(0)
    else:
        x.append(0)
    if business in business_dict and 'categories' in business_dict[business]:
        if business_dict[business]['categories'] and 'Beauty & Spas' in business_dict[business]['categories']:
            x.append(1)
        else:
            x.append(0)
    else:
        x.append(0)
    if business in business_dict and 'categories' in business_dict[business]:
        if business_dict[business]['categories'] and 'Health & Medical' in business_dict[business]['categories']:
            x.append(1)
        else:
            x.append(0)
    else:
        x.append(0)
    if business in business_dict and 'categories' in business_dict[business]:
        if business_dict[business]['categories'] and 'Automotive' in business_dict[business]['categories']:
            x.append(1)
        else:
            x.append(0)
    else:
        x.append(0)
    if business in business_dict and 'categories' in business_dict[business]:
        if business_dict[business]['categories'] and 'Bars' in business_dict[business]['categories']:
            x.append(1)
        else:
            x.append(0)
    else:
        x.append(0)
    if business in business_dict and 'categories' in business_dict[business]:
        if business_dict[business]['categories'] and 'Hair Salons' in business_dict[business]['categories']:
            x.append(1)
        else:
            x.append(0)
    else:
        x.append(0)        
        
    return x


def getHybridPrediction(item_based, model_based, max_neighbors, all_users_all_ratings_dict, all_businesses_all_ratings_dict):
    max_neighbors = 1062
    max_business_reviews = 556
    combined_ratings = []
    for i in range(len(item_based)):
        user = item_based[i][0]
        business = item_based[i][1]
        num_business_reviews = len(all_businesses_all_ratings_dict[business]) if business in all_businesses_all_ratings_dict else 0
        neighbors_count = len(all_users_all_ratings_dict[user]) if user in all_users_all_ratings_dict else 0
        cf_rating = item_based[i][2]
        model_rating = model_based[i]
        alpha = 0
        final_rating = (alpha*cf_rating + (1-alpha)*model_rating)
        combined_ratings.append(final_rating)
    return combined_ratings


    
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
        tip_file_path = "data/tip.json"
        output_file_path = "output23.csv"
    elif len(args) != 4:
        print("Usage: ./spark-submit task2_1.py <train_file_name> <test_file_name> <output_file_name>")
    else:
        folder_path = args[1]
        train_file_path = folder_path + "/yelp_train.csv"
        business_json_path = folder_path + '/business.json'
        photo_json_path = folder_path + '/photo.json'
        checkin_path = folder_path + '/checkin.json'
        user_json_path = folder_path + '/user.json'
        validation_file_path = folder_path + "/yelp_val.csv"      
        tip_file_path = folder_path + "/tip.json"
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
    
    user_rdd = sc.textFile(user_json_path).\
                map(lambda x: json.loads(x)).\
                map(lambda x: (x['user_id'], x))
                
    business_checkin_dict = sc.textFile(checkin_path).\
                        map(lambda x: json.loads(x)).\
                        map(lambda x: (x['business_id'], len(x['time']))).\
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
            map(lambda x: getNumCheckIn(x, business_checkin_dict)).\
            map(lambda x: attributeLength(x, business_dict)).\
            map(lambda x: getCategory(x, business_dict)).\
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
            map(lambda x: getNumCheckIn(x, business_checkin_dict)).\
            map(lambda x: attributeLength(x, business_dict)).\
            map(lambda x: getCategory(x, business_dict)).\
            map(lambda x: x[2:]).\
            collect()
            
    Y_test = sc.textFile(test_file_path).\
            filter(lambda x: x!=header).\
            map(lambda x: x.split(',')).\
            map(lambda x: float(x[2])).\
            collect()


    Y_train = train_rdd.\
            map(lambda x: x[2]).\
            collect()
    

    
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', \
                                       colsample_bytree = 0.33, \
                                       learning_rate = 0.1, \
                                       min_child_weight = 1,\
                                       max_depth = 8, \
                                       alpha = 1, \
                                       subsample = 0.8,\
                                       n_estimators = 130,
                                       seed = 100)
    
    xg_reg.fit(pd.DataFrame(X_train),pd.DataFrame(Y_train))
    Y_pred = xg_reg.predict(pd.DataFrame(X_test))
    model_based_pred = Y_pred
    
    item_based_pred = test_key_rdd.\
                    map(lambda x: item_based_prediction(x, train_all_users_all_ratings_dict, train_all_businesses_all_ratings_dict, avg = 3.75)).\
                    collect()

#   get max number of neighbors for a particular user's rating
    max_neighbors = max([len(values) for values in train_all_users_all_ratings_dict.values()])
    
    
    hybrid_ratings =  getHybridPrediction(item_based_pred, model_based_pred, max_neighbors, train_all_users_all_ratings_dict, train_all_businesses_all_ratings_dict)
    
#    all_cat = []
#    categories = [x['categories'] for x in business_dict.values()]
#    for category in categories:
#        if category:
#            all_cat.extend([i.strip() for i in category.split(',')])
#    counter = Counter(all_cat)
#    
    RMSE = get_RMSE(hybrid_ratings, Y_test)
#    
#    with open(output_file_path, 'w+') as file:
#        file.write('user_id, business_id, prediction\n')
#        for i in range(len(test_keys)):
#            user = test_keys[i][0]
#            business = test_keys[i][1]
#            prediction = Y_pred[i]
#            file.write(user + ',' + business + ',' + str(prediction) + '\n')
#    
    end = time.time()
    print('RMSE is ' + str(RMSE))
    print('time elapsed: ' + str(end - start) + ' seconds')
