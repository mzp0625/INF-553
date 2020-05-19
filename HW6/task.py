# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import sys
import time
from math import sqrt
from copy import deepcopy


def getSD(SUMSQ, SUM, N):
    sigma = [sqrt(SUMSQ[k]/N - (SUM[k]/N)**2) for k in range(len(SUM))]     
    return sigma

def mahalanobisDist(x, c, sigma):
    y = [(x[i] - c[i])/sigma[i] for i in range(len(x))]
    sum2 = sum([y[i]**2 for i in range(len(y))])
    return sqrt(sum2)

def mahalanobisDistClusters(c1, c2, sigma1, sigma2):
    y = [(c1[i] - c2[i])/(sigma1[i]*sigma2[i]) for i in range(len(c1))]
    sum2 = sum([y[i]**2 for i in range(len(y))])
    return sqrt(sum2)


def mergeCS(CS, d):
    
    for i in range(len(CS)-1):
        min_dist = 2*sqrt(d)
        min_index = None
        for j in range(i+1, len(CS)):
            N1, SUM1, SUMSQ1 = CS[i]['N'], CS[i]['SUM'], CS[i]['SUMSQ']
            N2, SUM2, SUMSQ2 = CS[j]['N'], CS[j]['SUM'], CS[j]['SUMSQ']
            c1 = [SUM1[k]/N1 for k in range(len(SUM1))]
            sigma1 = getSD(SUMSQ1, SUM1, N1) 
            c2 = [SUM2[k]/N2 for k in range(len(SUM2))]
            sigma2 = getSD(SUMSQ2, SUM2, N2) 
            
            dist = mahalanobisDistClusters(c1,c2, sigma1, sigma2)
            if dist < min_dist:
                min_dist = dist
                min_index = j
        
        if min_index: # merge these two clusters
            newCS = deepcopy(CS[:min_index]) + deepcopy(CS[min_index+1:])
            newCS[i]['N'] += CS[min_index]['N']
            for k in range(d):
                newCS[i]['SUM'][k] += CS[min_index]['SUM'][k]
                newCS[i]['SUMSQ'][k] += CS[min_index]['SUMSQ'][k]
                
            return mergeCS(newCS, d)
    
    return CS

def mergeCS_DS(CS, DS, d):
    for i in range(len(CS)):
        min_dist = 2*sqrt(d)
        min_index = None
        for j in range(len(DS)):
            N1, SUM1, SUMSQ1 = CS[i]['N'], CS[i]['SUM'], CS[i]['SUMSQ']
            N2, SUM2, SUMSQ2 = DS[j]['N'], DS[j]['SUM'], DS[j]['SUMSQ']
            c1 = [SUM1[k]/N1 for k in range(len(SUM1))]
            sigma1 = getSD(SUMSQ1, SUM1, N1) 
            c2 = [SUM2[k]/N2 for k in range(len(SUM2))]
            sigma2 = getSD(SUMSQ2, SUM2, N2) 
            
            dist = mahalanobisDistClusters(c1,c2, sigma1, sigma2)
            if dist < min_dist:
                min_dist = dist
                min_index = j
        if min_index: # merge these two clusters:
            newCS = deepcopy(CS[:i]) + deepcopy(CS[i+1:])
            DS[min_index]['N'] += CS[i]['N']
            for k in range(d):
                DS[min_index]['SUM'][k] += CS[i]['SUM'][k]
                DS[min_index]['SUMSQ'][k] += CS[i]['SUMSQ'][k]
            return mergeCS_DS(newCS, DS, d)
    return CS, DS
                
def predictDS(point, DS):
    min_dist, min_index = sys.maxsize, None
    for i in range(len(DS)):
        N, SUM, SUMSQ = DS[i]['N'], DS[i]['SUM'], DS[i]['SUMSQ']
        centroid = [SUM[i]/N for i in range(len(SUM))]
        sigma = getSD(SUMSQ, SUM, N) 
        dist = mahalanobisDist(point,centroid, sigma)
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        input_file = 'hw6_clustering.txt'
        n_cluster = 10
        output_file = 'hw6_output.txt'
        pass
    elif len(sys.argv) != 4:
        print('usage: python3 task.py <input_file> <n_cluster> <output_file>')
        exit(1)
    else:
        # 4 
        input_file = sys.argv[1]
        n_cluster = int(sys.argv[2])
        output_file = sys.argv[3]
        
    start = time.time()
    
    
    
    seed = 0
    # load data from file
    with open(input_file, 'r') as file:
        data = file.read().split('\n')[:-1]
        
    for i in range(len(data)):
        data[i] = data[i].split(',')
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    
    DS, CS, RS = [],[], []
    
    data_X = [data[i][2:] for i in range(len(data))]
    # 1) load 20% of data randomly
    data80, data20 = model_selection.train_test_split(data_X, test_size = 0.2)
    data_X = data80

    # 2) kmeans using Euclidean distance (default) & clusters = 5*n_cluster
    kmeans = KMeans(n_clusters = n_cluster*5).fit_predict(data20)
    
    # 3) move outliers
    clusters = [[] for _ in range(n_cluster*5)]
    lengths = []
    for i in range(len(kmeans)):
        clusters[kmeans[i]].append(i)
    for cluster in clusters:
        lengths.append(len(cluster))
        if len(cluster) == 1:
            RS.append(data20[cluster[0]])
    
    # 4) run k-means again with clusters = n_cluster
    for point in RS:
        data20.remove(point)
    
    kmeans = KMeans(n_clusters = n_cluster).fit_predict(data20)
    
    # 5) generate DS clusters using statistics
    clusters = [[] for _ in range(n_cluster)]
    for i in range(len(kmeans)):
        clusters[kmeans[i]].append(i)
    for i, cluster in enumerate(clusters):
        N = len(cluster)
        SUM = [sum([data20[cluster[j]][k] for j in range(len(cluster))]) for k in range(len(data20[0]))]
        SUMSQ = [sum([data20[cluster[j]][k]**2 for j in range(len(cluster))]) for k in range(len(data20[0]))]
        DS.append({'N':N, 'SUM':SUM, 'SUMSQ':SUMSQ})
    
    # 6) run k-means on RS with clusters = 5*n_cluster to generate CS and RS
    if(RS):
        K = min(n_cluster*5, len(RS))
        kmeans = KMeans(n_clusters = K).fit_predict(RS)
        clusters = [[] for _ in range(K)]
        for i in range(len(kmeans)):
            clusters[kmeans[i]].append(i)
            
        RS_remove = []
        newRS = []
        for cluster in clusters:
            if len(cluster) == 1:
                newRS.append(RS[cluster[0]])
            else:
                N = len(cluster)
                SUM = [sum([data20[cluster[j]][k] for j in range(len(cluster))]) for k in range(len(RS[0]))]
                SUMSQ = [sum([data20[cluster[j]][k]**2 for j in range(len(cluster))]) for k in range(len(RS[0]))]
                CS.append({'N':N, 'SUM':SUM, 'SUMSQ':SUMSQ})
#                RS_remove.append(RS[cluster[0]])
                
#        for point in RS_remove:
#            RS.remove(point)
        RS = newRS
        
    
    num_discard = sum([DS[i]['N'] for i in range(len(DS))])
    num_CS_clusters = len(CS)
    num_compression = sum([CS[i]['N'] for i in range(len(CS))])
    num_retained = len(RS)
    
    with open(output_file, 'w+') as file:
        file.write('The intermediate results:\n')
        file.write('Round 1: ' + str(num_discard) + ',' + str(num_CS_clusters) + ',' + str(num_compression) + ',' + str(num_retained) + '\n')
    
    # 7-12 
    split_percent = [0.25, 0.3333, 0.5, 1]
    
    d = len(data_X[0])
    Round = 2
    
    for split in split_percent:
        # load another 20% of data
        if split == 1:
            data20 = data_X
        else:
            data80, data20 = model_selection.train_test_split(data_X, test_size = split)
            data_X = data80
        
        # compare to centroids in DS and assign to cluster
        for point in data20:
            assigned_ds, min_dist = None, 2*sqrt(d)
            for j, cluster in enumerate(DS):
                N, SUM, SUMSQ = cluster['N'], cluster['SUM'], cluster['SUMSQ']
                centroid = [SUM[i]/N for i in range(len(SUM))]
                sigma = getSD(SUMSQ, SUM, N) 
                dist = mahalanobisDist(point, centroid, sigma)
                if dist < min_dist :
                    assigned_ds = j
                    min_dist = dist
            if assigned_ds != None:
                DS[assigned_ds]['N'] += 1
                DS[assigned_ds]['SUM'] = [DS[assigned_ds]['SUM'][j] + point[j] for j in range(len(point))]
                DS[assigned_ds]['SUMSQ'] = [DS[assigned_ds]['SUMSQ'][j] + point[j]**2 for j in range(len(point))]
            else: # assign to a CS
                assigned_cs, min_dist = None, 2*sqrt(d)
                for k, cluster in enumerate(CS):
                    N, SUM, SUMSQ = cluster['N'], cluster['SUM'], cluster['SUMSQ']
                    centroid = [SUM[i]/N for i in range(len(SUM))]
                    sigma = getSD(SUMSQ, SUM, N) 
                    dist = mahalanobisDist(point, centroid, sigma)
                    if dist < min_dist:
                        assigned_cs = k
                        min_dist = dist      
                if assigned_cs != None:
                    CS[assigned_cs]['N'] += 1
                    CS[assigned_cs]['SUM'] = [CS[assigned_cs]['SUM'][l] + point[l] for l in range(len(point))]
                    CS[assigned_cs]['SUMSQ'] = [CS[assigned_cs]['SUMSQ'][l] + point[l]**2 for l in range(len(point))]
                else: # assign to RS
                    RS.append(point)
                    
        # run K-means on RS with large 5*n_cluster to generate new CS and RS
        if(RS):
            K = min(n_cluster*5, len(RS))
            kmeans = KMeans(n_clusters = K).fit_predict(RS)
            clusters = [[] for _ in range(K)]
            lengths = []
            for i in range(len(kmeans)):
                clusters[kmeans[i]].append(i)
            
            newRS = []
#            RS_remove = []
            for cluster in clusters:
                if len(cluster) == 1:
                    newRS.append(RS[cluster[0]])
                else:
                    N = len(cluster)
                    SUM = [sum([data20[cluster[j]][k] for j in range(len(cluster))]) for k in range(len(RS[0]))]
                    SUMSQ = [sum([data20[cluster[j]][k]**2 for j in range(len(cluster))]) for k in range(len(RS[0]))]
                    CS.append({'N':N, 'SUM':SUM, 'SUMSQ':SUMSQ})
#                    RS_remove.append(RS[cluster[0]])
                    
#            for point in RS_remove:
#                RS.remove(point)        
            RS = newRS
            
        # merge CS within 2 sqrt(d) of each other
        CS = mergeCS(CS, d)
        
        
        if split == 1:
            # merge CS with DS if their mahalanobis dist < threshold
            CS, DS = mergeCS_DS(CS, DS, d)        
       
        num_discard = sum([DS[i]['N'] for i in range(len(DS))])
        num_CS_clusters = len(CS)
        num_compression = sum([CS[i]['N'] for i in range(len(CS))])
        num_retained = len(RS)
        
        with open(output_file, 'a') as file:
            file.write('Round ' + str(Round) + ': ' + str(num_discard) + ',' + str(num_CS_clusters) + ',' + str(num_compression) + ',' + str(num_retained) + '\n')
        Round += 1

    
    # accuracy
    prediction = []
    truth = []
    for i in range(len(data)):
        point = data[i][2:]
        if point in RS:
            prediction.append(-1)
        else:
            prediction.append(predictDS(point, DS))
        truth.append(data[i][1])
    accuracy = normalized_mutual_info_score(prediction, truth)
    
    with open(output_file, 'a') as file:
        file.write('\n')
        file.write('The clustering results:\n')
        for i in range(len(prediction)):
            file.write(str(i) + ',' + str(prediction[i]) + '\n')
                
    end = time.time()
    print('Process exited successfully. Time elapsed = ' + str(end - start) + ' seconds. Accuracy = ' + str(accuracy))

        
    