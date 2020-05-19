#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:56:15 2020

@author: mzp06256
"""
import sys
import time
from pyspark import SparkContext
from collections import defaultdict, deque
from copy import deepcopy

    
class Girvan_Newman:
    def __init__(self, adj_dict, betweenness):
        self.seen = set()
        self.adj_dict = adj_dict
        self.betweenness = betweenness
        self.seen = set()
        
    def generate_rooted_graph(self, start_node):
        Graph = defaultdict(lambda : defaultdict(list))
        current_lvl = [start_node]
        seen = set()
        while current_lvl:
            next_lvl = []
            for node in current_lvl:
                if node not in seen:
                    seen.add(node)
                    for neighbor in self.adj_dict[node]:
                        if neighbor not in seen and neighbor not in current_lvl:
                            next_lvl.append(neighbor)
                            Graph[node]['children'].append(neighbor)
                            Graph[neighbor]['parents'].append(node)
            current_lvl = next_lvl
            
        # do another bfs to find shortest path
        seen = set()
        Graph[start_node]['num_paths'] = 1
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            seen.add(node)
            if Graph[node]['parents']:
                Graph[node]['num_paths'] = sum([Graph[parent]['num_paths'] for parent in Graph[node]['parents']])
            for neigh in self.adj_dict[node]:
                if neigh not in seen:
                    queue.append(neigh)
        return Graph
    
    
    def betweenness_calc(self, node, Graph, p):
        
        if not p:
            retval = (sum([self.betweenness_calc(child, Graph, node) for child in Graph[node]['children']]) + 1)
            return        
        
        if not Graph[node]['children'] and not Graph[node]['parents']:
            return
        
        if not Graph[node]['children']:
            demoninator = sum([Graph[parent]['num_paths'] for parent in Graph[node]['parents']])
            numerator = Graph[p]['num_paths']
            credit = 1
            retval = credit*numerator/demoninator
            if node not in self.seen:
                self.seen.add(node)
                for parent in Graph[node]['parents']:
                    numerator = Graph[parent]['num_paths']
                    self.betweenness[frozenset({node, parent})] += credit * numerator/demoninator
            return retval
        
        else:
            demoninator = sum([Graph[parent]['num_paths'] for parent in Graph[node]['parents']])
            numerator = Graph[p]['num_paths']                
            credit = sum([self.betweenness_calc(child, Graph, node) for child in Graph[node]['children']]) + 1
            retval = credit*numerator/demoninator
            if node not in self.seen:
                self.seen.add(node)
                for parent in Graph[node]['parents']:
                    numerator = Graph[parent]['num_paths']
                    self.betweenness[frozenset({node, parent})] += credit * numerator/demoninator
            return retval
    
    def get_total_betweenness(self):
        for node in self.adj_dict:
            Graph = self.generate_rooted_graph(node)
            self.betweenness_calc(node, Graph, None)
            self.seen = set()
        for edge in self.betweenness:
            self.betweenness[edge] /= 2
        return self.betweenness
    

def modularity(Aij, m, adj_dict, original_adj_dict):
    Q = 0
    S = []
    seen = set()
    nodes = [key for key in adj_dict]
    
    for node in nodes:
        if node not in seen:
            S.append(set())
            queue = deque([node])
            while queue:
                current = queue.popleft()
                seen.add(current)
                S[-1].add(current)
                neighbors = adj_dict[current]
                for neigh in neighbors:
                    if neigh not in seen:
                        seen.add(neigh)
                        queue.append(neigh)
    
    for s in S:
        for i in range(len(s)):
            for j in range(len(s)):
                node_i = list(s)[i]
                node_j = list(s)[j]
                k_i = len(original_adj_dict[node_i])
                k_j = len(original_adj_dict[node_j])
                
                Q += Aij[int(node_i)-1][int(node_j)-1] - k_i*k_j/2/m
    Q /= (2*m)
    return Q, S
        
            
    

if __name__ == "__main__":
#    os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.6.0-spark2.3- s_2.11"
    # testing in IDE
    if len(sys.argv) == 1:
        input_file_path = "power_input.txt"
        betweenness_output_file_path = "betweenness.txt"
        community_output_file_path = "community.txt"
    elif len(sys.argv) != 4:
        print("Usage: spark-submit task2.py <input_file_path> <betweenness_output_file_path> <community_output_file_path>")
        exit(1)
    else:
        input_file_path = sys.argv[1]
        betweenness_output_file_path = sys.argv[2]
        community_output_file_path = sys.argv[3]
                
    start = time.time()
    
    sc = SparkContext('local[*]', 'mzma_task2')
    sc.setLogLevel("WARN")
        
    edge_rdd = sc.textFile(input_file_path).\
                map(lambda x: x.split()).\
                map(lambda x: (x[0], x[1]))
    edges = edge_rdd.collect()
    sc.stop()

    for i in range(len(edges)):
        edges[i] = frozenset(edges[i])
    
    
    adj_dict = defaultdict(list)
    for edge in edges:
        adj_dict[tuple(edge)[0]].append(tuple(edge)[1])
        adj_dict[tuple(edge)[1]].append(tuple(edge)[0])
    original_adj_dict = deepcopy(adj_dict)
    
    
    gn = Girvan_Newman(adj_dict, defaultdict(int))
    betweenness = gn.get_total_betweenness()
    betweenness = [[tuple(k),v] for (k,v) in betweenness.items()]
    for i in range(len(betweenness)):
        betweenness[i][0] = sorted(betweenness[i][0])
    
    betweenness.sort(key = lambda x: (-x[1], x[0][0]))
        
    with open(betweenness_output_file_path, 'w+') as file:
        for x in betweenness:
            file.write('(\'' + x[0][0] + '\', \'' + x[0][1] + '\'), ' + str(x[1]) + '\n')
    
    
    m = len(edges)
    # make Aij
    Aij = [[0]*len(adj_dict) for _ in range(len(adj_dict))]
    for node in adj_dict:
        i = int(node)-1
        for neighbor in adj_dict[node]:
            j = int(neighbor) - 1
            Aij[i][j] = 1
            
    Q_max, communities = modularity(Aij, m, adj_dict, original_adj_dict)

    Q_list = [Q_max]
    while (betweenness):
        edge, between = betweenness[0]
        
        adj_dict[edge[0]].remove(edge[1])
        adj_dict[edge[1]].remove(edge[0])
        
        # need to calculate betweenness again
        gn = Girvan_Newman(adj_dict, defaultdict(int))
        betweenness = gn.get_total_betweenness()
        betweenness = [[tuple(k),v] for (k,v) in betweenness.items()]
        for i in range(len(betweenness)):
            betweenness[i][0] = sorted(betweenness[i][0])
        
        betweenness.sort(key = lambda x: (-x[1], x[0][0]))
        
        Q, S = modularity(Aij, m, adj_dict, original_adj_dict)  
        print('Current Q is ' + str(Q))
        Q_list.append(Q)
        if Q > Q_max:
            Q_max = Q
            communities = S

    for i in range(len(communities)):
        communities[i] = sorted(list(communities[i]))
    communities.sort(key = lambda x: (len(x), x[0][0]))
    
    with open(community_output_file_path, 'w+') as file:
        for community in communities:
            file.write('\''+'\', \''.join(community)+'\'' + '\n')
    
        
    end = time.time()
    print(str(len(communities)) + ' communities discovered')
    print('process exited successfully. Time elapsed = ' + str(end - start) + ' seconds.')