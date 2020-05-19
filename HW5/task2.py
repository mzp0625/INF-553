#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:25:12 2020

@author: mzp06256
"""

import sys
from blackbox import BlackBox
import random
import binascii
import math
from statistics import median, mean
import time 

if __name__ == "__main__":

    if len(sys.argv) == 1:
        input_filename = 'users.txt'
        stream_size = 300
        num_of_asks = 30
        output_filename = 'task2_out.csv'
    elif len(sys.argv) != 5:
        print('Usage : python task1.py <input_filename> stream_size num_of_asks <output_filename>')
        exit(1)
    else:
        input_filename = sys.argv[1]
        stream_size = int(sys.argv[2])
        num_of_asks = int(sys.argv[3])
        output_filename = sys.argv[4]
        
    start = time.time()
    
    num_hash = 100
    print('num hash:' + str(num_hash))
    
    p = [90203,84179,97577,78691,90641,96293,88513,74453,99529,74759,87691,88211,77347,91291,76253,95483,83939,86627,86111,71153,99367,83597,82483,91813,77047,70621,96401,90977,72167,89069,72493,86501,95101,71633,80273,84463,87943,84137,91939,99577,87553,97673,70297,71597,70919,99901,78487,99257,96553,83561,86381,74719,73351,77659,83833,72817,75931,83813,86857,81647,95621,90803,95413,89989,77141,76213,76387,94121,79757,99607,86689,76289,89021,77431,76157,76003,70139,96737,89797,81041,74231,83063,71209,70379,73597,75629,90533,75403,99241,73523,90007,72353,96461,89041,98573,89959,96907,87743,91577,91951]
    
    a = []
    b = []
    
    
    while len(a) != num_hash:
        num = random.randrange(1,69997) 
        if num not in a:
            a.append(num)
            
    while len(b) != num_hash:
        num = random.randrange(-69997,69997)
        if num not in a and num not in b:
            b.append(num)
            
    def hash_func(x,i):
        global a, b, p
        return bin((a[i]*x + b[i])% p[i]%69997)[2:]

    hash_function_list = [hash_func]*num_hash
    
    def myhashs(s):
        result = []
        x = int(binascii.hexlify(s.encode('utf8')),16)
        for i,f in enumerate(hash_function_list):
            result.append(f(x, i))
        return result
    
    def num_trailing_zeros(s):
        return len(s) - len(s.strip('0'))
    
    def combine_hash(max_trailing_zeros_list):
        return round(2**mean(max_trailing_zeros_list))

    bx = BlackBox()
    with open(output_filename, 'w+') as file:
        file.write('Time,Ground Truth,Estimation\n')
        for i in range(num_of_asks):
            max_trailing_zeros_list = [0]*num_hash
            data = bx.ask(input_filename, stream_size)
            seen = set()
            for s in data:
                seen.add(s)
                result = myhashs(s)
                for j in range(len(result)):
                    max_trailing_zeros_list[j] = max(max_trailing_zeros_list[j],num_trailing_zeros(result[j])) 
                    num_unique = combine_hash(max_trailing_zeros_list)
            print(str(i), str(len(seen)) ,str(num_unique))
            file.write(str(i)+','+str(len(seen))+','+str(num_unique)+'\n')
    
    end = time.time()
    
    print('Process exited successfully. Time elapsed = ' + str(end - start) + ' seconds')