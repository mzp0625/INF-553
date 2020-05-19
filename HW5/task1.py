#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:06:40 2020

@author: mzp06256
"""

import sys
from blackbox import BlackBox
import random
import binascii
import math
 


if __name__ == "__main__":

    if len(sys.argv) == 1:
        input_filename = 'users.txt'
        stream_size = 100
        num_of_asks = 60
        output_filename = 'task1_out.csv'
    elif len(sys.argv) != 5:
        print('Usage : python task1.py <input_filename> stream_size num_of_asks <output_filename>')
    else:
        input_filename = sys.argv[1]
        stream_size = int(sys.argv[2])
        num_of_asks = int(sys.argv[3])
        output_filename = sys.argv[4]
        
    num_hash = int(round(math.log(2)*(69997/num_of_asks/stream_size)))
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
        return (a[i]*x + b[i])% p[i]%69997
    
    hash_function_list = [hash_func]*num_hash
    
    def myhashs(s):
        result = []
        x = int(binascii.hexlify(s.encode('utf8')),16)
        for i,f in enumerate(hash_function_list):
            result.append(f(x, i))
        return result
        
        
    A = [0]*69997
    
    m = 69997
    
    bx = BlackBox()
    
    seen = set()
        
    # If for some hash function hj(o’) = i and A[i] = 0, stop and report o’ not in S
    
    with open(output_filename, 'w+') as file:
        file.write('Time,FPR\n')
        for i in range(num_of_asks):
            fp = 0
            tn = 0
            data = bx.ask(input_filename, stream_size)
            for s in data:
                flag = True
                result = myhashs(s)
                for j in result:
                    if A[j] == 0:
                        A[j] = 1
                        flag = False
                if flag and s not in seen:
                    fp += 1
                elif not flag:
                    tn += 1
                seen.add(s)
            fpr = fp/(tn + fp)
            print(i, fpr)
            file.write(str(i) + ',' + str(fpr) + '\n')
        
                
    
    
    