#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:25:31 2020

@author: mzp06256
"""

import sys
from blackbox import BlackBox
import random

if __name__ == "__main__":

    
    if len(sys.argv) == 1:
        input_filename = 'users.txt'
        stream_size = 100
        num_of_asks = 30
        output_filename = 'task3_out.csv'
    elif len(sys.argv) != 5:
        print('Usage : python task1.py <input_filename> stream_size num_of_asks <output_filename>')
        exit(1)
    else:
        input_filename = sys.argv[1]
        stream_size = int(sys.argv[2])
        num_of_asks = int(sys.argv[3])
        output_filename = sys.argv[4]
    
    random.seed(553)
    
    saved_users = []
    bx = BlackBox()
    n = 0
    print('seqnum, 0_id, 20_id, 40_id, 60_id, 80_id\n')
    
    with open(output_filename, 'w+') as file:
        file.write('seqnum, 0_id, 20_id, 40_id, 60_id, 80_id\n')
        for i in range(num_of_asks):
            data = bx.ask(input_filename, stream_size)
            for s in data:
                n += 1
                if n <= 100:
                    saved_users.append(s)
                elif random.randint(0, 100000) % n < 100:
                    index = random.randint(0, 100000) % 100
                    saved_users[index] = s
            print((i+1)*stream_size, saved_users[0], saved_users[20],saved_users[40],saved_users[60],saved_users[80])
            file.write(str((i+1)*stream_size) + ',' + str(saved_users[0]) + ',' + str(saved_users[20]) + ',' + str(saved_users[40]) + ',' + str(saved_users[60]) + ',' +str(saved_users[80]) + '\n')            
