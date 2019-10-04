import numpy as np 
import sys
import os

def get_idx(col):
    if(col == 1):
        return 0
    elif(col == 2):
        return 1
    elif(col == 3):
        return 2
    else:
        return 3

file = 'kddcup.data'
with open(file, 'r') as f:
    l = [[str(num) for num in line.split(',')] for line in f]

#columns which are not numerical.
word_index = [1, 2, 3, 41]
categories = [[], [], [], []]

for i in l:
    for w in word_index:
        if(i[w] in categories[get_idx(w)]):
            continue
        else:
            categories[get_idx(w)].append(i[w])

print(str(categories))
