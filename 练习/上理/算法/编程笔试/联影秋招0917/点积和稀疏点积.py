import re


a = list(range(10))

b = list(range(10))

def dot(a, b):
    temp = 0
    for idx in range(len(a)):
       
        temp += a[idx]*b[idx]
    
    return temp

print(dot(a, b))


import numpy as np