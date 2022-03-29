import os, glob
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import pandas as pd


def extract(filename):
    '''
    This function extracts the the q and intensity values from the filename and outputs a matrix of (n,2) Q vs I
    '''

    # open files for extraction 
    f = open(filename, 'r')

    q = []
    I = []
    for i in f.readlines():
        dat =i.split(' ')
        if len(dat) > 4:
            continue

        q.append(float(dat[0]))
        I.append(float(dat[2]))

    q = np.array(q)
    I = np.array(I)

    data = np.concatenate((q.reshape(-1,1),I.reshape(-1,1)), axis = 1)

    return data