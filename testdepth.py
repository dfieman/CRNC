import sys
import math
import numpy as np
import itertools
from scipy.special import gamma, factorial
from matplotlib import pyplot as plt
from itertools import product
import random
from osgeo import gdal
import datetime
import time
import multiprocessing
from functools import partial
import numba
from numba import jit
#Function for placing landslides

@jit(nopython = True)
def landslideMasker(lslength,lsdepth,depthDEM,slopeDEM,slopeVALUE,mask):
    row = []
    col = []
    for i in range (0, np.shape(slopeDEM)[0]):
        for j in range (0, np.shape(slopeDEM)[1]):
            if i+lslength <= np.shape(slopeDEM)[0] and j+lslength <= np.shape(slopeDEM)[1]:
                landslideMatrix = slopeDEM[i:i+lslength, j:j+lslength]
                if landslideMatrix.min() >= slopeVALUE:
                    maskMatrix = mask[i:i+lslength, j:j+lslength]
                    if np.all(maskMatrix == 0): #for not overlapping landslides
                        row.append(i)
                        col.append(j)
                        #print(row)
                        #print(col)
    if len(row) > 0:
        row_index = np.array(list(range(len(row))))#for randomising which index to use for landslide
        #print(row_index)
        rand_index = np.random.choice(row_index,size=1)
        #print(rand_index)
        #For slicing
        depthDEM[row[rand_index[0]]:row[rand_index[0]]+lslength, col[rand_index[0]]:col[rand_index[0]]+lslength] = lsdepth
        #Replacement for mask so not overlapping
        mask[row[rand_index[0]]:row[rand_index[0]]+lslength, col[rand_index[0]]:col[rand_index[0]]+lslength] = 1
    return depthDEM, mask

if __name__ == '__main__':

    slope_value = 20
    test_slope = np.arange(49000000).reshape(7000,7000)
    test_surface = np.zeros_like(test_slope,dtype="float")
    mask_array = np.zeros_like(test_surface,dtype="int")
    lengths = np.arange(1,11)
    print(lengths)
    depths = np.arange(10)
    #print(depths)
    #cpu_pool = multiprocessing.Pool(processes=250)
    starttime = time.process_time()
    
    #results = cpu_pool.starmap(partial(landslideMasker, depthDEM = test_surface, slopeDEM = test_slope, slopeVALUE = slope_value, mask=mask_array), zip(lengths,depths))
    #print(results)

    for lslength, lsdepth in zip(lengths, depths):
        landslide_array, new_mask_array = landslideMasker(lslength, lsdepth, test_surface, test_slope, slope_value, mask_array)
    #print(landslide_array)
    #print(new_mask_array)
    print(time.process_time()-starttime)
