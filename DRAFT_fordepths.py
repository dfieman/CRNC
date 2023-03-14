#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#Draft code on determining depths"
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
import numba
from numba import jit
import time

#Function for placing landslides:

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

    filename = "Hapuku_FocalSt_Radius11_DSM_2015.tif" #input filename of DEM
    folder = "/Users/home/fiemandi/Rasters/" #where you want output DEMs to be saved

    elevation_raster = gdal.Open(folder+filename)
    nodata = elevation_raster.GetRasterBand(1).GetNoDataValue()
    elevation_array = elevation_raster.GetRasterBand(1).ReadAsArray()
    elevation_array[elevation_array==nodata]=np.nan
    driver = gdal.GetDriverByName("GTiff")
    surface_array = np.zeros_like(elevation_array)
    slopeDEM = gdal.DEMProcessing('slope.tif',elevation_raster,"slope",computeEdges=True) #default of slope raster is degrees
    slope_array = slopeDEM.GetRasterBand(1).ReadAsArray()
    nodata_s = slopeDEM.GetRasterBand(1).GetNoDataValue()
    print("computed slope DEM")
    slope_array[slope_array==nodata_s]=np.nan

    if np.shape(slope_array) == np.shape(surface_array):
        print("Shape of DEM is:", np.shape(surface_array))
    else:
        ("Slope needs to be same shape as surface")
        sys.exit()


    #User inputs:
    mag = 4 #magnitude of event based on Malamud et al. (2004)
    areals_min = 50 #minimum landslide area (m^2)
    areals_max = 2.1e6 #maximum landslide area (m^2)
    N_ls = 1000 #total number of landslides in the catchment
    a = 1280 #scalar controlling lcoation of maximum probability
    s = -132 #scalar controlling exponential decay for small landslides
    p = 1.4 #scalar controlling power-law decay for medium and large landslides
    e = 0.05 #scalar used for volume (Malamud et al. 2004)
    y = 1.5 #exponent used for volume (Malamud et al. 2004)

    def pdf(x): #probabilty density function for landslides
        return (1/(a*gamma(p)))*((a/(x-s))**(p+1))*np.exp(-1*a/(x-s))

    bins = math.isqrt(N_ls) #how many landslide areas/bins in the catchment
    bin_width = (areals_max-areals_min)/bins #bin width
    ls_areas = np.geomspace(areals_min, areals_max, bins) #chosen landslide areas based on the max and min area and how many bins you want equal in logspace in m^2
    #print(ls_areas)

    #getting bin widths
    bin_width = []
    
    for i in range(0,(np.shape(ls_areas)[0])):
        if i > 0 and i < np.shape(ls_areas)[0]-1:
            width = ((ls_areas[i+1]-ls_areas[i])/2)+((ls_areas[i]-ls_areas[i-1])/2)
        elif i == 0:
            width = ((ls_areas[i+1]-ls_areas[i])/2)+(ls_areas[0]-areals_min)
        else: #i == np.shape(ls_areas)[0]-1:
            width = ((ls_areas[i]-ls_areas[i-1])/2)+(areals_max-ls_areas[i])
        bin_width.append(width)
    #print(bin_width)
    
    NperArea = np.rint(pdf(ls_areas)*N_ls*bin_width) #Number of landslides in each bin
    
    #For plotting pdf with landslide areas
    ls_area = np.linspace(0, 3000000, num = 3000000)
    plt.plot(ls_area, pdf(ls_area), color = 'red')
    plt.scatter(ls_areas, pdf(ls_areas), color = 'green')
    plt.yscale('log')
    plt.xscale('log')
    #plt.show()
    
    ls_length = np.asarray(np.sqrt(ls_areas), dtype="int") #length of landslides in meters
    
    #creates list of each landslide
    lengthLS = []
    for l in range(0,len(ls_length)):
        each_len = np.repeat(ls_length[l],NperArea[l])
        for n in each_len:
            lengthLS.append(n)
            random.shuffle(lengthLS)#shuffles list so not landslide area is not biased
    
    #print(lengthLS)
    depthLS = (e*((np.array(lengthLS)**2)**y))/(np.array(lengthLS)**2) # m - based on the volume-area scaling in Malamud 2004
    #print(depthLS)
    #Checks:
    if len(depthLS) == len(lengthLS) and len(lengthLS) == int(np.sum(NperArea)):
        print("Checks complete")
        print("Total number of landslides is:",len(lengthLS))
    else:
        print("Check dimension of depth and length arrays")
    
    #Locating coordinates where slope is >20. This is a test. Need to add DEM
    
    mask_array = np.zeros_like(slope_array,dtype="int")
    slope_value = 20.0 #slope value of failure
    length_test = [30,25,7]
    depth_test = [30.0,25.8,7.4]
    
    starttime = time.process_time()
    
    #list of depths = test_depth
    for lslength, lsdepth in zip(lengthLS, depthLS):#change these
        landslide_array, landslide_placements = landslideMasker(lslength, lsdepth, surface_array, slope_array, slope_value, mask_array)
    
    print("function time:", time.process_time()-starttime)
    
    #Convert surface array  to a tif:
    driver = gdal.GetDriverByName("GTiff")
    landslide_raster = driver.CreateCopy(folder+"landslide_depths.tif", elevation_raster, strict=0,
                                             options=["TILED=YES","COMPRESS=PACKBITS"])
    landslide_band = landslide_raster.GetRasterBand(1).ReadAsArray()
    landslide_band = landslide_array
    landslide_band[np.isnan(landslide_band)] = nodata
    landslide_raster.GetRasterBand(1).WriteArray(landslide_band)
    
    # #This code if searching from middle point####
    # radius = 2 #needs to be ((ls_depth/cell_size)-1)/2 radius is how many cells above and below i,j
    # for i in range(0,np.shape(test_matrix)[0]):
        # for j in range (0, np.shape(test_matrix)[1]):
            # if i-radius >= 0 and i+radius =< np.shape(test_matrix)[0] and j-radius >= 0 and j+radius =< np.shape(test_matrix)[1]:
                # try:
                    # temporary_matrix = test_matrix[(i-radius):(i+radius+1),(j-radius):(j+radius+1)] #add +1 because it does not include that value!
                    # print(temporary_matrix)
                    # if temporary_matrix.min() >= value:
                        # row.append(i)
                        # col.append(j)
                # except Exception as e:
                    # print(e)
    
    # print(row)
    # print(col)
    
    
    # #This code if searching from corner
    # radius = 3 #This is the actual length (m) of the landslide. Need to consider cell size too
    # for i in range(0, np.shape(test_matrix)[0]):
        # for j in range(0, np.shape(test_matrix)[1]):
            # if i+radius <= np.shape(test_matrix)[0] and j+radius <= np.shape(test_matrix)[1]:
                # temporary_matrix = test_matrix[i:i+(radius),j:j+(radius)]
                # #print(temporary_matrix)
                # if temporary_matrix.min() >= value:
                    # row.append(i)
                    # col.append(j)
    # print(row)
    # print(col)
    
    
    
