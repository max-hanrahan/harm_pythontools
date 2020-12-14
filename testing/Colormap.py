###Megan Marshall
###Last Updated: January 28th, 2015
###Function to create 2D color maps from a 2D array

import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
import scipy as sp
from scipy.interpolate import griddata
import datetime as dt
import matplotlib as mpl
#filename=raw_input("Enter name of file with quantities from __init__.py: ")

###For Stampede###
#data=np.load('out0.npz')
#numM=data['numMag']
#numR=data['numRey']
#denMR=data['denMR']
#bz=data['bz']
#amag=numM/denMR
#arey=numR/denMR

data=np.load("2dout2.npz")
stress=data['str2d']
bz=data['bz2d']
abz=data['abz2d']

str=stress[20:123,:]
b_z=bz[20:123,:]
ab_z=abz[20:123,:]

plt.clf()
#figure(1)
plt.subplot(121)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['blue','orange','red'],256)
img_str=plt.imshow(stress.T,interpolation='nearest',cmap=cmap2,origin='lower')
plt.colorbar(img_str,cmap=cmap2)
plt.title("Stress")
plt.xlabel("r")
plt.ylabel("theta")
#plt.savefig('ColormapTest.png')

#figure(1) #not actually making new plot but printing over stress plot
plt.subplot(122) #subplots work, but look weird
img_bz=plt.imshow(abz.T,interpolation='nearest',cmap=cmap2,origin='lower')
plt.colorbar(img_bz,cmap=cmap2)
plt.title("|b_z|")
plt.xlabel("r")
plt.ylabel("theta")
#plt.savefig('plot_out2_abz.png')
#plt.clf()

