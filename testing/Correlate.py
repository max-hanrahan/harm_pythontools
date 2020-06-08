###Megan Marshall
###Last updated: November 17th, 2014
###Function to correlate two series with a buffer taken from the data series itself

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
import scipy as sp
from scipy.interpolate import griddata
import datetime as dt

###Read in quantities calculated in Jon's __init__.py
filename=raw_input("Enter name of file with quantities from __init__.py: ")
anls=np.load(filename)
alphamag=anls['alphamag'] #Maxwell Stress
alphamagp=anls['alphamagp'] #Maxwell stress from perturbed velocities
alphareynolds=anls['alphareynolds'] #Reynolds stress
bz=anls['bz'] #vertical magnetic field
absbz=anls['absbz'] #absolute value of the vertical magnetic field

ST=alphamag+alphareynolds #total stress
SP=alphamagp+alphareynolds #stress due to perturbations of magnetic field/velocity

P=115 ###Keplarian orbit for r=11r_g scaled to timestep

def corr(a,b,t1,window,shift):
    '''A function to correlate two data arrays for a given window of interest shifted by some lag'''
    #do I need to account for size when the time window I want has more indices (time not in integer steps)? Will affect scaling on lag
    
    ## a, b are the functions to be correlated
    ## t1 is the beginning of the interval being examined
    ## window is the size of the portion being examined
    ## shift is how far back and forth the functions will move
    ## If a leads b, correlation is maximized at positive lag
    ## If b leads a, correlation is maximized at negative lag
    t2=window+t1
    corr=np.zeros((2*shift+1,),dtype=np.float32)  ## using dtype=float32 because that's what comes from __init__.py
    corr1=np.zeros((2*shift+1,),dtype=np.float32)
    lag=np.zeros((2*shift+1,),dtype=np.float32)
    auto_a=np.zeros((2*shift+1,),dtype=np.float32)
    auto_b=np.zeros((2*shift+1,),dtype=np.float32)
   
    for i in range(0,2*shift+1):
        lag[i]=2*(i-shift)
        
    for i in range(0,2*shift+1):
        #amax=a[t1-0.5*lag[i]:t2-0.5*lag[i]].max()
        #bmax=b[t1+0.5*lag[i]:t2+0.5*lag[i]].max()
        anorm=a#/amax
        bnorm=b#/bmax
        corr[i]=np.sum(anorm[t1-0.5*lag[i]:t2-0.5*lag[i]]*bnorm[t1+0.5*lag[i]:t2+0.5*lag[i]])
	cor=corr
	
        corr1[i]=np.sum(a[t1-0.5*lag[i]:t2-0.5*lag[i]]*b[t1+0.5*lag[i]:t2+0.5*lag[i]])
        auto_a[i]=np.sum(a[t1-0.5*lag[i]:t2-0.5*lag[i]]**2)
	auto_b[i]=np.sum(b[t1+0.5*lag[i]:t2+0.5*lag[i]]**2)
	norm1=np.sqrt(auto_a*auto_b)
	cor1=corr1/norm1
	

    return lag, cor, cor1

def gl_corr(a,b,t1,window,shift):
    '''A function to normalize a global correlation of two data series'''
    lag, cor, f= corr(a,b,t1,window,shift)
    junk, auto_a, f = corr(a,a,t1,window,0)
    junk, auto_b, f = corr(b,b,t1,window,0)
    norm=np.sqrt(auto_a*auto_b)
    cor=cor/norm

    return lag,cor

def glcorr(shift):
    '''Function to calculate and save the correlation functions I want'''
    gl_ti=shift+1 #to avoid using t=0 fieldline field data
    gl_wind=len(ST)-2*shift-1
    s=int(shift)
    
    lag, st_bz_g = gl_corr(ST,bz,gl_ti,gl_wind,shift)
    junk, st_abz_g = gl_corr(ST,absbz,gl_ti,gl_wind,shift)
    junk, sp_bz_g = gl_corr(SP,bz,gl_ti,gl_wind,shift)
    junk, sp_abz_g = gl_corr(SP,absbz,gl_ti,gl_wind,shift)

    corrs=open("gl_corrs"+s+".npz","w")
    np.savez(corrs,lag=lag,st_bz_g=st_bz_g,st_abz_g=st_abz_g,sp_bz_g=sp_bz_g,sp_abz_g=sp_abz_g)
    corrs.close()

def loccorr(loc_ti,loc_wind,shift):
    '''Function to calculate and save the correlation functions I want'''
    #Will be annoying in the future if I want multiple local correlations

    lag, junk, st_bz_l = corr(ST,bz,loc_ti,loc_wind,shift)
    junk, junk, st_abz_l = corr(ST,absbz,loc_ti,loc_wind,shift)
    junk, junk, sp_bz_l = corr(SP,bz,loc_ti,loc_wind,shift)
    junk, junk, sp_abz_l = corr(SP,absbz,loc_ti,loc_wind,shift)

    fname=raw_input("Enter name for correlation file: ")
    corrs=open(fname,"w")
    np.savez(corrs,lag=lag,st_bz_l=st_bz_l,st_abz_l=st_abz_l,sp_bz_l=sp_bz_l,sp_abz_l=sp_abz_l)
    corrs.close()

#plt.xlabel(r'Time (t/$\tau$)') #this just useful code for plotting, not really necessary for this script
