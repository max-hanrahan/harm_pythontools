import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
import scipy as sp
from scipy.interpolate import griddata
import datetime as dt
import matplotlib as mpl
from matplotlib.colors import Normalize
import __init__

__init__.main()

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_slice(fnumber, phi, cap, floor, IC=True, stress=True):
    '''Plotting function to visualize 2d slices of the disc                                                                                                                                                  
    fnumber = which fieldline file to load                                                                                                                                                                   
    phi = index of position to take slice at                                                                                                                                                                 
    cap = upper bound on quantity being plotted (if unwanted, use absurdly high value)                                                                                                                       
    floor = lower bound on quantity being plotted (if unwanted, use absurdly low value)                                                                                                                      
    IC = True: using the internal coordinates (Boyer-Lindquist)                                                                                                                                              
    IC = False: use spherical polar coordinates                                                                                                                                                              
    stress = True: plots stress terms (currently just Maxwell 6/12/15)                                                                                                                                       
    stress = False: plots vertical magnetic field                                                                                                                                                            
    radii of slice chosen with nxin, nxout'''

    # first load grid file
    global use2dglobal
    use2dglobal=True
    __init__.grid3d("gdump.bin", use2d=use2dglobal)
    # now try loading a single fieldline file                                                                        
    __init__.rfd("fieldline"+str(fnumber)+".bin")
    # now plot something you read-in                                                                                 
    plt.clf()
    plt.figure(1)
 #                                                                                                                
    ###############################                                                                                  
    if 1==1:
        (rhoclean,ugclean,uublob,maxbsqorhonear,maxbsqorhofar,condmaxbsqorho,condmaxbsqorhorhs,rinterp)=__init__.getrhouclean(rho,ug,uu)
        __init__.cvel()                                                     
        #                                                                                                            
        diskcondition=condmaxbsqorho
        # only around equator, not far away from equator                                                             
        diskcondition=diskcondition*(bsq/rho<1.0)*(np.fabs(h-np.pi*0.5)<0.1)                                                                 
        diskeqcondition=diskcondition                                                                                                                                                                       
    ##############################
    ###choose the radial extent of the plot
    nxin=__init__.iofr(5)
    nxout=__init__.iofr(30)
    if IC:
        ###using internal coordinates
        myx=r[nxin:nxout,:,0]
        myy=ph[nxin:nxout,:,0]
        myz=h[nxin:nxout,:,0]
    else:
        ###use spherical polar
        myx=r[nxin:nxout,:,0]*np.sin(h[nxin:nxout,:,0])*np.cos(ph[nxin:nxout,:,0])                                      
        myy=r[nxin:nxout,:,0]*np.sin(h[nxin:nxout,:,0])*np.sin(ph[nxin:nxout,:,0])                                      
        myz=r[nxin:nxout,:,0]*np.cos(h[nxin:nxout,:,0])
    #############################
    if stress:
        numMag=jabs(-bu[1]*np.sqrt(gv3[1,1])*bd[3]*np.sqrt(gn3[3,3]))
        denMR=(bsq*0.5+(gam-1.0)*ug)
        amag=numMag/denMR
        avgexists=checkiffullavgexists()
        if avgexists==1:
            loadavg()
            loadedavg=1
            numRey=jabs(rho*(uu[1]-avg_uu[1])*np.sqrt(gv3[1,1])*(ud[3]-avg_ud[3])*np.sqrt(gn3[3,3]))
        else:
            numRey=jabs(rho*(uu[1])*np.sqrt(gv3[1,1])*(ud[3])*np.sqrt(gn3[3,3]))
        numRey=jabs(rho*(uu[1])*np.sqrt(gv3[1,1])*(ud[3])*np.sqrt(gn3[3,3]))
        arey=numRey/denMR
        myfun=r*amag#+arey                                                                                                                                                                                 
        myfun[myfun<=floor]=floor                                                                                   
        myfun[myfun>=cap]=cap                                                                                       

    else:
        myfun=r*bu[2]*np.sqrt(gv3[2,2]) #quasi-orthonormal                                                                                                                                               
        myfun[myfun<=floor]=floor
        myfun[myfun>=cap]=cap
    #
    #######################################                                                                          
    ax = plt.gca()
    ax.pcolor(myx,myz,myfun[nxin:nxout,:,phi],norm=[None,MidpointNormalize(midpoint=0)][plot>=1])  #try pcolormesh - faster                                               
    plc(myfun[nxin:nxout,:,phi],xcoord=myx,ycoord=myz,ax=ax,cb=True,nc=50) #nc = number of contour
    ax.grid(linestyle='-')
    #print("cap="+str(cap))                                                                                          
    #print("floor="+str(floor))
    if stress:
        plt.savefig('amag_f'+str(fnumber)+'phi'+str(ph)+'.png')
    else:
        plt.savefig('bz_f'+str(fnumber)+'phi'+str(ph)+'.png')

plot_slice(4000,0,IC=True,stress=True)
