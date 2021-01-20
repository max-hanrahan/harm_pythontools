###Simplified code with only the functions needed to replicate Marshall, McKinney, and Avara 2018
###needs a lot of commenting and to updated to work with Python3 (Megan 6/5/20)

###Initializing global variables to conform to python3 syntax (Max 12/14/20)
global numcolumns
global nzgdump

def setpythonpath():
    # PYTHONPATH from os environment might include arbitrary paths, including those not accessible on supercomputer by a compute node, so set manually
    # Assumes if user needs local "py" that copied to local directory, then force use of that version.
    # Can't allow module to not exist or else would search outside local directory and could find (e.g.) home that Kraken doesn't allow.
    import os,sys,inspect
    #
    global pythonpath
    if os.path.exists("py")==1:
        print(("Original PYTHONPATH=%s" % (os.environ['PYTHONPATH'])))
        pythonpath=os.path.abspath("py")
        # cmd_folder = os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])
        os.environ['PYTHONPATH']=pythonpath
        #sys.path.append(pythonpath)
        sys.path.insert(0, pythonpath)
        print(("New PYTHONPATH=%s" % (os.environ['PYTHONPATH'])))
        print(("New sys.path=%s" % (sys.path)))
        # now if import, will look in local "py" path first
        sys.stdout.flush()
    else:
        print(("No pythonpath found, assuming PYTHONPATH=%s will work." % (os.environ['PYTHONPATH'])))
        sys.stdout.flush()
        pythonpath=""
    #
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':/nics/b/home/jmckinne/bin'
    #
    ISKRAKEN=1
    # set latex path, else on Kraken can't find "latex"
    if ISKRAKEN==1:
        # user must set their own path to "latex" binary
        os.environ['PATH'] = os.environ['PATH'] + ':/nics/b/home/jmckinne/bin'
        # or have to link to latex binary in run and PYTHONPATH dirs.
        # cd $PYTHONPATH ; ln -s ~/bin/latex .
        # cd rundir/moviedir/ ; ln -s ~/bin/latex .

def getpythonpath():
    import os
    localpythonpath=os.environ['PYTHONPATH'].split(os.pathsep)

def setmplconfigpath(uniquenum=None):
    #
    # SUPERNOEMARK: below inapplicable to Nautilus for some reason.  Makes Nautilus fail to find some packages if MPLCONFIGDIR not unset.
    #
    ISNAUTILUS=1 # sometimes physics-179.umd.edu needs this for one file for some reason. (#11 out of 16 usually)
    #
    if ISNAUTILUS==1:
        return
    #
    # also need to set MPLCONFIGDIR to something unique, otherwise (e.g.) Kraken can complain that "File exists" when matplotlib calls mkdir(name,mode).
    import os,sys
    try:
        # below also removes trailing /
        origmplconfigdir=os.path.abspath(os.environ['MPLCONFIGDIR'])
        print(("Original MPLCONFIGDIR=%s" % (origmplconfigdir))) ; sys.stdout.flush()
        mycwd=origmplconfigdir + "%d" % (uniquenum)
        print(("New MPLCONFIGDIR=%s" % (mycwd))) ; sys.stdout.flush()
    except KeyError:
        print("No original MPLCONFIGDIR") ; sys.stdout.flush()
        # below always has no trailing /
        mycwd=os.getcwd() + "/maplotlibdir" + "%s" % (uniquenum)
        print(("New MPLCONFIGDIR=%s" % (mycwd))) ; sys.stdout.flush()
    #
    os.environ['MPLCONFIGDIR'] = mycwd
    #os.environ['MPLCONFIGDIR'] = os.environ['MPLCONFIGDIR']# + ':/nics/b/home/jmckinne/bin'
    # now MPLCONFIGDIR is unique, so remove this dir and let matplotlib create it -- else Kraken complains.
    if os.path.isdir(mycwd)==1:
        import shutil
        shutil.rmtree(mycwd)
    #
    try:
        # try importing matplotlib.  If error, then create directory myself (Nautilus vs. Kraken issue)
        import matplotlib
    except:
        try:
            os.makedirs(mycwd)
            print(("try mkdir for MPLCONFIGDIR=%s" % (mycwd))) ; sys.stdout.flush()
        except:
            print("Problem: matplotlib import didn't work, but directory couldn't be made") ; sys.stdout.flush()
            print(("os.path.exists: %d" % (os.path.exists(mycwd)))) ; sys.stdout.flush()
    #


# redirect stderr and stdout to unique files per runnumber if relevant
def redirectstderrout(runtype=None,uniquenum=None,uppernum=None):
    import os,sys
    mystderrname=os.getcwd() + "/python_u_%d_%d_%d.stderr.out" % (runtype,uniquenum,uppernum)
    mystdoutname=os.getcwd() + "/python_u_%d_%d_%d.stdout.out" % (runtype,uniquenum,uppernum)
    #
    if os.path.exists(mystderrname)==1:
        os.remove(mystderrname)
    #
    if os.path.exists(mystdoutname)==1:
        os.remove(mystdoutname)
    #
    global oldstderr,oldstdout,newstderr,newstdout
    #newstderr = os.open(mystderrname,os.O_RDWR|os.O_CREAT)
    #newstdout = os.open(mystdoutname,os.O_RDWR|os.O_CREAT)
    newstderr = open(mystderrname,'w')
    newstdout = open(mystdoutname,'w')
    oldstderr = sys.stderr
    oldstdout = sys.stdout
    sys.stderr = newstderr
    sys.stdout = newstdout



#############################
#
# things to run when script loaded
#
##############################
def runglobalsetup(argv=None):
    import os,sys
    if argv is None:
        argv = sys.argv
    #
    #
    #
    # force python path to be set before loading rest of file, including modules.
    #setpythonpath()
    #
    global runtype
    if len(sys.argv[1:])>0:
        runtype=int(sys.argv[1])
    else:
        print("No run type specified") ; sys.stdout.flush()
        runtype=-1
    #
    global modelname
    if len(sys.argv[2:])>0:
        modelname = sys.argv[2]
    else:
        modelname = "UnknownModel"
    #
    print(("ModelName = %s" % (modelname) )) ; sys.stdout.flush()
    #
    #
    # below should agree with jon_makemovie_programstart.c.  But below used more generally.
    if(runtype==2 or runtype==3 or runtype==4):
        if len(sys.argv[4:])>0 and argv[3]!="plot":
            runnumber=int(argv[3])
            uppernum=int(argv[4])
            print(("runtype=%d has runnumber=%d uppernum=%d" % (runtype,runnumber,uppernum))) ; sys.stdout.flush()
            # force unique path or else mkdir in matplotlib will barf on some systems.
            setmplconfigpath(uniquenum=runnumber)
            redirectstderrout(runtype=runtype,uniquenum=runnumber,uppernum=uppernum)
        elif argv[3]=="plot":
            print("Doing plot type of run") ; sys.stdout.flush()
        else:
            print(("runtype=%d should have runnumber and uppernum but doesn't!" % (runtype))) ; sys.stdout.flush()
            exit
    #
    #



#######################################
#
# run things when script ran
#
#######################################
runglobalsetup()


###################################
#
# GLOBAL IMPORTS
#
###################################

# import modules necessary for this file
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
from matplotlib import mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable


from streamlines import streamplot #streamlines is from harm_pythontools, not actually a python package
from streamlines import fstreamplot

import gc
import numpy as np
from numpy import ma

import array

import scipy as sp
from scipy import fftpack
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.special import sph_harm,lpmn,gammaln
from scipy.optimize import leastsq
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

import os,glob
import pylab
import sys
import streamlines #again, from harm_pythontools
import re
from datetime import datetime

#import resource #Unix only, used for printusage
import random as rnd

# On Linux (from python cookbook http://code.activestate.com/recipes/286222/:
_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    return _VmB('VmSize:') - since

def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since

def stacksize(since=0.0):
    '''Return stack size in bytes.
    '''
    return _VmB('VmStk:') - since

#Windows only, but not really used:
#def memory():
#    import os
#    from wmi import WMI
#    w = WMI('.')
#    result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
#    return int(result[0]['WorkingSet'])

# http://docs.python.org/library/resource.html
# http://stackoverflow.com/questions/938733/python-total-memory-used
# ru_maxrss is kilobytes of total memory usage
def printusage():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    print(usage) ; sys.stdout.flush()
    #resource.struct_rusage(ru_utime=0.028000999999999998, ru_stime=0.020000999999999998, ru_maxrss=8280, ru_ixrss=0, ru_idrss=0, ru_isrss=0, ru_minflt=1601, ru_majflt=0, ru_nswap=0, ru_inblock=32, ru_oublock=0, ru_msgsnd=0, ru_msgrcv=0, ru_nsignals=0, ru_nvcsw=191, ru_nivcsw=6)
    #
    # below for unix
    memoryusage=memory()
    print(("memoryusage=%g" % (memoryusage))) ; sys.stdout.flush()
    #

###Setting common zero color in pcolor - From Joe Kington's answer here: http://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


######################################################
#
# functions created to handle THETAROT transformation and some basic math stuff
#
######################################################
def sqrt(x):
    return(np.sqrt(x))
def pow(x,y):
    return(x**y)
def sin(x):
    return(np.sin(x))
def mysin(x):
    return(sin(x))
def cos(x):
    return(np.cos(x))
def mycos(x):
    return(cos(x))
def tan(x):
    return(np.tan(x))

def cot(x):
    y=1.0/np.tan(x)
    y[np.fabs(np.mod(x,np.pi))<1E-14]=0.0
    return(y)

def csc(x):
    y=1.0/np.sin(x)
    y[np.fabs(np.mod(x,np.pi))<1E-14]=0.0
    return(y)

def sec(x):
    y=1.0/np.cos(x)
    y[np.fabs(np.mod(x,0.5*np.pi))<1E-14]=0.0
    return(y)

#arctan2 is identical to the atan2 function of the underlying C library.
def atan2(x,y):
    return(np.arctan2(x,y))

#Note that mathematica's Arctan[x,y] = C's atan2(y,x) (i.e. args are flipped in order)
def arctanmath(x,y):
    return(atan2(y,x))

def myfloat(f,acc=1):
    """ acc=1 means np.float32, acc=2 means np.float64 """
    if acc==1:
        return( np.float32(f) )
    else:
        return( np.float64(f) )

def myfloatalt(f):
    return( np.float64(f) )

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def divideavoidinf(x):
    SMALL=1E-30
    y=1.0*np.sign(x)/(np.fabs(x)+SMALL)
    return(y)

def scaletofullwedge(val):
    return(val * 2.0*np.pi/(dxdxp[3,3,0,0,0]*float(nz)*_dx3))

def mk2d3d(version2d):
    return(np.array([version2d[:,:,0].T,]*nz).T)

def getnearpos(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def mdot(a,b):
    """
    Computes a contraction of two tensors/vectors.  Assumes
    the following structure: tensor[m,n,i,j,k] OR vector[m,i,j,k],
    where i,j,k are spatial indices and m,n are variable indices.
    """
    if a.ndim == 4 and b.ndim == 4:
          c = (a*b).sum(0)
    elif a.ndim == 5 and b.ndim == 4:
          c = np.empty(amax(a[:,0,:,:,:].shape,b.shape),dtype=b.dtype)
          for i in range(a.shape[0]):
                c[i,:,:,:] = (a[i,:,:,:,:]*b).sum(0)
    elif a.ndim == 4 and b.ndim == 5:
          c = np.empty(amax(b[0,:,:,:,:].shape,a.shape),dtype=a.dtype)
          for i in range(b.shape[1]):
                c[i,:,:,:] = (a*b[:,i,:,:,:]).sum(0)
    elif a.ndim == 5 and b.ndim == 5:
          c = np.empty((a.shape[0],b.shape[1],a.shape[2],a.shape[3],max(a.shape[4],b.shape[4])),dtype=a.dtype)
          for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                      c[i,j,:,:,:] = (a[i,:,:,:,:]*b[:,j,:,:,:]).sum(0)
    else:
           raise Exception('mdot', 'wrong dimensions')
    return c

# do np.tensordot with full 3D first arg and axisym (nz=1 like) second arg
# so don't have to allocate extra memory for dxdxp
# tendordot stupid for per-point operations, and wasteful
#http://stackoverflow.com/questions/5344843/a-loopless-3d-matrix-operation-in-python
# these tensordot01 and tensor00 handle dxdxp[trans] with any nz=1 or normal full size, so can be used with dxdxp or transV2Vmetric
def tensordot01(uu,dxdxp):
    #
    result=(uu[:,None,None,None] * dxdxp[:,:,None,None,None]).sum(axis=1).reshape(4,nx,ny,nz)
    #
    return(result)

def tensordot00(uu,dxdxp):
    #
    dxdxptrans=np.transpose(dxdxp,(1,0,2,3,4))
    result = (uu[:,None,None,None] * dxdxptrans[:,:,None,None,None]).sum(axis=1).reshape(4,nx,ny,nz)
    return(result)

def intangle(qty,hoverr=None,thetamid=np.pi/2,minbsqorho=None,maxbsqorho=None,inflowonly=None,outflowonly=None,mumax=None,mumin=None,maxbeta=None,unboundonly=None,which=1,doavgn3=1):
    integrand = qty
    #
    #somehow gives slightly different answer than when computed directly
    if hoverr is not None:
        insidehor = np.abs(h-thetamid)<hoverr
    else:
        hoverr = np.pi/2
        thetamid = np.pi/2
        insidehor = 1.0
    #
    # minbsqorho to look at flow in high mag regions to approximate floor injection
    if minbsqorho != None:
        insideminbsqorho = bsq/rho>=minbsqorho
    else:
        insideminbsqorho = 1
    #
    # maxbsqorho for mdin
    if maxbsqorho != None:
        insidemaxbsqorho = bsq/rho<=maxbsqorho
    else:
        insidemaxbsqorho = 1
    #
    # inflowonly for mdin
    # NOTEMARK: Not to be used for efficiency, but only Mdot(r) reporting
    if inflowonly != None:
        insideinflowonly = uu[1]<0.0
    else:
        insideinflowonly = 1
    #
    if outflowonly != None:
        insideoutflowonly = uu[1]>0.0
    else:
        insideoutflowonly = 1
    #
    #
    #v4asq=bsq/(rho+ug+(gam-1)*ug)
    #mum1fake=uu[0]*(1.0+v4asq)-1.0
    # override (mum1fake or mu do poorly for marking boundary of jet)
    mum1fake=bsq/rho
    # mumax for wind
    # NOTEMARK: Not to be used for efficiency, but only Mdot(r) reporting
    if mumax is None:
        insidemumax = 1
    else:
        insidemumax = 1
        insidemumax = insidemumax * (mum1fake<mumax)
    #
    # mumin for jet
    # NOTEMARK: Not to be used for efficiency, but only Mdot(r) reporting
    if mumin is None:
        insidemumin = 1
    else:
        insidemumin = 1
        insidemumin = insidemumin * (mum1fake>=mumin)
    #
    if unboundonly is None:
        insideunbound = 1
    else:
        insideunbound = (isunbound==1)
    #
    # beta for wind
    #beta=((gam-1)*ug)*divideavoidinf(bsq*0.5)
    beta=((gam-1)*ug)/(1E-30 + bsq*0.5)
    betatot=((gam-1)*ug+(4.0/3.0-1)*urad)/(1E-30 + bsq*0.5)
    if maxbeta is None:
        insidebeta = 1
    else:
        insidebeta = (betatot<maxbeta) # using tot now
    #
    #
    superintegrand=(integrand*insideinflowonly*insideoutflowonly*insidehor*insideminbsqorho*insidemaxbsqorho*insidemumin*insidemumax*insideunbound*insidebeta*which)
    #
    if doavgn3==1:
        # this will be function of r
        integral=(superintegrand).sum(axis=2).sum(axis=1)*_dx2*_dx3
        # below factor accounts for whether inputted phi slice or averaged or full nz in size, so that integral is correctly summed to full amount
        integral=integral*(nz/len(superintegrand[0,0,:]))
        # below factor accounts for HARM vs. normal full 2\pi size of \phi-direction
        integral=scaletofullwedge(integral)
    else:
        # this will be function of r,\phi
        integral=(superintegrand).sum(axis=1)*_dx2
    #
    return(integral)

# need integrate when nz=1 because just avg2d data.  Means averaged-out, so sum is recovered by multiplying by number of cells in \phi
def intangle_foravg2d(qty,hoverr=None,thetamid=np.pi/2,minbsqorho=None,maxbsqorho=None,inflowonly=None,outflowonly=None,mumax=None,mumin=None,maxbeta=None,unboundonly=None,which=1,doabs=0):
    #
    if doabs==1:
        qtynew=np.fabs(qty)
    else:
        qtynew=qty
    #
    # translate from avg to normal quantity
    global bsq,rho,ug,uu,ud,rhoclean,ugclean,rhounclean,ugunclean,enth,unb,isunbound,tiny,entropy,Erf,urad,uradu,uradd
    bsq=avg_bsq
    rho=avg_rho
    KAPPAUSER=avg_KAPPAUSER
    KAPPAESUSER=avg_KAPPAESUSER
    tauradintegrated=avg_tauradintegrated
    tauradeffintegrated=avg_tauradeffintegrated
    ug=avg_ug
    urad=avg_urad
    uu=avg_uu
    ud=avg_ud
    # averages already removed jet densities
    rhoclean=rho
    rhounclean=rho
    ugclean=ug
    ugunclean=ug
    enth=1+ugclean*gam/rhoclean
    entropy=(gam-1.0)*ugclean/rho**(gam)
    unb=enth*ud[0]
    # unbound here means *thermally* rather than kinetically (-u_t>1) or fully thermo-magnetically (\mu>1) unbound.
    isunbound=(-unb>1.0)
    tiny=np.finfo(rho.dtype).tiny
    #
    result=intangle(qtynew,hoverr=hoverr,thetamid=thetamid,minbsqorho=minbsqorho,maxbsqorho=maxbsqorho,inflowonly=inflowonly,outflowonly=outflowonly,mumax=mumax,mumin=mumin,maxbeta=maxbeta,unboundonly=unboundonly,which=which)
    return(result)

def mdotcalc(ivalue=None,**kwargs):
    """
    Computes the absolute flux through the sphere i = ivalue
    """
    #1D function of theta only:
    md = intangle( -gdet*rho*uu[1], **kwargs)
    if ivalue==None:
        return(md)
    else:
        return(md[ivalue])

# to control whether really take abs for alphamag and alphareynolds stuff
def jabs(var):
    #return(np.abs(var))
    return(var)

def amax(arg1,arg2):
    arr1 = np.array(arg1)
    arr2 = np.array(arg2)
    ret=np.zeros_like(arr1)
    ret[arr1>=arr2]=arr1[arr1>=arr2]
    ret[arr2>arr1]=arr2[arr2>arr1]
    return(ret)

def elinfcalc(a):
    # assume disk rotation sense is always positive, but a can be + or -
    risco=Risco(a)
    risco2=Risco(-a)
    #
    #print( "risco=%g" % (risco) )
    #
    if a<0.9999999:
        einf=(1.0-2.0/risco+a/(risco)**(3.0/2.0))/(1.0-3.0/risco+2.0*a/(risco)**(3.0/2.0))**(1.0/2.0)
        #print( "einf=%g" % (einf) )
        linf=(np.sqrt(risco)*(risco**2.0-2.0*a*np.sqrt(risco)+a**2))/(risco*(risco**2.0-3.0*risco+2.0*a*np.sqrt(risco))**(1.0/2.0))
        #print( "linf=%g" % (linf) )
    else:
        if risco<2.0:
            # einf
            einf=0.57735
            # linf
            linf=0.0
        else:
            # einf
            einf=0.946729
            # linf
            linf=4.2339
        #
    #
    return einf,linf

# http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-a-an-extrapolated-result-beyond-the-input-ran
#from scipy.interpolate import interp1d
#from scipy import arange, array, exp
# use of extrap1d must be with np.array type (not just single value)
def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y
    #
    def pointwise(x):
        if x < xs[0]:
            ivalue=ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            if np.sign(ivalue)!=np.sign(ys[0]):
                ivalue=0.0 # don't allow sign to change
            return(ivalue)
        elif x > xs[-1]:
            ivalue=ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            if np.sign(ivalue)!=np.sign(ys[-1]):
                ivalue=0.0 # don't allow sign to change
            return(ivalue)
        else:
            return interpolator(x)
    #
    def ufunclike(xs):
        return sp.array(list(map(pointwise, sp.array(xs))))
    #
    return ufunclike

def Risco(a):
    Z1 = 1 + (1. - a**2)**(1./3.) * ((1. + a)**(1./3.) + (1. - a)**(1./3.))
    Z2 = (3*a**2 + Z1**2)**(1./2.)
    risco = 3 + Z2 - np.sign(a)* ( (3 - Z1)*(3 + Z1 + 2*Z2) )**0.5
    return(risco)

def odot(a,b):
    """ Outer product of two vectors a^mu b_nu"""
    #the shape of the product is (4,4,nx,ny,max(a.nz,b.nz))
    outer_product = np.zeros(np.concatenate((np.array((4,4)),amax(a[0].shape,b[0].shape))),dtype=np.float32,order='F')
    for mu in np.arange(4):
        for nu in np.arange(4):
            outer_product[mu,nu] = a[mu]*b[nu]
    return(outer_product)

# new vectorized
# http://www.johnny-lin.com/cdat_tips/tips_array/boolean.html
def fix_hp(th,ph):
    #
    M_PI = np.pi
    # keep \phi between 0 and 2\pi.  Can always do that full rotation.
    # assume never more out of phase that 1 full rotation
    ph[ph<0.0]=ph[ph<0.0]             + 2.0*M_PI
    ph[ph>=2.0*M_PI]=ph[ph>=2.0*M_PI] - 2.0*M_PI
    #
    # keep \theta between 0 and \pi and \phi between 0 and 2\pi
    # but need to be at same physical SPC location, not arbitrary rotation

    # have to change ph first since depends upon th
    ph[np.logical_and(np.logical_or(th<0.0,th>=M_PI),ph<=M_PI)]=ph[np.logical_and(np.logical_or(th<0.0,th>=M_PI),ph<=M_PI)] + M_PI
    ph[np.logical_and(np.logical_or(th<0.0,th>=M_PI),ph> M_PI)]=ph[np.logical_and(np.logical_or(th<0.0,th>=M_PI),ph> M_PI)] - M_PI

    # now can change th that doesn't depend upon ph
    th[th<0.0]=th[th<0.0]*(-1.0)
    th[th>=M_PI]=M_PI-th[th>=M_PI]

    #
    return(th,ph)

###################################
#
# Not sure what these are for, but other functions call them
#
###################################
def KAPPA_ES_CODE(rhocode,Tcode):
    global GGG,CCCTRUE,MSUNCM,MPERSUN,LBAR,TBAR,VBAR,RHOBAR,MBAR,ENBAR,UBAR,TEMPBAR,ARAD_CODE_DEF,XFACT,ZATOM,AATOM,MUE,MUI,OPACITYBAR,MASSCM,KORAL2HARMRHO1,Leddcode,Mdoteddcode,rhoeddcode,ueddcode,beddcode
    y=(0.2*(1.0+XFACT)/OPACITYBAR)
    return(y)
def KAPPA_FF_CODE(rhocode,Tcode):
    global GGG,CCCTRUE,MSUNCM,MPERSUN,LBAR,TBAR,VBAR,RHOBAR,MBAR,ENBAR,UBAR,TEMPBAR,ARAD_CODE_DEF,XFACT,ZATOM,AATOM,MUE,MUI,OPACITYBAR,MASSCM,KORAL2HARMRHO1,Leddcode,Mdoteddcode,rhoeddcode,ueddcode,beddcode
    y=(1.0E23*ZATOM*ZATOM/(MUE*MUI)*(rhocode*RHOBAR)*pow(Tcode*TEMPBAR,-7.0/2.0)/OPACITYBAR)
    return(y)

###################################
#
# Functions for computing horizon quantities
#
###################################
def horcalc(hortype=1,which1=1,which2=1,denfactor=None):
    """
    Compute root mean square deviation of disk body from equatorial plane
    """
    if denfactor is None:
        denfactor=rholab
    #
    # determine when have to revert to which2
    which=which1
    testit=which1.sum(axis=1)
    for i in np.arange(0,nx):
        for k in np.arange(0,nz):
            if testit[i,k]==0:
                which[i,:,k]=which2[i,:,k]
    #
    tiny=np.finfo(rho.dtype).tiny
    #
    thetamid3d=np.zeros((nx,ny,nz),dtype=h.dtype)
    # sum over \theta
    if hortype==1:
        up=(gdet*denfactor*(h-np.pi/2)*which).sum(axis=1)
        dn=(gdet*denfactor*which).sum(axis=1)
        # so average over \theta
        thetamid2d=up/(dn+tiny)+np.pi/2.0
        #print("thetamid2d")
        #god=thetamid2d[iofr(100),:]
        #print(god)
        for j in np.arange(0,ny):
            thetamid3d[:,j,:] = thetamid2d  # thetamid2d depends upon r and \phi
    else:
        thetamid3d=0.5*np.pi+thetamid3d
    #
    up=(gdet*denfactor*(h-thetamid3d)**2*which).sum(axis=1)
    #up=(gdet*denfactor*(h-1.57)**2*which).sum(axis=1)
    dn=(gdet*denfactor*which).sum(axis=1)
    hoverr2d= (up/(dn+tiny))**0.5
    hoverr3d=np.empty((nx,ny,nz),dtype=h.dtype)
    for j in np.arange(0,ny):
        hoverr3d[:,j,:] = hoverr2d
    #
    #
    return((hoverr3d,thetamid3d))

def areahor():
    # first load grid file
    grid3d("gdump.bin",use2d=True)
    #
    rhor=1+(1-a**2)**0.5
    print(("rhor=%g" % (rhor)))
    #
    ihor = np.floor(iofr(rhor)+0.5)
    idxdxp11=dxdxp[2][2]/(dxdxp[2][2]*dxdxp[1][1]-dxdxp[2][1]*dxdxp[1][2])
    unit=idxdxp11 # dx1/dr
    #
    area = np.sum(unit[ihor,:,:]*gdet[ihor,:,:]*_dx2*_dx3)
    print(("area=%g" % (area)))
    areashould = 4.0*np.pi/3.0*(3.0*rhor*rhor+a*a)
    print(("areashould=%g" % (areashould)))
    areaup = np.sum(unit[ihor,0:ny//2,:]*gdet[ihor,0:ny//2,:]*_dx2*_dx3)
    print(("areaup=%g" % (areaup)))
    areaupshould = areashould/2.0
    print(("areaupshould=%g" % (areaupshould)))
    scale=scaletofullwedge(1.0)
    print(("scale=%g" % (scale)))
    #
    area = np.sum(unit[ihor,:,:]*gdet[ihor+1,:,:]*_dx2*_dx3)
    print(("area=%g" % (area)))
    areaup = np.sum(unit[ihor,0:ny//2,:]*gdet[ihor+1,0:ny//2,:]*_dx2*_dx3)
    print(("areaup=%g" % (areaup)))

def horfluxcalc(ivalue=None,jvalue=None,takeabs=1,takecumsum=0,takeextreme=0,minbsqorho=10,inflowonly=None,outflowonly=None,whichcondition=True,uphalf=None):
    """
    Computes the absolute flux through the sphere i = ivalue
    """
    global gdetB, _dx2, _dx3
    #1D function of theta only:
    if takeabs==1:
        tosum=np.abs(gdetB[1]*(bsq/rho>=minbsqorho))
    else:
        tosum=gdetB[1]*(bsq/rho>=minbsqorho)
    #
    if inflowonly==None:
        tosum=tosum
    else:
        tosum=tosum*(uu[1]<0)
    #
    if outflowonly==None:
        tosum=tosum
    else:
        tosum=tosum*(uu[1]>0)
    #
    if uphalf==None:
        tosum=tosum
    else:
        tosum=tosum*(h<=np.pi*0.5)
    #
    #
    tosum=tosum*(whichcondition==True)
    #
    dfabs = (tosum).sum(2)*_dx2*_dx3
    #
    #account for the wedge
    dfabs=scaletofullwedge(dfabs)
    #
    if takecumsum==0:
        fabs = dfabs.sum(axis=1)
        if ivalue == None:
            return(fabs)
        else:
            return(fabs[ivalue])
        #
    else:
        fabs = dfabs.cumsum(axis=1)
        if ivalue == None and jvalue == None:
            if takeextreme==1:
                bigj=np.zeros(nx,dtype=int)
                finalresult=np.zeros(nx,dtype=float)
                for ii in np.arange(0,nx):
                    condition=(np.fabs(fabs[ii,:])==np.max(np.fabs(fabs[ii,:])))
                    condition=condition*(np.fabs(fabs[ii,:])>1E-15)
                    tempresultO=np.where(condition==1)[0]
                    tempresult=tempresultO.astype(np.integer)
                    #
                    # assume all values are zero if here, so just choose one of the zero values
                    if len(tempresult)==0:
                        tempresult=ny//2
                    #
                    if type(tempresult) is not int:
                        tempresult=tempresult[0]
                    #
                    #print("tempresult")
                    #print(tempresult)
                    #print("ii=%d tempresult=%d fabs=%g" % (ii,tempresult,fabs[ii,tempresult]) )
                    bigj[ii]=tempresult
                    finalresult[ii]=fabs[ii,bigj[ii]]
                #
                #print("shape of bigj")
                #print(bigj.shape)
                #
                #print("sizefinalresult")
                #print(finalresult.shape)
                #
                return(finalresult)
            else:
                return(fabs)
        elif ivalue is not None:
            if takeextreme==1:
                bigj=np.where(np.fabs(fabs[ivalue,:])==np.max(np.fabs(fabs[ivalue,:])))[0]
                #print("bigj=%d" % (bigj) )
                return(fabs[ivalue,bigj])
            else:
                return(fabs[ivalue,:])
            #
        elif jvalue is not None:
            return(fabs[:,jvalue])
        else:
            return(fabs[ivalue,jvalue])

# trans requires input of full 3d Vmetric (not just use2d gdump version)
def set_transV2Vmetric(Vmetric=None,b0=0.0):
    #
    # r,h,ph are Vmetric[] for both trans matrices.
    # these are full 3D r,h,ph
    r=Vmetric[1]
    h=Vmetric[2]
    ph=Vmetric[3]
    #
    #
    #b0=THETAROT
    #
    transV2Vmetric=np.zeros((4,4,nx,ny,nz),dtype=r.dtype)
    # r,h,ph are Vmetric[]
    # transV2Vmetric^\mu[Vmetric]_\nu[V] u^\nu[V] : So first index is Vmetric-type.  Second index is V-type.  Operates on contravariant V-type.
    transV2Vmetric[0,0]=1. + gv3[0,0]*0.0
    transV2Vmetric[0,1]=0. + gv3[0,0]*0.0
    transV2Vmetric[0,2]=0. + gv3[0,0]*0.0
    transV2Vmetric[0,3]=0. + gv3[0,0]*0.0
    transV2Vmetric[1,0]=0. + gv3[0,0]*0.0
    transV2Vmetric[1,1]=1. + gv3[0,0]*0.0
    transV2Vmetric[1,2]=0. + gv3[0,0]*0.0
    transV2Vmetric[1,3]=0. + gv3[0,0]*0.0
    transV2Vmetric[2,0]=0. + gv3[0,0]*0.0
    transV2Vmetric[2,1]=0. + gv3[0,0]*0.0
    transV2Vmetric[2,2]=pow (pow (cos (h)*sin (b0) - 1.*cos (b0)*cos (ph)*sin (h),2.) + pow (sin (h),2.)*pow (sin (ph),2.),-0.5)*(-1.*cos (h)*cos (ph)*sin (b0) + cos (b0)*sin (h))
    transV2Vmetric[2,3]=pow (pow (cos (h)*sin (b0) - 1.*cos (b0)*cos (ph)*sin (h),2.) + pow (sin (h),2.)*pow (sin (ph),2.),-0.5)*sin (b0)*sin (h)*sin (ph)
    transV2Vmetric[3,0]=0. + gv3[0,0]*0.0
    transV2Vmetric[3,1]=0. + gv3[0,0]*0.0
    transV2Vmetric[3,2]=-1.*pow (pow (cos (h)*sin (b0) - 1.*cos (b0)*cos (ph)*sin (h),2.) + pow (sin (h),2.)*pow (sin (ph),2.),-1.)*sin (b0)*sin (ph)
    transV2Vmetric[3,3]=pow (pow (cos (h)*sin (b0) - 1.*cos (b0)*cos (ph)*sin (h),2.) + pow (sin (h),2.)*pow (sin (ph),2.),-1.)*sin (h)*(-1.*cos (h)*cos (ph)*sin (b0) + cos (b0)*sin (h))

    if 0==1: # don't need, so don't waste memory
        transVmetric2V=np.zeros((4,4,nx,ny,nz),dtype=r.dtype)
        # r,h,ph are Vmetric[]
        # transVmetric2V^\mu[V]_\nu[Vmetric] u^\nu[Vmetric] : So first index is V-type.  Second index is Vmetric-type.  Operates on contravariant Vmetric-type.
        transVmetric2V[0,0]=1. + gv3[0,0]*0.0
        transVmetric2V[0,1]=0. + gv3[0,0]*0.0
        transVmetric2V[0,2]=0. + gv3[0,0]*0.0
        transVmetric2V[0,3]=0. + gv3[0,0]*0.0
        transVmetric2V[1,0]=0. + gv3[0,0]*0.0
        transVmetric2V[1,1]=1. + gv3[0,0]*0.0
        transVmetric2V[1,2]=0. + gv3[0,0]*0.0
        transVmetric2V[1,3]=0. + gv3[0,0]*0.0
        transVmetric2V[2,0]=0. + gv3[0,0]*0.0
        transVmetric2V[2,1]=0. + gv3[0,0]*0.0
        transVmetric2V[2,2]=pow (pow (cos (h)*cos (ph)*sin (b0) - 1.*cos (b0)*sin (h),2.) + pow (sin (b0),2.)*pow (sin (ph),2.),-1.)*pow (pow (cos (h)*sin (b0) - 1.*cos (b0)*cos (ph)*sin (h),2.) + pow (sin (h),2.)*pow (sin (ph),2.),0.5)*(-1.*cos (h)*cos (ph)*sin (b0) + cos (b0)*sin (h))
        transVmetric2V[2,3]=-1.*sin (b0)*sin (ph)
        transVmetric2V[3,0]=0. + gv3[0,0]*0.0
        transVmetric2V[3,1]=0. + gv3[0,0]*0.0
        transVmetric2V[3,2]=csc (h)*pow (pow (cos (h)*cos (ph)*sin (b0) - 1.*cos (b0)*sin (h),2.) + pow (sin (b0),2.)*pow (sin (ph),2.),-1.)*pow (pow (cos (h)*sin (b0) - 1.*cos (b0)*cos (ph)*sin (h),2.) + pow (sin (h),2.)*pow (sin (ph),2.),0.5)*sin (b0)*sin (ph)
        transVmetric2V[3,3]=cos (b0) - 1.*cos (ph)*cot (h)*sin (b0)
        #
    #return(transV2Vmetric,transVmetric2V)
    return(transV2Vmetric)

def rotate_VtoVmetric(V,Vmetric,b0=0.0):
    # V is tnew,rnew,hnew,phnew (i.e. actual r,h,ph in original gdump)
    tnew=V[0]
    rnew=V[1]
    hnew=V[2]
    phnew=V[3]

    #FTYPE told,rold,hold,phold; // what is inside metric functions like set_gcov()
    told=tnew
    rold=rnew

    #b0=THETAROT

    cb0=cos(b0)
    sb0=sin(b0)
    sh=sin(hnew)
    shsq=sh**2.0
    ch=cos(hnew)
    sph=sin(phnew)
    sphsq=sph**2.0
    cph=cos(phnew)

    arg1=sqrt((ch*sb0 + cb0*cph*sh)**2.0 + shsq*sphsq)

    hold=arctanmath (cb0*ch - cph*sb0*sh , arg1)

    phold=arctanmath (ch*sb0 + cb0*cph*sh,sh*sph)

    # hold=arctanmath (cos (b0)*cos (hnew) - 1.*cos (phnew)*sin (b0)*sin (hnew),pow (pow (cos (hnew)*sin (b0) + cos (b0)*cos (phnew)*sin (hnew),2) + pow (sin (hnew),2)*pow (sin (phnew),2),0.5))
    # phold=arctanmath (cos (hnew)*sin (b0) + cos (b0)*cos (phnew)*sin (hnew),sin (hnew)*sin (phnew))

    M_PI=np.pi

    # constrain th,ph to Cartesian-correct locations within limited \theta,\phi span.
    (hold,phold)=fix_hp(hold,phold)

    ###################
    Vmetric=np.copy(V)
    #
    Vmetric[0]=told
    Vmetric[1]=rold
    Vmetric[2]=hold
    Vmetric[3]=phold
    #
    return(Vmetric)

###################################
#
# Functions for finding the grid coordinates of r, theta, phi
#
###################################
def iofr(rval):
    return(iofrpole(rval))

def iofrpole(rval):
    if rval<=Rin or rval<=r[0,0,0]:
        return(0)
    elif rval>=Rout or rval>=r[-1,0,0]:
        return(ti[-1,0,0])
    else:
        res = interp1d(r[:,0,0], ti[:,0,0], kind='linear')
        return(np.floor(res(rval)+0.5))

def jofh(hval,i):
    return(np.floor(jofhfloatsimple(hval,i)+0.5))

def jofhfloatsimple(hval,i):
    res = interp1d(h[i,:,0], tj[i,:,0], kind='linear')
    # return float result
    return(res(hval))

# general 1D interpolations
def iofrfloat(pickti,pickr,rval):
    #
    if nx==1:
        return(0.0*rval)
    #
    #rval[rval<=Rin or rval<pickr[0]] = pickr[0]
    #rval[rval>=Rout or rval>pickr[-1]] = pickr[-1]
    rval[rval<=Rin] = Rin
    rval[rval>=Rout] = Rout
    #
    res = interp1d(pickr[:], pickti[:], kind='linear')
    resextrap = extrap1d(res)
    # return  of float result
    ival=resextrap(rval)
    ival[ival<0.0]=1E-10
    ival[ival>=nx-1.0]=(1.0-1E-10)*(nx-1.0)
    return(ival)

def jofhfloat(picktj,pickh,hval):
    #
    if ny==1:
        return(0.0*hval)
    #
    #print("pickh picktj")
    #print(pickh)
    #print(picktj)
    res = interp1d(pickh[:], picktj[:], kind='linear')
    resextrap = extrap1d(res)
    # return  of float result
    jval=resextrap(hval)
    # ensure within limits (given how setup theta,phi, should always be good, but might be at very edge (e.g. tj=128) due to extrapolation.
    jval[np.logical_and(jval<0.0,hval>=0.0)]=1E-10
    jval[np.logical_and(jval>=ny-1,hval<np.pi)]=(1.0-1E-10)*(ny-1.0)
    return(jval)

def kofphfloat(picktk,pickph,phval):
    #
    if nz==1:
        return(0.0*phval)
    #
    phval[phval<0.0] = phval[phval<0.0] + 2.0*np.pi
    phval[phval>2.0*np.pi] = phval[phval>2.0*np.pi] - 2.0*np.pi
    #
    res = interp1d(pickph[:], picktk[:], kind='linear') #,bounds_error=False,fill_value=-1)
    resextrap = extrap1d(res)
    kval=resextrap(phval) #  of floats return
    kval[np.logical_and(kval<0.0,phval>=0.0)]=1E-10
    kval[np.logical_and(kval>=nz-1.0,phval<2.0*np.pi)]=(1.0-1E-10)*(nz-1.0)
    return(kval)

###################################
#
# Functions for loading the simulation grid
#
###################################

def grid3d(dumpname,use2d=False,doface=False,usethetarot0=False): #read grid dump file: header and body
    #
    #
    #
    if usethetarot0==True:
        filename="dumps/gdump.THETAROT0.bin"
        dumpname="gdump.THETAROT0.bin" # override input
    else:
        filename="dumps/gdump.bin"
        # just use input dumpname
    #
    if os.path.isfile(filename):
        # only need header of true gdump.bin to get true THETAROT
        rfdheaderonly(filename)
    else:
        # if no gdump, use last fieldline file that is assumed to be consistent with gdump that didn't exist.
        # allows non-creation of gdump if restarting with tilt from non-tilt run.  So then enver have to have gdump.bin with THETAROT tilt.
        rfdheaderlastfile()
    #
    # for rfd() to use to see if different nz size
    global nzgdumptrue
    nzgdumptrue=nz
    #
    global nxgdump,nygdump,nzgdump,THETAROTgdump
    nxgdump=nx
    nygdump=ny
    nzgdump=nz
    THETAROTgdump=THETAROT
    #
    # determine if need to read THETAROT0 or normal general gdump
    #../../dumps/gdump.THETAROT0.bin
    #
    #
    # for THETAROT!=0, assume gdump.THETAROT0.bin exists corresponding to the non-rotated THETAROT=0 version.
    # Using this vastly speeds-up read-in and doesn't use excessive (too much!) memory required for full 3D interpolation of (a minimum) gv3 while reading in all other things because binary and using np.fromfile().
    if np.fabs(THETAROT-0.0)>1E-13 and use2d==True:
        realdumpname="gdump.THETAROT0.bin"
        # NOTEMARK: older sasha runs that weren't tilted had 32-64 phi-zones, whereas new has 128 phi-zones.  But once read-in, only use one-phi zone for this file and that's all that's needed.  The actual nz and full 3D things (ti,tj,tk,x1,x2,x3,r,h,ph) will be overwritten or corrected when rfd() is called
        # Note:  So for tilted runs, *only* need non-tilted gdump and that only has to be axisymmetric for BH solutions!
    else:
        realdumpname=dumpname
    #
    print(( "realdumpname=%s" % (realdumpname) )) ; sys.stdout.flush()
    #
    # load axisymmetric metric-grid data
    # this sets THETAROT=0 if THETAROT true is non-zero.  rfd() is responsible for setting THETAROT for each fieldline file so data inputted is transformed/interpolated correctly.
    grid3d_load(dumpname=realdumpname,use2d=use2d,doface=doface,loadsimple=False)
    #
    # get other things
    gridcellverts()
    #
    gc.collect() #try to release unneeded memory
    print( "Done grid3d!" ) ; sys.stdout.flush()

def grid3d_load(dumpname=None,use2d=False,doface=False,loadsimple=False): #read grid dump file: header and body
    #The internal cell indices along the three axes: (ti, tj, tk)
    #The internal uniform coordinates, (x1, x2, x3), are mapped into the physical
    #non-uniform coordinates, (r, h, ph), which correspond to radius (r), polar angle (theta), and toroidal angle (phi).
    #There are more variables, e.g., dxdxp, which is the Jacobian of (x1,x2,x3)->(r,h,ph) transformation, that I can
    #go over, if needed.
    global nx,ny,nz,lnz,_startx1,_startx2,_startx3,_dx1,_dx2,_dx3,gam,a,Rin,Rout
    global nzgdump
    global ti,tj,tk,x1,x2,x3,r,h,ph,gn3,gv3,dxdxp,gdet
    # global ck,conn
    print(( "Reading grid from " + "dumps/" + dumpname + " ..." )) ; sys.stdout.flush()
    gin = open( "dumps/" + dumpname, "rb" )
    #
    #First line of grid dump file is a text line that contains general grid information:
    header = gin.readline().split()
    #dimensions of the grid
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    #
    # for rfd() to use to see if different nz size
    nzgdump=nz
    #
    #grid internal coordinates starting point
    _startx1=myfloatalt(float(header[4]))
    _startx2=myfloatalt(float(header[5]))
    _startx3=myfloatalt(float(header[6]))
    #cell size in internal coordintes
    _dx1=myfloatalt(float(header[7]))
    _dx2=myfloatalt(float(header[8]))
    _dx3=myfloatalt(float(header[9]))
    #other information:
    #polytropic index
    gam=myfloatalt(float(header[11]))
    #black hole spin
    a=myfloatalt(float(header[12]))
    rhor = 1+(1-a**2)**0.5
    #Spherical polar radius of the innermost radial cell
    Rin=myfloatalt(float(header[14]))
    #Spherical polar radius of the outermost radial cell
    Rout=myfloatalt(float(header[15]))
    #read grid dump per-cell data
    #
    if use2d:
        lnz = 1
    else:
        lnz = nz
    #
    print( "Done reading grid header" ) ; sys.stdout.flush()
    #
    ncols = 126
    if dumpname.endswith(".bin"):
        print(( "Start reading grid as binary with lnz=%d" % (lnz) )) ; sys.stdout.flush()
        body = np.fromfile(gin,dtype=np.float64,count=ncols*nx*ny*lnz)
        gd = body.view().reshape((-1,nx,ny,lnz),order='F')
        gin.close()
        print(( "Done reading grid as binary with lnz=%d" % (lnz) )) ; sys.stdout.flush()
    else:
        print(( "Start reading grid as text with lnz=%d" % (lnz) )) ; sys.stdout.flush()
        gin.close()
        gd = np.loadtxt( "dumps/" + dumpname,
                      dtype=np.float64,
                      skiprows=1,
                      unpack = True ).view().reshape((126,nx,ny,lnz), order='F')
        print(( "End reading grid as text with lnz=%d" % (lnz) )) ; sys.stdout.flush()
    gd=myfloat(gd)
    gc.collect()
    #
    print( "Done reading grid" ) ; sys.stdout.flush()
    #
    # always load ti,tj,tk,x1,x2,x3,r,h,ph
    # SUPERNOTEMARK: for use2d, note that tk depends upon \phi unlike all other things for a Kerr metric in standard coordinates
    ti,tj,tk,x1,x2,x3,r,h,ph = gd[0:9,:,:,:].view()
    #covariant metric components, g_{\mu\nu}
    gv3 = gd[89:105].view().reshape((4,4,nx,ny,lnz), order='F').transpose(1,0,2,3,4)
    #
    # only load if not loading simple version required for THETAROT transformation of 3D grid
    if loadsimple==0:
        #get the right order of indices by reversing the order of indices i,j(,k)
        #conn=gd[9:73].view().reshape((4,4,4,nx,ny,lnz), order='F').transpose(2,1,0,3,4,5)
        #contravariant metric components, g^{\mu\nu}
        gn3 = gd[73:89].view().reshape((4,4,nx,ny,lnz), order='F').transpose(1,0,2,3,4)
        #metric determinant
        gdet = gd[105]
        # don't need ck, so don't load
        #ck = gd[106:110].view().reshape((4,nx,ny,lnz), order='F')
        #grid mapping Jacobian
        dxdxp = gd[110:126].view().reshape((4,4,nx,ny,lnz), order='F').transpose(1,0,2,3,4)
def gridcellverts():
    ##################################
    #CELL VERTICES:
    global tif,tjf,tkf,rf,hf,phf
    #RADIAL:
    #add an extra dimension to rf container since one more faces than centers
    rf = np.zeros((r.shape[0]+1,r.shape[1]+1,r.shape[2]+1))
    #operate on log(r): average becomes geometric mean, etc
    rf[1:nx,0:ny,0:lnz] = (r[1:nx]*r[0:nx-1])**0.5 #- 0.125*(dxdxp[1,1,1:nx]/r[1:nx]-dxdxp[1,1,0:nx-1]/r[0:nx-1])*_dx1
    #extend in theta
    rf[1:nx,ny,0:lnz] = rf[1:nx,ny-1,0:lnz]
    #extend in phi
    rf[1:nx,:,lnz]   = rf[1:nx,:,lnz-1]
    #extend in r
    rf[0] = 0*rf[0] + Rin
    rf[nx] = 0*rf[nx] + Rout
    #ANGULAR:
    hf = np.zeros((h.shape[0]+1,h.shape[1]+1,h.shape[2]+1))
    hf[0:nx,1:ny,0:lnz] = 0.5*(h[:,1:ny]+h[:,0:ny-1]) #- 0.125*(dxdxp[2,2,:,1:ny]-dxdxp[2,2,:,0:ny-1])*_dx2
    hf[1:nx-1,1:ny,0:lnz] = 0.5*(hf[0:nx-2,1:ny,0:lnz]+hf[1:nx-1,1:ny,0:lnz])
    #populate ghost cells in r
    hf[nx,1:ny,0:lnz] = hf[nx-1,1:ny,0:lnz]
    #populate ghost cells in phi
    hf[:,1:ny,lnz] = hf[:,1:ny,lnz-1]
    #populate ghost cells in theta (note: no need for this since already initialized everything to zero)
    hf[:,0] = 0*hf[:,0] + 0
    hf[:,ny] = 0*hf[:,ny] + np.pi
    #TOROIDAL:
    phf = np.zeros((ph.shape[0]+1,ph.shape[1]+1,ph.shape[2]+1))
    phf[0:nx,0:ny,0:lnz] = ph[0:nx,0:ny,0:lnz] - dxdxp[3,3,0,0,0]*0.5*_dx3
    #extend in phi
    phf[0:nx,0:ny,lnz]   = ph[0:nx,0:ny,lnz-1] + dxdxp[3,3,0,0,0]*0.5*_dx3
    #extend in r
    phf[nx,0:ny,:]   =   phf[nx-1,0:ny,:]
    #extend in theta
    phf[:,ny,:]   =   phf[:,ny-1,:]
    #indices
    #tif=np.zeros(ti.shape[0]+1,ti.shape[1]+1,ti.shape[2]+1)
    #tjf=np.zeros(tj.shape[0]+1,tj.shape[1]+1,tj.shape[2]+1)
    #tkf=np.zeros(tk.shape[0]+1,tk.shape[1]+1,tk.shape[2]+1)
    tif=np.arange(0,(nx+1)*(ny+1)*(lnz+1)).reshape((nx+1,ny+1,lnz+1),order='F')
    tjf=np.arange(0,(nx+1)*(ny+1)*(lnz+1), dtype = 'float').reshape((nx+1,ny+1,lnz+1),order='F')
    tkf=np.arange(0,(nx+1)*(ny+1)*(lnz+1),dtype = 'float').reshape((nx+1,ny+1,lnz+1),order='F')
    tif %= (nx+1)
    tjf /= (nx+1)
    tjf %= (ny+1)
    tkf /= (ny+1)*(lnz+1)

###################################
#
# Functions for loading fieldline files
#
###################################
def rfd(fieldlinefilename,**kwargs):
    # MEMMARK: 5+4+1+4+4+4+1+16=39 full 3D vars
    #read information from "fieldline" file:
    #Densities: rho, u,
    #Velocity components: u1, u2, u3,
    #Cell-centered magnetic field components: B1, B2, B3,
    #Face-centered magnetic field components multiplied by metric determinant: gdetB1, gdetB2, gdetB3
    global rho,ug,uu,B,gdetB,Erf,urad,uradu, numcolumns
    #
    #read image
    #
    # get starting time so can compute time differences
    start_time=datetime.now()
    #
    #### read header
    fname= "dumps/" + fieldlinefilename
    fin = open(fname, "rb" )
    rfdheader(fin=fin)
    fin.close()
    #
    #
    # check if last-read gdump has same nx,ny,nz,THETAROT as fieldline file
    if nx==nxgdump and ny==nygdump and nz==nzgdump and THETAROT==THETAROTgdump:
        print("fieldline file has same nx,ny,nz,THETAROT as gdump file") ; sys.stdout.flush()
    else:
        print("fieldline file has different nx,ny,nz,THETAROT as gdump file, so reading correct gdump file") ; sys.stdout.flush()
        if THETAROT==0.0:
            grid3d("gdump.bin",use2d=use2dglobal,usethetarot0=True)
        else:
            grid3d("gdump.bin",use2d=use2dglobal,usethetarot0=False)
        # re-read fieldline file header so (e.g.) time is correct
    #
    #
    # generally re-read header in case grid3d() was loaded
    fin = open(fname, "rb" )
    rfdheader(fin=fin)
    #read grid dump per-cell data
    #
    if(0):
        # new way
        # http://www.mail-archive.com/numpy-discussion@scipy.org/msg07631.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
        # still doesn't quite work once beyond this function!
        numcolumns=int(header[29])
        from numpy import memmap
        d = np.memmap(fin, dtype='float32', mode='c', shape=(numcolumns,nx,ny,nz),order='F')
    #
    else:
        # old way:
        body = np.fromfile(fin,dtype=np.float32,count=-1)
        #body = np.load(fin,dtype=np.float32,count=-1)
        d=body.view().reshape((-1,nx,ny,nz),order='F')
        del(body)
    #
    fin.close()
    #
    # fix-up pole : Should have been done during code -- still not perfectly stable near pole
    whichpoledeath=2
    #
    if whichpoledeath==1:
        if np.fabs(THETAROT>0.0):
            print("Fixing primitives") ; sys.stdout.flush()
            for primi in np.arange(0,11):
                d[primi,:,0,:]=d[primi,:,3,:]
                d[primi,:,1,:]=d[primi,:,3,:]
                d[primi,:,2,:]=d[primi,:,3,:]
                d[primi,:,ny-1,:]=d[primi,:,ny-4,:]
                d[primi,:,ny-2,:]=d[primi,:,ny-4,:]
                d[primi,:,ny-3,:]=d[primi,:,ny-4,:]
    #
    #
    #rho, u, -hu_t, -T^t_t/U0, u^t, v1,v2,v3,B1,B2,B3
    #matter density in the fluid frame
    rho=np.zeros((1,nx,ny,nz),dtype='float32',order='F')
    rho=d[0,:,:,:]
    #matter internal energy in the fluid frame
    ug=np.zeros((1,nx,ny,nz),dtype='float32',order='F')
    ug=d[1,:,:,:]
    #d[4] is the time component of 4-velocity, u^t
    #d[5:8] are 3-velocities, v^i
    uu=np.zeros((4,nx,ny,nz),dtype='float32',order='F')
    uu=d[4:8,:,:,:]  #again, note uu[i] are 3-velocities (as read from the fieldline file)
    #multiply by u^t to get 4-velocities: u^i = u^t v^i
    uu[1:4]=uu[1:4] * uu[0]
    #
    if whichpoledeath==2:
        if np.fabs(THETAROT>0.0):
            print("Fixing primitives") ; sys.stdout.flush()
            # form relative 4-velocity
            eta=np.copy(uu)*0 # setup memory
            alpha=1.0/np.sqrt(-gn3[0,0])
            eta[0]=-alpha
            beta=np.copy(uu)*0
            beta[:]=alpha**2*gn3[0,:] # only spatial part will be used
            etaup=np.copy(uu)*0
            etaup[:]=-beta[:]/alpha # spatial part
            etaup[0]=1.0/alpha
            #
            gammarel=-uu[0]*eta[0]
            urel=np.copy(uu)*0
            urel=uu - gammarel * etaup  # so uu = urel + gamma * (-beta/alpha)
            #
            # only spatial part of urel will be used
            urel[:,:,0,:]=urel[:,:,3,:]
            urel[:,:,1,:]=urel[:,:,3,:]
            urel[:,:,2,:]=urel[:,:,3,:]
            urel[:,:,ny-1,:]=urel[:,:,ny-4,:]
            urel[:,:,ny-2,:]=urel[:,:,ny-4,:]
            urel[:,:,ny-3,:]=urel[:,:,ny-4,:]
            #
            # now back to uu
            qsq = gv3[1,1]*urel[1]*urel[1] + gv3[2,2]*urel[2]*urel[2] + gv3[3,3]*urel[3]*urel[3] + 2.0*(gv3[1,2]*urel[1]*urel[2] + gv3[1,3]*urel[1]*urel[3] + gv3[2,3]*urel[2]*urel[3])
            gamma = np.sqrt(1.0 + qsq)
            #
            uu[:]=urel[:]-(gamma/alpha)*beta[:] # spatial part
            uu[0]=gamma/alpha

    #
    #B = np.zeros_like(uu)
    #cell-centered magnetic field components
    #B[1:4,:,:,:]=d[8:11,:,:,:]
    # start at 7 so B[1] is correct.  7=<ignore> 8=B[1] 9=B[2] 10=B[3]
    B=np.zeros((4,nx,ny,nz),dtype='float32',order='F')
    # have to make copy so mods to B[0] won't change d[7]
    B=np.copy(d[7:11,:,:,:])
    B[0]=0*B[0]
    #
    #
    #
    print(("numcolumnshere: %d" % (numcolumns))) ; sys.stdout.flush()
    #
    gotgdetB=0
    if(d.shape[0]>=14 and numcolumns==11+3):
        gotgdetB=1
        print(("Getting gdetB: dshape0=%d" % (d.shape[0]))) ; sys.stdout.flush()
        #new image format additionally contains gdet*B^i
        #face-centered magnetic field components multiplied by gdet
        # below assumes gdetB[0] is never needed
        gdetB=np.zeros((4,nx,ny,nz),dtype='float32',order='F')
        # have to make copy so mods to gdetB[0] won't change B[3]
        gdetB = np.copy(d[10:14,:,:,:])
        gdetB[0]=0*gdetB[0]
    #
    #
    ######################################################
    # get any radiation variables
    #
    global gotrad
    gotrad=0
    if(numcolumns==16):
        gotrad=1
        Erf=np.zeros((1,nx,ny,nz),dtype='float32',order='F')
        uradu=np.zeros((4,nx,ny,nz),dtype='float32',order='F')
        Erf=d[11,:,:,:] # radiation frame radiation energy density
        #
        # approximation, but correct if used in pressure ultimately
        urad=Erf
        #
        uradu=d[12:16,:,:,:]  #again, note uu[i] are 3-velocities (as read from the fieldline file)
        #multiply by u^t to get 4-velocities: u^i = u^t v^i
        uradu[1:4]=uradu[1:4] * uradu[0]
        #
        maxErf=np.max(Erf)
        minErf=np.min(Erf)
        print(("maxErf=%g minErf=%g" % (maxErf,minErf))) ; sys.stdout.flush()
        #
    else:
        Erf=rho*0+1E-30
        urad=Erf*0+1E-30
        uradd=uu*0+1E-30
        uradu=uu*0+1E-30
    #
    global TRACKVPOT,        MCOORD,        DODISS,        DOEVOLVEMETRIC,        WHICHVEL,        WHICHEOM,        REMOVERESTMASSFROMUU,        RELTYPE,        EOMTYPE,        WHICHEOS,        DOENTROPY,        WHICHENTROPYEVOLVE,        CALCFARADAYANDCURRENTS,        DOPOLEDEATH,        DOPOLESMOOTH,        DOPOLEGAMMADEATH,        IF3DSPCTHENMPITRANSFERATPOLE,        EOMRADTYPE,        WHICHRADSOURCEMETHOD,        OUTERDEATH,        OUTERDEATHRADIUS
    if(EOMTYPE==0):
        rho=rho+1E-30
        ug=ug+1E-30
    #
    #
    #############################################################################################
    # see if THETAROT non-zero so need to rotate and transform data
    #
    #
    #DEBUGTHETAROT=1
    DEBUGTHETAROT=0
    #
    #
    # Only need to rotate data to align with THETAROT0 gdump if use2dglobal==True
    if use2dglobal==True and (DEBUGTHETAROT or np.fabs(THETAROT-0.0)>1E-13):
        #
        print(("rfd(before rfdtransform) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
        # whether to always do transformation (for testing)
        # save result so can use it if repeat
        fnamenpz =  "dumps/" + fieldlinefilename + ".npz"
        print(("THETAROT=%21.15g for fnamenpz=%s" % (THETAROT,fnamenpz))) ; sys.stdout.flush()
        #
        # but see: http://stackoverflow.com/questions/82831/how-do-i-check-if-a-file-exists-using-python
        if os.path.exists(fnamenpz):
            print(("THETAROT=%21.15g for loadz" % (THETAROT))) ; sys.stdout.flush()
            #
            #http://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html#numpy.load
            #data=np.load(fnamenpz,mmap_mode='r')
            data=np.load(fnamenpz)
            # no idea why default is reversed order of array put into npz
            #datanameslist=data.files.reverse()
            #rho=data[datanameslist[0]]
            #ug=data[datanameslist[1]]
            #uu=data[datanameslist[2]]
            #B=data[datanameslist[3]]
            #if gotgdetB==1:
            #    gdetB=data[datanameslist[4]]
            rho=data['rho']
            ug=data['ug']
            uu=data['uu']
            B=data['B']
            if gotgdetB==1:
                gdetB=data['gdetB']
            #
            if numcolumns==16:
                Erf=data['Erf']
                uradu=data['uradu']
            data.close()
        else:
            print(("THETAROT=%21.15g for rfdtransform" % (THETAROT))) ; sys.stdout.flush()
            # then need to get transformed quantities
            # transform uu,B into coordinates where spin is pointing in zhat.
            rfdtransform(gotgdetB=gotgdetB)
            #
            # need to save since doesn't exist yet
            if gotgdetB==1:
                np.savez(fnamenpz,rho=rho,ug=ug,uu=uu,B=B,gdetB=gdetB)
            else:
                np.savez(fnamenpz,rho=rho,ug=ug,uu=uu,B=B)
            #
            #
            print(("rfd(after rfdtransform) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
    else:
        print(("No rdtrans for file=%s with THETAROT=%g" % (fname,THETAROT))) ;  sys.stdout.flush()
    #
    #
    #############################################################################################
    # compute extra things
    #
    ######################
    rfdprocess(gotgdetB=gotgdetB)
    #
    #############################################################################################
    #### now do radiation-dependent stuff
    rddims(gotrad)
    getkappas(gotrad)
    #
    #
    print(("rfd(after rfdprocess) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()

def rfdheader(fin=None):
    global t,nx,ny,nz,startx1,startx2,startx3,_dx1,_dx2,_dx3,nstep,gam,a,R0,Rin,Rout,hslope,rundt,defcoord
    global MBH,QBH,EP3,THETAROT,_is,_ie,_js,_je,_ks,_ke,whichdump,whichdumpversion,numcolumns
    global rhor
    #global header
    #
    #
    header = fin.readline().split()
    #
    numheaderitems=len(header) #.shape[0]
    #
    #
    #time of the dump
    t = myfloatalt(np.float64(header[0]))
    print(("rfdheader: t=%g" % (t))) ; sys.stdout.flush()
    #dimensions of the grid
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    #
    startx1=myfloatalt(float(header[4]))
    startx2=myfloatalt(float(header[5]))
    startx3=myfloatalt(float(header[6]))
    #cell size in internal coordintes
    _dx1=myfloatalt(float(header[7]))
    _dx2=myfloatalt(float(header[8]))
    _dx3=myfloatalt(float(header[9]))
    #
    nstep=int(header[10])
    #other information:
    #polytropic index
    gam=myfloatalt(float(header[11]))
    #black hole spin
    a=myfloatalt(float(header[12]))
    rhor=1+(1-a**2)**0.5
    R0=myfloatalt(float(header[13]))
    #Spherical polar radius of the innermost radial cell
    Rin=myfloatalt(float(header[14]))
    #Spherical polar radius of the outermost radial cell
    Rout=myfloatalt(float(header[15]))
    #
    hslope=myfloatalt(float(header[16]))
    #
    rundt=myfloatalt(float(header[17]))
    defcoord=int(header[18])
    #
    global TRACKVPOT,        MCOORD,        DODISS,        DOEVOLVEMETRIC,        WHICHVEL,        WHICHEOM,        REMOVERESTMASSFROMUU,        RELTYPE,        EOMTYPE,        WHICHEOS,        DOENTROPY,        WHICHENTROPYEVOLVE,        CALCFARADAYANDCURRENTS,        DOPOLEDEATH,        DOPOLESMOOTH,        DOPOLEGAMMADEATH,        IF3DSPCTHENMPITRANSFERATPOLE,        EOMRADTYPE,        WHICHRADSOURCEMETHOD,        OUTERDEATH,        OUTERDEATHRADIUS
    if numheaderitems==53:
        TRACKVPOT=int(header[32])
        MCOORD=int(header[33])
        DODISS=int(header[34])
        DOEVOLVEMETRIC=int(header[35])
        WHICHVEL=int(header[36])
        WHICHEOM=int(header[37])
        REMOVERESTMASSFROMUU=int(header[38])
        RELTYPE=int(header[39])
        EOMTYPE=int(header[40])
        WHICHEOS=int(header[41])
        DOENTROPY=int(header[42])
        WHICHENTROPYEVOLVE=int(header[43])
        CALCFARADAYANDCURRENTS=int(header[44])
        DOPOLEDEATH=int(header[45])
        DOPOLESMOOTH=int(header[46])
        DOPOLEGAMMADEATH=int(header[47])
        IF3DSPCTHENMPITRANSFERATPOLE=int(header[48])
        EOMRADTYPE=int(header[49])
        WHICHRADSOURCEMETHOD=int(header[50])
        OUTERDEATH=int(header[51])
        OUTERDEATHRADIUS=int(header[52])
    else:
        # just set defaulst that will generally work for default setup of GRMHD, etc.
        TRACKVPOT=1 # yes
        MCOORD=2 # KSCOORDS
        DODISS=0 # no
        DOEVOLVEMETRIC=0 # no
        WHICHVEL=2 # VELREL4
        WHICHEOM=0 # WITHGDET
        REMOVERESTMASSFROMUU=2 # fully removed
        RELTYPE=0 # normal rel
        EOMTYPE=3 # GRMHD=3 and FFDE=0
        WHICHEOS=1 # ideal gas
        DOENTROPY=1 # DOEVOLVEENTROPY
        WHICHENTROPYEVOLVE=1 # EVOLVESIMPLEENTROPY with DODISS=0 and EVOLVEFULLENTROPY with off
        CALCFARADAYANDCURRENTS=1 # yes
        DOPOLEDEATH=0
        DOPOLESMOOTH=0
        DOPOLEGAMMADEATH=0
        IF3DSPCTHENMPITRANSFERATPOLE=0
        EOMRADTYPE=0 # EOMRADNONE
        WHICHRADSOURCEMETHOD=3 # SOURCEMETHODIMPLICIT
        OUTERDEATH=0 # no
        OUTERDEATHRADIUS=1E7
    #
    if numheaderitems>=32:
        print("Found 32 header items, reading them in\n")  ; sys.stdout.flush()
        MBH=myfloatalt(float(header[19]))
        QBH=myfloatalt(float(header[20]))
        EP3=myfloatalt(float(header[21]))
        THETAROT=myfloatalt(float(header[22]))
        #
        _is=int(header[23])
        _ie=int(header[24])
        _js=int(header[25])
        _je=int(header[26])
        _ks=int(header[27])
        _ke=int(header[28])
        whichdump=int(header[29])
        whichdumpversion=int(header[30])
        numcolumns=int(header[31])
    #
    if numheaderitems==31:
        print("Found 31 header items, reading them in and setting THETAROT=0.0\n")  ; sys.stdout.flush()
        MBH=myfloatalt(float(header[19]))
        QBH=myfloatalt(float(header[20]))
        EP3=myfloatalt(float(header[21]))
        THETAROT=0.0
        #
        _is=int(header[22])
        _ie=int(header[23])
        _js=int(header[24])
        _je=int(header[25])
        _ks=int(header[26])
        _ke=int(header[27])
        whichdump=int(header[28])
        whichdumpversion=int(header[29])
        numcolumns=int(header[30])
    #
    if numheaderitems==30:
        print("Found 30 header items, reading them in and setting EP3=THETAROT=0.0\n")  ; sys.stdout.flush()
        MBH=myfloatalt(float(header[19]))
        QBH=myfloatalt(float(header[20]))
        EP3=0.0
        THETAROT=0.0
        #
        _is=int(header[21])
        _ie=int(header[22])
        _js=int(header[23])
        _je=int(header[24])
        _ks=int(header[25])
        _ke=int(header[26])
        whichdump=int(header[27])
        whichdumpversion=int(header[28])
        numcolumns=int(header[29])

def rfdheaderonly(filename="dumps/fieldline0000.bin"):
    fin = open(filename, "rb" )
    rfdheader(fin=fin)
    fin.close()

def rfdheaderlastfile():
    flist = glob.glob( os.path.join("dumps/", "fieldline*.bin") )
    sort_nicely(flist)
    lastfieldlinefile=flist[-1]
    #rfd("fieldline0000.bin")  #to definea
    rfdheaderonly(lastfieldlinefile)

def rfdprocess(gotgdetB=0):
    #
    # external globals
    global rho,ug,uu,B,gdetB,Erf,urad,uradu
    # derived quantities
    global lrho,rholab,lrholab,ug,uut,uu,uux,rhor,r,h,ph,rhoclean,rholabclean,rhounclean,rholabunclean,ugclean,ugunclean,uuclean,entropy
    global gdetB # tells either exists before or will be created here
    global maxbsqorhonear,maxbsqorhofar,condmaxbsqorho,condmaxbsqorhorhs,rinterp
    #
    #
    lrho=np.zeros((1,nx,ny,nz),dtype='float32',order='F')
    lrho = np.log10(rho)
    # lab-frame density
    rholab=rho*uu[0]
    lrholab = np.log10(rholab)
    #
    #if the input file contains additional data
    #
    if(gotgdetB==0):
        print("No data on gdetB, approximating it.") ; sys.stdout.flush()
        gdetB = np.zeros((4,nx,ny,nz),dtype='float32',order='F')
        print("shapes:") ; sys.stdout.flush()
        print((gdet.shape)) ; sys.stdout.flush()
        print((B.shape)) ; sys.stdout.flush()
        print((gdetB.shape)) ; sys.stdout.flush()
        gdetB[1:4] = gdet * B[1:4]
        #
        #
    #
    # get floor-cleaned versions (override uu[3] completely)
    #
    getbsq_pre()
    (rhoclean,ugclean,uuclean,maxbsqorhonear,maxbsqorhofar,condmaxbsqorho,condmaxbsqorhorhs,rinterp)=getrhouclean(rho,ug,uu)
    maxbsqorho=np.copy(maxbsqorhonear)
    uu=np.copy(uuclean)
    rhounclean=np.copy(rho)
    ugunclean=np.copy(ug)
    #
    # lab-frame versions of clean densities
    rholabunclean=rhounclean*uu[0]
    rholabclean=rhoclean*uu[0]
    #
    if 1==0:
        # go here if want rho and ug to be clean versions always.  Including for movie frames.
        # assume if need bsqorho or something like that, ok if goes to infinity (except some plots need fixing)
        rho=np.copy(rhoclean)
        ug=np.copy(ugclean)
    #
    #     if 'gdet' in globals():
    #         #first set everything approximately (B's are at shifted locations by half-cell)
    #         B = gdetB/gdet
    #         #then, average the inner cells to proper locations
    #         B[1,0:nx-1,:,:] = 0.5*(gdetB[1,0:nx-1,:,:]+gdetB[1,1:nx,:,:])/gdet[0:nx-1,:,:]
    #         B[2,:,0:ny-1,:] = 0.5*(gdetB[2,:,0:ny-1,:]+gdetB[2,:,1:ny,:])/gdet[:,0:ny-1,:]
    #         B[3,:,:,0:nz-1] = 0.5*(gdetB[3,:,:,0:nz-1]+gdetB[3,:,:,1:nz])/gdet[:,:,0:nz-1]
    #         #note: last cells on the grids (near upper boundaries of each dir are at
    #         #      approximate locations
    #     else:
    #         print( "rfd: warning: since gdet is not defined, I am skipping the computation of cell-centered fields, B" )
    # else:
    if 'r' in globals() and r.shape[2] != nz:
        #dynamically change the 3rd dimension size
        rnew = np.zeros((nx,ny,nz),dtype=r.dtype)
        hnew = np.zeros((nx,ny,nz),dtype=h.dtype)
        phnew = np.zeros((nx,ny,nz),dtype=ph.dtype)
        rnew += r[:,:,0:1]
        hnew += h[:,:,0:1]
        #compute size of phi wedge assuming dxdxp[3][3] is up to date
        phiwedge = dxdxp[3][3][0,0,0]*_dx3*nz
        a_phi = phiwedge/(2.*nz)+np.linspace(0,phiwedge,num=nz,endpoint=False)
        phnew += a_phi[None,None,:]
        del r
        del h
        del ph
        r = rnew
        h = hnew
        ph = phnew
        print("phnew") ; sys.stdout.flush()
        print((phnew[0,0,:])) ; sys.stdout.flush()
        gc.collect()
    else:
        print("r in globals has shape2 of nz") ; sys.stdout.flush()
    #
    # other stuff
    entropy=(gam-1.0)*ugclean/rho**(gam)

# handle THETAROT!=0
def rfdtransform(gotgdetB=0):
    #
    # get starting time so can compute time differences
    start_time=datetime.now()
    print(("rfdtransform(start) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
    #
    # modified globals
    global nzgdump
    global r,h,ph
    global rho,ug,uu,B,gdetB,Erf,urad,uradu
    #
    print(("what: %d\n",nz)) ; sys.stdout.flush()
    print(("what2: %d\n",nzgdump)) ; sys.stdout.flush()
    #
    ###########################################
    # first deal with ti,tj,tk,x1,x2,x3,r,h,ph that are gdump data size, while need true ph at least on same sized-grid as rfd() data is on.
    ###########################################
    #
    ti1d=ti[:,0,0].view().reshape(-1)
    tj1d=tj[0,:,0].view().reshape(-1)
    tk1d=tk[0,0,:].view().reshape(-1)
    tk1dnew=0.5 + np.float32(np.arange(0,nz)) # cell centered tk using rfd's nz
    x11d=x1[:,0,0].view().reshape(-1)
    x21d=x2[0,:,0].view().reshape(-1)
    x31d=x3[0,0,:].view().reshape(-1)
    endx3=startx3 + _dx3*nz
    x31dnew=startx3 + tk1dnew*(endx3-startx3)/(nz-0.0)
    # r and h are not constant along x1,x2 grid lines.  Only ph is constant for constant tk,x3.
    #tk1dnew = griddata(tk1d, vartointerp, (V[1,:,0,0], V[2,0,:,0], V[3,0,0,:]), method='linear')
    #    zi = griddata((x, y), lrho, (xi[None,:], yi[:,None]), method='cubic')
    ph1d=ph[0,0,:].view().reshape(-1)
    ph1dnew=2.0*np.pi*x31dnew  # i.e. x3=0 is phi=0 and x3=1 is phi=2pi

    print("ph1dnew")
    print(ph1dnew) ; sys.stdout.flush()
    #griddata(tk1d, x31d, tk1dnew, method='cubic')
    #print("shapes: ",x1.shape,x2.shape,x3.shape,r.shape,x11d.shape,x11d[:,None,None].shape,x21d.shape,x21d[None,:,None].shape,x31dnew.shape,x31dnew[None,None,:].shape) ; sys.stdout.flush()
    #
    #http://mail.scipy.org/pipermail/astropy/2011-April/001255.html
    print(("pre-r2d shape (nx=%d ny=%d)" % (nx,ny))) ; sys.stdout.flush()
    print((r[:,:,0].shape)) ; sys.stdout.flush()
    # .view().reshape(-1)
    r2d=r[:,:,0].view().reshape((nx,ny))
    h2d=h[:,:,0].view().reshape((nx,ny))
    #print("r2d")
    #print(r2d)
    #print("h2d")
    #print(h2d)
    #r3d = np.tile(r2d,nz).view().reshape((nx,ny,nz))
    #h3d = np.tile(h2d,nz).view().reshape((nx,ny,nz))
    #r3d = np.tile(r2d,(nz,1,1))
    r3d = np.transpose(np.tile(r2d,(nz,1,1)),(1,2,0))
    #h3d = np.tile(h2d,(nz,1,1))
    h3d = np.transpose(np.tile(h2d,(nz,1,1)),(1,2,0))
    #print("r3d")
    #print(r3d[:,0,0])
    print("h3d")
    print((h3d[0,:,0]))
    #rnew = griddata((x1, x2, x3), r, (x11d[:,None,None], x21d[None,:,None], x31dnew[None,None,:]), method='linear')
    #r=rnew # overwrite
    #hnew = griddata((x1, x2, x3), h, (x11d, x21d, x31dnew), method='linear')
    #h=hnew # overwrite
    # below phnew works if ph(x3) only and not ph(x1,x2).  This is currently true.  Using np.tile is more accurate than using griddata that would extrapolate near boundaries.
    #http://stackoverflow.com/questions/5559851/numpy-constructing-a-3d-array-from-a-1d-array
    #ph3d = np.tile(ph1dnew,nx*ny).reshape((nx,ny,nz))
    ph3d = np.tile(ph1dnew,(nx,ny,1))
    #
    # DEBUG:
    #print("gods: %g %g %g : %g %g %g\n" % (r[20,15,0],h[20,15,0],ph[20,15,0],r3d[20,15,20],h3d[20,15,20],ph3d[20,15,20]))
    #
    # set nzgdump since updated 3d things.  Now won't have to do this again unless read-in gdump again.
    #    nzgdump=nz
    #
    #
    ###########################################
    # Assign Vmetric
    ###########################################
    #
    Vmetric=np.zeros((4,nx,ny,nz),dtype='float32')
    Vmetric[0]=Vmetric[0]*0.0
    Vmetric[1]=r3d
    Vmetric[2]=h3d
    Vmetric[3]=ph3d
    #
    print(("rfdtransform(done assign Vmetric) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
    # DEBUG:
    #print("shapes")
    #print(Vmetric[1].shape)
    #print(Vmetric[2].shape)
    #print(Vmetric[3].shape)
    #sys.stdout.flush()
    #
    #
    ###########################################
    # Assign Vorig
    ###########################################
    #
    print("rotate_VtoVmetric BEGIN\n");sys.stdout.flush()
    # get Vorig if different than Vmetric
    # This assumes original grid use same exact r,\theta,\phi grid on same x1,x2,x3 and same ti,tj,tk.
    Vorig=np.copy(Vmetric)
    # no time-component operations
    # get Vmetric(V)
    # Note that ti,tj,tk,x1,x2,x3,r,h,ph,dxdxp don't change -- we simply identify them with Vmetric instead of V, so that V no longer is relevant except as required to interpolate grid data to Vmetric positions from V positions and to transform vectors/tensors as required.  But we still need the stored value of Vorig on the grid, which means we need Vorig=V(Vmetric).
    # assume ti,tj,tk,x1,x2,x3,r,h,ph,dxdxp already rotated, and now Vmetric already.
    # then just get original V
    #Vorig=rotate_Vmetric2V(Vmetric,Vorig,b0=-THETAROT)
    # NO: use Vmetric for fieldline data's V as required, so that get out de-rotated version of r,th,ph as required
    # takes (e.g.) zhat and moves it -15deg around y-axis, so that below the reinterp3dspc() assumes Vmetric offset with +15deg around y-axis meant original data's zaxis that's is not offset and so shows up at +15deg.
    Vorig=rotate_VtoVmetric(Vmetric,Vorig,b0=-THETAROT)
    #
    #print("Vmetricalongj")
    #print(Vmetric[2,0,:,0])
    #sys.stdout.flush()
    #
    #print("Vorigalongj")
    #print(Vorig[2,0,:,0])
    #sys.stdout.flush()
    #
    print("rotate_VtoVmetric END\n");sys.stdout.flush()
    print(("rfdtransform(done assign Vorig) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
    #
    #
    ###########################################
    # Interpolate (and transform vectors from X->V)
    ###########################################
    #
    print("reinterp3dspc BEGIN\n");sys.stdout.flush()
    #
    if 0==1:
        # 3D interpolate from Vorig positions on grid to Vmetric positions
        # in this context, Vorig is de-rotated r,th,ph, while Vmetric is new grid with BH spin axis pointing along zhat.
        rhoi=reinterp3dspc(Vorig,Vmetric,rho)
        ugi=reinterp3dspc(Vorig,Vmetric,ug)
        uu0i=reinterp3dspc(Vorig,Vmetric,uu[0])
        uu1i=reinterp3dspc(Vorig,Vmetric,uu[1])
        uu2i=reinterp3dspc(Vorig,Vmetric,uu[2])
        uu3i=reinterp3dspc(Vorig,Vmetric,uu[3])
        B1i=reinterp3dspc(Vorig,Vmetric,B[1])
        B2i=reinterp3dspc(Vorig,Vmetric,B[2])
        B3i=reinterp3dspc(Vorig,Vmetric,B[3])
        if gotgdetB==1:
            gdetB1i=reinterp3dspc(Vorig,Vmetric,gdetB[1])
            gdetB2i=reinterp3dspc(Vorig,Vmetric,gdetB[2])
            gdetB3i=reinterp3dspc(Vorig,Vmetric,gdetB[3])
        #
        # NOT SETUP FOR URADU GODMARK
        rho=rhoi
        ug=ugi
        uu[0]=uu0i
        uu[1]=uu1i
        uu[2]=uu2i
        uu[3]=uu3i
        #B[0] is still zero
        B[1]=B1i
        B[2]=B2i
        B[3]=B3i
        if gotgdetB==1:
            #gdetB[0] is still zero
            gdetB[1]=gdetB1i
            gdetB[2]=gdetB2i
            gdetB[3]=gdetB3i
    else:
        # Jon's SPC-optimized version of inteprolation
        # and overwrite non-interpolated versions
        if gotgdetB==1:
            (rho,ug,uu[0],uu[1],uu[2],uu[3],B[1],B[2],B[3],gdetB[1],gdetB[2],gdetB[3])=reinterp3dspc_opt_all(Vorig,Vmetric,rho,ug,uu,B,gdetB=gdetB)
        else:
            (rho,ug,uu[0],uu[1],uu[2],uu[3],B[1],B[2],B[3])=reinterp3dspc_opt_all(Vorig,Vmetric,rho,ug,uu,B)
    #
    print("reinterp3dspc END\n");sys.stdout.flush()
    print(("rfdtransform(done assign interpolated prims) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
    #
    #
    ###########################################
    # Get transformation matrices that actually only depend upon Vmetric=r,h,ph
    ###########################################
    #
    # transV2Vmetric^\mu[Vmetric]_\nu[V] u^\nu[V] : So first index is Vmetric-type.  Second index is V-type.  Operates on contravariant V-type.
    # transVmetric2V^\mu[V]_\nu[Vmetric] u^\nu[Vmetric] : So first index is V-type.  Second index is Vmetric-type.  Operates on contravariant Vmetric-type.
    print("set_transV2Vmetric BEGIN\n");sys.stdout.flush()
    #
    (transV2Vmetric)=set_transV2Vmetric(Vmetric=Vmetric,b0=-THETAROT)
    #(transV2Vmetric)=set_transV2Vmetric(Vmetric=Vmetric,b0=+THETAROT)
    gc.collect() #try to release unneeded memory
    #
    # test transV2Vmetric
    #
    utest1old=np.zeros((4),dtype=r.dtype)
    utest1old[0]=1.0
    utest1old[1]=1.0
    utest1old[2]=0.0
    utest1old[3]=0.0
    utest1new=np.tensordot(utest1old,transV2Vmetric[:,:,60,ny//2,nz/4],axes=[0,1])
    print("utest1new");sys.stdout.flush()
    print(utest1new);sys.stdout.flush()
    #
    utest1old=np.zeros((4),dtype=r.dtype)
    utest1old[0]=1.0
    utest1old[1]=0.0
    utest1old[2]=1.0
    utest1old[3]=0.0
    utest1new=np.tensordot(utest1old,transV2Vmetric[:,:,60,ny//2,nz/4],axes=[0,1])
    print("utest2new");sys.stdout.flush()
    print(utest1new);sys.stdout.flush()
    #
    utest1old=np.zeros((4),dtype=r.dtype)
    utest1old[0]=1.0
    utest1old[1]=0.0
    utest1old[2]=0.0
    utest1old[3]=1.0
    utest1new=np.tensordot(utest1old,transV2Vmetric[:,:,60,ny//2,nz/4],axes=[0,1])
    print("utest3new");sys.stdout.flush()
    print(utest1new);sys.stdout.flush()
    #
    #    print("set_transV2Vmetric END\n");sys.stdout.flush()
    print(("rfdtransform(done get transV2Vmetric) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
    #
    ###########################################
    # Transform vector components from Vorig -> Vmetric -> Xmetric
    ###########################################
    #
    # transform tensors (gv3=gv3_{\mu[V]\nu[V]} and dxdxp=dx^\mu[V]/dxp^\nu[V], but dxdxp just dV/dX that we understand now is just dVmetric/dXmetric that is already new grid by assumed rotation)
    # So only have to transform gv3 (everything remains correct/consistent as long as all vector components are interpolated in same spatial way and transformed in correct/consistent way)
    # using transV just does a local rotation, not global.  Global relocation done by interpolation already.
    # u^\mu[V] -> u^\mu[Vmetric]
    # problem is u.u=-1 won't be satisfied except to truncation error.  But for python analysis, never need it to be exactly correct.
    #
    # full transformation is:
    # dx^{\mu'''[X]} = dx^{\mu[Xmetric]} \Lambda^{\mu'[Vmetric]}_{\mu[Xmetric]} \Lambda^{\mu''[V]}_{\mu'[Vmetric]} \Lambda^{\mu'''[X]}_{\mu''[V]}
    # dx^{\mu'''[Xmetric]} = dx^{\mu[X]} \Lambda^{\mu'[V]}_{\mu[X]} \Lambda^{\mu''[Vmetric]}_{\mu'[V]} \Lambda^{\mu'''[Xmetric]}_{\mu''[Vmetric]} <--- this one is what we do here.
    #
    # Note that dxdxp=dV^\mu/dX^\nu=\Lambda^\mu[V]_\nu[X] at Vorig on original grid is same value as dVmetric^\mu/dXmetric^\nu on Vmetric on new grid. But, transV2Vmetric has to operate on V, not X.
    print("tensordots BEGIN\n");sys.stdout.flush()
    #
    # this gives inverse without transposition, so gives (dX^\nu/dV^\mu)^T = \Lambda_\mu[V]^\nu[X] and \Lambda_\mu[Vmetric]^\nu[Xmetric]
    idxdxp=np.copy(dxdxp)
    for ii in np.arange(0,nx):
        for jj in np.arange(0,ny):
            idxdxp[:,:,ii,jj,0]=np.linalg.inv(dxdxp[:,:,ii,jj,0])
    #
    print("About to do uunew1");sys.stdout.flush()
    printusage()
    #
    if 0==1:
        uunew1=tensordot01(uu,dxdxp) #,axes=([0],[1])) # now u^\mu[V]
    else:
        uunew1=uu # assume already applied dxdxp at same r,theta,phi location as uu
    #
    printusage()
    uunew2=tensordot01(uunew1,transV2Vmetric) #,axes=([0],[1])) # now u^\nu[Vmetric]
    uunew3=tensordot00(uunew2,idxdxp) #,axes=([0],[0])) # now u^\nu[Xmetric]
    uu=np.copy(uunew3) # overwrite
    #
    # assumes no transformation on time component, which is true for the spatial rotation involving THETAROT
    if 0==1:
        Bnew1=tensordot01(B,dxdxp) #,axes=([0],[1])) # now u^\mu[V]
    else:
        Bnew1=B # assume already applied dxdxp at same r,theta,phi location as uu
    #
    Bnew2=tensordot01(Bnew1,transV2Vmetric) #,axes=([0],[1])) # now u^\nu[Vmetric]
    Bnew3=tensordot00(Bnew2,idxdxp) #,axes=([0],[0])) # now u^\nu[Xmetric]
    B=np.copy(Bnew3) # overwrite
    #
    if gotgdetB==1:
        # assumes no transformation on time component, which is true for the spatial rotation involving THETAROT
        # assume for gdetB that gdet changes little for Vorig and Vmetric.  Inaccurate near BH, but only use gdetB in special cases.  Even very near BH, gdet doesn't change too much with THETAROT changes.
        # To get accurate, would have to divide out gdet[Vorig] and multiply by gdet[V].
        # But, then divB=0 won't be very accurate still.  To have that, would have to form A_i and recompute face values of gdetB.
        if 0==1:
            gdetBnew1=tensordot01(gdetB,dxdxp) #,axes=([0],[1])) # now u^\mu[V]
        else:
            gdetBnew1=gdetB # assume already applied dxdxp at same r,theta,phi location as uu
        #
        gdetBnew2=tensordot01(gdetBnew1,transV2Vmetric) #,axes=([0],[1])) # now u^\nu[Vmetric]
        gdetBnew3=tensordot00(gdetBnew2,idxdxp) #,axes=([0],[0])) # now u^\nu[Xmetric]
        gdetB=np.copy(gdetBnew3) # overwrite
    #
    # NOT SETUP FOR URADU GODMARK
    #
    printusage()
    print("tensordots END\n");sys.stdout.flush()
    print(("rfdtransform(done tensordots) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()

# need bsq even if modify later ud or uu, but assume bsq not sensitive to those modifications so this is ok
def getbsq_pre():
    global ud, bu, bd, bsq
    ud = mdot(gv3,uu)                  #g_mn u^n
    #
    bu=np.empty_like(uu)              #allocate memory for bu
    #set component per component
    bu[0]=mdot(B[1:4], ud[1:4])             #B^i u_i
    bu[1:4]=(B[1:4] + bu[0]*uu[1:4])/uu[0]  #b^i = (B^i + b^t u^i)/u^t
    bd=mdot(gv3,bu)
    bsq=mdot(bu,bd)

def rddims(gotrad):
    global GGG,CCCTRUE,MSUNCM,MPERSUN,LBAR,TBAR,VBAR,RHOBAR,MBAR,ENBAR,UBAR,TEMPBAR,ARAD_CODE_DEF,XFACT,ZATOM,AATOM,MUE,MUI,OPACITYBAR,MASSCM,KORAL2HARMRHO1,Leddcode,Mdoteddcode,rhoeddcode,ueddcode,beddcode
    if(gotrad==1):
        # then also get radiation constants
        fname= "dimensions.txt"
        fin = open(fname, "rt" )
        dimfile = fin.readline().split()
        numheaderitems=len(dimfile) #.shape[0]
        #
        GGG = np.float64(dimfile[0])
        CCCTRUE = np.float64(dimfile[1])
        MSUNCM = np.float64(dimfile[2])
        MPERSUN = np.float64(dimfile[3])
        LBAR = np.float64(dimfile[4])
        TBAR = np.float64(dimfile[5])
        VBAR = np.float64(dimfile[6])
        RHOBAR = np.float64(dimfile[7])
        MBAR = np.float64(dimfile[8])
        ENBAR = np.float64(dimfile[9])
        UBAR = np.float64(dimfile[10])
        TEMPBAR = np.float64(dimfile[11])
        ARAD_CODE_DEF = np.float64(dimfile[12])
        XFACT = np.float64(dimfile[13])
        ZATOM = np.float64(dimfile[14])
        AATOM = np.float64(dimfile[15])
        MUE = np.float64(dimfile[16])
        MUI = np.float64(dimfile[17])
        OPACITYBAR = np.float64(dimfile[18])
        MASSCM = np.float64(dimfile[19])
        KORAL2HARMRHO1 = np.float64(dimfile[20])
        fin.close()
        #
        MSUN=1.9891E33
        sigmaT=0.665E-24
        mproton=1.673E-24
        einf,linf=elinfcalc(a)
        effnom=1.0-einf
        #
        Ledd=4*np.pi*GGG*(MPERSUN*MSUN)*mproton*CCCTRUE/sigmaT
        Leddcode = Ledd/ENBAR*TBAR
        Mdotedd = Ledd/(CCCTRUE**2*effnom)
        Mdoteddcode = Mdotedd/MBAR*TBAR
        rhoedd = Mdotedd/CCCTRUE*(GGG*MPERSUN*MSUN/CCCTRUE**2)/(GGG*MPERSUN*MSUN/CCCTRUE**2)**3
        rhoeddcode = rhoedd/RHOBAR
        uedd = rhoedd*CCCTRUE**2
        bedd = np.sqrt(uedd)
        ueddcode = uedd/UBAR
        beddcode = bedd/np.sqrt(UBAR)
        #
        print(("CCCTRUE=%g ENBAR=%g TBAR=%g Ledd=%g Mdotedd=%g einf=%g linf=%g uedd=%g bedd=%g" % (CCCTRUE,ENBAR,TBAR,Ledd,Mdotedd,einf,linf,uedd,bedd))) ; sys.stdout.flush()
        print(("CCCTRUE=%g ENBAR=%g TBAR=%g" % (CCCTRUE,ENBAR,TBAR))) ; sys.stdout.flush()
    else:
        GGG=1
        CCCTRUE=1
        MSUNCM=1
        MPERSUN=1
        LBAR=1
        TBAR=1
        VBAR=1
        RHOBAR=1
        MBAR=1
        ENBAR=1
        UBAR=1
        TEMPBAR=1
        ARAD_CODE_DEF=1
        XFACT=1
        ZATOM=1
        AATOM=1
        MUE=1
        MUI=1
        OPACITYBAR=1
        MASSCM=1
        KORAL2HARMRHO1=1
        #
        Ledd=1
        Leddcode=1
        Mdotedd=1
        Mdoteddcode=1
        rhoedd=1
        rhoeddcode=1
        uedd=1
        ueddcode=1
        bedd=1
        beddcode=1

def getkappas(gotrad):
    global KAPPAUSER,KAPPAESUSER
    if(gotrad==0):
        KAPPA=1.0
        KAPPAES=1.0
        # Below should be as same in global.depmnemonics.rad.h
        TEMPMINKELVIN=1.0E-10 # Kelvin
        TEMPMIN=(TEMPMINKELVIN/TEMPBAR)
        pg=(gam-1.0)*ug # ideal gas
        prad=(4.0/3.0-1.0)*urad # radiation isotropic in some frame
        Tgas=pg/rho # gas temperature for ideal gas
        KAPPAUSER=rho*0
        KAPPAESUSER=rho*0
        #
    else:
        # now compute auxillary opacity related quantities since only otherwise in raddump???? files and not in fieldline files
        KAPPA=1.0
        KAPPAES=1.0
        # KORALTODO: Put a lower limit on T~1E4K so not overly wrongly opaque in spots where u_g->0 anomologously?
        TEMPMINKELVIN=1.0E-10 # Kelvin
        TEMPMIN=(TEMPMINKELVIN/TEMPBAR)
        # ideal gas assumed for Tgas
        # code pg
        pg=(gam-1.0)*ug  #clean # use clean to keep pg low and Tgas will have floor like below  # no, need to use what was in simulation to be consistent with simulation's idea of what optical depth was
        # and of used ugclean above, then in funnel temperature would be very small and kappaff would be huge.
        #
        prad=(4.0/3.0-1.0)*urad
        # code Tgas for ideal gas
        Tgas=pg/rho
        # use rho to keep kappa low.
        KAPPAUSER=(rho*KAPPA*KAPPA_FF_CODE(rho,Tgas+TEMPMIN))
        KAPPAESUSER=(rho*KAPPAES*KAPPA_ES_CODE(rho,Tgas))

###################################
#
# Functions for computing quantities not found in fieldline files
#
###################################
def cvel():
    # MEMMARK: (4+4+4+1+1+4+4+4+4+6)=36 full 3D vars
    global ud,etad, etau, gamma, vu, vd, bu, bd, bsq,beta,betatot,betatoplot,Q1,Q2,Q2toplot,uradd,tauradintegrated,tauradeffintegrated
    #
    #
    ud = mdot(gv3,uu)                  #g_mn u^n
    etad = np.zeros_like(uu)
    etad[0] = -1/(-gn3[0,0])**0.5      #ZAMO frame velocity (definition)
    etau = mdot(gn3,etad)
    gamma=-mdot(uu,etad)                #Lorentz factor as measured by ZAMO
    vu = uu - gamma*etau               #u^m = v^m + gamma eta^m
    vd = mdot(gv3,vu)
    #
    bu=np.empty_like(uu)              #allocate memory for bu
    #set component per component
    bu[0]=mdot(B[1:4], ud[1:4])             #B^i u_i
    bu[1:4]=(B[1:4] + bu[0]*uu[1:4])/uu[0]  #b^i = (B^i + b^t u^i)/u^t
    bd=mdot(gv3,bu)
    bsq=mdot(bu,bd)
    #
    beta=((gam-1)*ug)/(1E-30 + bsq*0.5)
    betatot=((gam-1)*ug+(4.0/3.0-1.0)*urad)/(1E-30 + bsq*0.5)
    #betatoplot=np.ma.masked_array(beta,mask=(np.isnan(beta)==True)*(np.isinf(beta)==True)*(beta>1E10))
    betatoplot=np.copy(betatot)
    #betatoplot[(betatoplot<30)]=30.0
    #betatoplot[(betatoplot>1E3)]=1E3
    #betatoplot[np.isinf(betatoplot)]=1E3
    #betatoplot[np.isnan(betatoplot)]=1E3
    ##############################
    # compute some things one might plot
    #
    # disk mass density scale height
    #diskcondition=(beta>2.0)
    # was (bsq/rho<1.0)
    #diskcondition=diskcondition*(mum1fake<1.0)
    # just avoid floor mass
    #cond1=(bsq/rho<30)
    #cond2=(bsq/rho<10)
    #condmaxbsqorho=cond1*(r<9.0)+cond2*(r>=9.0)
    # need smooth change since notice small glitches in things with the above
    #rinterp=(r-9.0)*(1.0-0.0)/(0.0-9.0) # gives 0 for use near 9   gives 1 for use near 0
    #rinterp[rinterp>1.0]=1.0
    #rinterp[rinterp<0.0]=0.0
    #condmaxbsqorho=(bsq/rho < rinterp*30.0 + (1.0-rinterp)*10.0)
    diskcondition1=condmaxbsqorho
    diskcondition2=condmaxbsqorho
    # was denfactor=rho, but want uniform with corona and jet
    hoverr3dtoplot,thetamid3dtoplot=horcalc(hortype=1,which1=diskcondition1,which2=diskcondition2,denfactor=rholab)
    hoverr2dtoplot=hoverr3dtoplot.sum(2)/(nz)
    thetamid2dtoplot=thetamid3dtoplot.sum(2)/(nz)
    #
    #
    #
    Q1,Q3,OQ2=compute_resires(hoverrwhich=hoverr3dtoplot)
    Q2=hoverr3dtoplot/OQ2
    #Q2toplot=np.ma.masked_array(Q2,mask=(np.isnan(Q2)==True)*(np.isinf(Q2)==True)*(Q2>1E10))
    Q2toplot=np.copy(Q2)
    #Q2toplot[(Q2toplot<0)]=0.0
    #Q2toplot[(Q2toplot>1E3)]=1E3
    #Q2toplot[np.isinf(Q2toplot)]=1E3
    #Q2toplot[np.isnan(Q2toplot)]=1E3
    #
    aphi = fieldcalc()
    #
    uradd = mdot(gv3,uradu)                  #g_mn urad^n
    #
    # get tau's
    taurad1integrated,taurad1flipintegrated,taurad2integrated,taurad2flipintegrated,tauradintegrated,tauradeff1integrated,tauradeff1flipintegrated,tauradeff2integrated,tauradeff2flipintegrated,tauradeffintegrated=compute_taurad()

# even for average 2D data, clean-out floor already because some of the below involves multiplication or division that cannot be removed after averaging
# also, since region where bsqorho<maxbsqorho moves around, leakage occurs if only cutting based upon time-phi-averaged bsqorho
def getrhouclean(rho,ug,uu):
    #
    # default values
    maxbsqorhohigh=40
    maxbsqorhonear=30
    maxbsqorhofar=10
    maxuu0high=50
    #
    #
    len3rho=len(rho[0,0,:])
    #
    if len3rho==1 and nz>1:
        # assume 2D averages involved where last dimension is 1.  Need to keep consistent with how used on other things.
        print("rho wasn't nz in size, so assume 2D avg with only nz=1 effectively") ; sys.stdout.flush()
    #
    #
    # generic reformatting of r to fit nz changes (assumes r doesn't vary with phi)
    faker=np.zeros((nx,ny,len3rho),dtype=r.dtype)
    for kk in np.arange(0,len3rho):
        faker[:,:,kk] = r[:,:,0]
    #
    rinterp=(faker-9.0)*(1.0-0.0)/(0.0-9.0) # gives 0 for use near 9   gives 1 for use near 0
    rinterp[rinterp>1.0]=1.0
    rinterp[rinterp<0.0]=0.0
    #
    condmaxbsqorhorhs=rinterp*maxbsqorhonear + (1.0-rinterp)*maxbsqorhofar
    condmaxbsqorho=(bsq/rho < condmaxbsqorhorhs) # used as spatial conditional to replace single value of maxbsqorho
    #
    print(("t=%g" % (t)))
    print("r")
    print((r[:,ny//2,0]))
    print("condmaxbsqorhorhs along r")
    print((condmaxbsqorhorhs[:,ny//2,0]))
    print("condmaxbsqorho along eq")
    print((condmaxbsqorho[0,:,0]))
    #
    bsqorho=bsq/rho # want this to be using original rho
    #
    rhoclean = np.copy(rho)
    rhoclean[bsqorho>maxbsqorhonear]=1E-30
    rhoclean[condmaxbsqorho==0]=1E-30
    #
    ugclean = np.copy(ug)
    ugclean[bsqorho>maxbsqorhonear]=1E-30
    ugclean[condmaxbsqorho==0]=1E-30
    #
    douuclean=0 # choose (decided to keep original uu3 values even if floor messes them up, because otherwise have to modify uu3 at all radii in way that has \Omega_F constant along field lines.  Can't just modify near horizon since then energy flux not conserved.  Otherwise, worked well.)
    uuclean = np.copy(uu)
    #
    if douuclean==1:
        #
        #
        #############
        # force omegaf to be close to stationary/axisymmetric solution where floor dominates
        rh = 1+(1-a**2)**0.5
        omegah=a/(2.0*rh)
        #
        # only correct uu3 if close to limit of bsqorho where sensitive uu3 can be messed-up by floor still -- like near the pole
        # relevant for very close to horizon
        # do this at each time, because doing it after the fact (e.g. in SM) after averaging leads to artifacts and questionable cut-off values for bsqorho
        #
        # assumes dxdxp already defined
        newdxdxp33=dxdxp[3,3]
        if(len(dxdxp[3,3][0,0,:])!=nz):
            newdxdxp33=np.zeros((nx,ny,nz),dtype=rho.dtype)
            for kk in np.arange(0,nz):
                newdxdxp33[:,:,kk]=dxdxp[3,3][:,:,0]
        #
        # DEBUG:
        #print("shapes") ; sys.stdout.flush()
        #print(rho.shape) ; sys.stdout.flush()
        #print(bsq.shape) ; sys.stdout.flush()
        #print(bsqorho.shape) ; sys.stdout.flush()
        #print(uu.shape) ; sys.stdout.flush()
        #print(newdxdxp33.shape) ; sys.stdout.flush()
        #
        # setup parabolic \Omega_F
        # Interpolate to parabola form so closer to what real solution would allow so less likely to go outside light cone (Check that somehow)
        # This is valid in sense that solution inside pure magnetosphere agrees with para solution for \Omega_F, far from monopole.  So assume rest of magnetosphere near limits of numerical validity and near floor is close to para.
        condhupper=(h>np.pi*0.5)
        hbz=np.copy(h)
        hbz[condhupper]=(np.pi-h[condhupper])
        omegafpara=(0.25*np.sin(hbz)**2*(1+np.log(1+np.cos(hbz))))/(4.0*np.log(2.0)+np.sin(hbz)**2+(np.sin(hbz)**2-2*(1.0+np.cos(hbz)))*np.log(1.0+np.cos(hbz)))
        omegafohpara=omegafpara/(1.0/(2.0*2.0))
        #
        typeomegaf=1 # 0=mono 1=para
        if typeomegaf==0:
            newomegaf=omegah*0.5/newdxdxp33
        else:
            newomegaf=omegafohpara*omegah/newdxdxp33
        #
        # RHS restriction is just to match arrays
        cond=np.fabs(B[1]*np.sqrt(gv3[1,1]))<np.fabs(B[2]*np.sqrt(gv3[2,2]))
        numcond=np.sum(cond)
        cond2=(bsqorho>maxbsqorhohigh)
        numcond2=np.sum(cond2)
        print(("numcond=%d numcond2=%d" % (numcond,numcond2))); sys.stdout.flush()
        #
        maxiter=5
        for iter in np.arange(0,maxiter):
            #
            # define new uu3 assuming stationary-axisymmetric flow
            newuu3a=newomegaf*uuclean[0] + (uuclean[1]/B[1])*B[3]
            newuu3b=newomegaf*uuclean[0] + (uuclean[2]/B[2])*B[3]
            newuu3=newuu3a
            newuu3[cond]=newuu3b[cond]
            #
            # use cleaned-up uu3 in high bsqorho region only
            uuclean[3][cond2]=newuu3[cond2]
            #
            # fix uu0 so u.u=-1 is enforced.  Not doing this leads to completely wrong solutions for (e.g.) T^r_t and T^r_\phi
            # Can't change uu[3] without changing uu[0] for consistency.
            if 1==0:
                # fix using 3-vel (changes u^r too much so T^r_t and T^r_\phi change too much)
                # Also, can't just convert to 3-vel since may go outside light cone even if that's what's required to choose that omegaf.
                vunew = np.zeros_like(uuclean)
                vunew[0]=0
                vunew[1]=uuclean[1]/uuclean[0]
                vunew[2]=uuclean[2]/uuclean[0]
                vunew[3]=uuclean[3]/uuclean[0]
                #
                vdnew = mdot(gv3,vunew) ; vdnew[0]=0
                vsq = mdot(vdnew,vunew)
                uu0sqnew=1.0/ ( -gv3[0,0] - vsq - 2.0*mdot(vunew,gv3[0,:]) )
                condbaduu0=(uu0sqnew<1E-20)
                condbaduu0=condbaduu0+(uu0sqnew>maxuu0high**2)
                numcondbaduu0=np.sum(condbaduu0>0)
                print(("iter=%d numcondbaduu0=%d pickuu0sqnew=%g\n",iter,numcondbaduu0,uu0sqnew[condbaduu0])) ; sys.stdout.flush()
                uu0new = np.sqrt(uu0sqnew)
                # set new clean solution, but don't fix if new uu0 became bad (i.e. u^t<0) even if satisfies bsqorho condition
                uuclean[0][condbaduu0==0]=uu0new[condbaduu0==0]
                uuclean[1][condbaduu0==0]=vunew[1][condbaduu0==0]*uu0new[condbaduu0==0]
                uuclean[2][condbaduu0==0]=vunew[2][condbaduu0==0]*uu0new[condbaduu0==0]
                uuclean[3][condbaduu0==0]=vunew[3][condbaduu0==0]*uu0new[condbaduu0==0]
            else:
                # fix using 4-vel (ensures u^r is the same, so that b^2 u^r u_t is as close as possible to same, so that T^r_t and T^r_\phi are close to original
                uunew = np.zeros_like(uuclean)
                uunew[1]=uu[1] # remains the same!
                uunew[2]=uu[2] # remains the same!
                uunew[3]=uuclean[3] # set using new v^3 but with old u^t
                #
                # get uunew[0]:
                AA=gv3[0,0]
                tempuunew=np.copy(uunew)
                tempuunew[0]=0*tempuunew[0]
                BB=2.0*mdot(tempuunew,gv3[0,:])
                tempudnew = mdot(gv3,tempuunew) ; tempudnew[0]=0
                usq=mdot(tempudnew,tempuunew)
                CC=1.0 + usq
                disc=BB**2 - 4.0*AA*CC
                uu0newa=(-BB + np.sqrt(disc))/(2.0*AA)
                uu0newb=(-BB - np.sqrt(disc))/(2.0*AA)
                uunew[0]=np.copy(uuclean[0])
                #
                # avoid changing if makes it a bad solution
                condbaduu0=(disc<0.0)
                condbaduu0=condbaduu0+(uu0newa<1E-10)
                condbaduu0=condbaduu0+(uu0newa>maxuu0high)
                numcondbaduu0=np.sum(condbaduu0>0)
                #
                pickeduu0new=np.copy(uuclean[0])
                # pick b solution if actually closer uu0 to original uu0 value
                condpickuu0=(np.fabs(uu0newb-pickeduu0new)<np.fabs(uu0newa-pickeduu0new))
                sumcondpickuu0=np.sum(condpickuu0)
                pickeduu0new[condpickuu0]=uu0newb[condpickuu0]
                #
                # avoid final assignment if bad uu0 solution
                uunew[0][condbaduu0==0]=pickeduu0new[condbaduu0==0]
                #
                print(("iter=%d numcondbaduu0=%d sumcondpickuu0=%d uu0newabad=%g uu0newbbad=%g\n",iter,numcondbaduu0,sumcondpickuu0,uu0newa[condbaduu0],uu0newb[condbaduu0])) ; sys.stdout.flush()
                #
                uuclean[0]=uunew[0]
                uuclean[1]=uunew[1]
                uuclean[2]=uunew[2]
                uuclean[3]=uunew[3]
            #
            #
            # but now vu3 won't be quite right, so have to iterate a few times so uu0 converges with desired vu3
            cond2pick=cond2[5,:,0]
            uu0pick=uuclean[0][5,:,0][cond2pick]
            print(("iter=%d uuclean[0]=%g\n",iter,uu0pick)) ; sys.stdout.flush()
        #
        #udnew = mdot(gv3,uunew)
        #
        #
        #
        # test result
        omegaftest=uuclean[3]/uu[0] - (uu[1]/uu[0]/B[1])*B[3]
        omegatest=omegaftest*newdxdxp33
        omegarattest=omegatest[5,:,0]/omegah
        chosenomegaf=newomegaf[5,:,0]*newdxdxp33[5,:,0]/omegah
        print(("chosenomegaf=%g omegarattest=%g ",chosenomegaf,omegarattest)) ; sys.stdout.flush()
    #
    return(rhoclean,ugclean,uuclean,maxbsqorhonear,maxbsqorhofar,condmaxbsqorho,condmaxbsqorhorhs,rinterp)

def compute_resires(hoverrwhich=None):
    #
    # Note:  Toroidal field case: A_\theta -> A_1 A_2 A_3 via idxdxp's and then B1,B2,B3 computed via differences of A_i.
    # So even if only setting A_\theta implies B^\theta = 0, Btheta = dx^\theta/dxp^i B^i will give non-zero value due to truncation error.
    #
    mydH = r*dxdxp[2][2]*_dx2  # GODMARK: inaccurate a bit as \theta component because dxdxp[1][2] and dxdxp[2][1] are non-zero.  So say so in paper.
    #mydH = r*(_dx1*dxdxp[2][1] + _dx2*dxdxp[2][2]) # GODMARK: Seems logical, but wrong.
    mydP = r*np.sin(h)*dxdxp[3][3]*_dx3
    #
    #
    ################ IF USE OMEGAFIX==1, should use Kep Omega here
    #
    # BELOW FOR OMEGAFIX==0
    #omega = np.fabs(dxdxp[3][3]*uu[3]/uu[0])+1.0e-15
    # much of thick disk remains sub-Keplerian, so for estimate of Q must force consistency with assumptions of the Qmri measure
    # BELOW FOR OMEGAFIX==1
    # GODMARK: Leaving as OMEGAFIX==1 because want to average omega and vau2 separately (avoids stupid limit where omaga=0 could be true randomly and kill result's meaning)
    #R = r*np.sin(h)
    #omega = 1.0/(a + R**(3.0/2.0))
    #
    omega=1.0 + r*0.0 # now no longer using OMEGAFIX (i.e. OMEGAFIX=0 should be set)
    #
    #####################################
    #
    # don't use 0==1 part anymore (since can't readily compute res2 consistently)
    if 0==1:
        vau2 = np.abs(bu[2])/np.sqrt(rho+bsq+gam*ug)
        lambdamriu2 = 2*np.pi * vau2 / omega
        res=np.fabs(lambdamriu2/_dx2)
        res2=0
    #
    # distinguish between b^2 and b^\theta since off of equator they are different than if b^\theta=0 want that to be captured in Qmri measures.
    bu0ks=bu[0]*dxdxp[0][0]
    bu1ks=bu[1]*dxdxp[1][1] + bu[2]*dxdxp[1][2]
    bu2ks=bu[1]*dxdxp[2][1] + bu[2]*dxdxp[2][2]
    bu3ks=bu[3]*dxdxp[3][3]
    #
    # inverse of dx^{ks}/dx^{mks}
    idxdxp00=1/dxdxp[0][0]
    idxdxp11=dxdxp[2][2]/(dxdxp[2][2]*dxdxp[1][1]-dxdxp[2][1]*dxdxp[1][2])
    idxdxp12=dxdxp[1][2]/(dxdxp[2][1]*dxdxp[1][2]-dxdxp[2][2]*dxdxp[1][1])
    idxdxp21=dxdxp[2][1]/(dxdxp[2][1]*dxdxp[1][2]-dxdxp[2][2]*dxdxp[1][1])
    idxdxp22=dxdxp[1][1]/(dxdxp[2][2]*dxdxp[1][1]-dxdxp[2][1]*dxdxp[1][2])
    idxdxp33=1/dxdxp[3][3]
    #
    bd0ks=bd[0]*idxdxp00
    bd1ks=bd[1]*idxdxp11+bd[2]*idxdxp21
    bd2ks=bd[1]*idxdxp12+bd[2]*idxdxp22
    bd3ks=bd[3]*idxdxp33
    #
    #
    if 1==1:
        #bsqvert=bu[2]*bd[2]
        bsqvert=bu2ks*bd2ks
        va2sq = np.fabs(bsqvert/(rho+bsq+gam*ug))
        lambda2 = 2.0*np.pi * np.sqrt(va2sq) / omega
        # vertical grid cells per MRI wavelength
        res=np.fabs(lambda2/mydH)
        #
        #bsqphidir=bu[3]*bd[3]
        bsqphidir=bu3ks*bd3ks
        va3sq = np.fabs(bsqphidir/(rho+bsq+gam*ug))
        lambda3 = 2.0*np.pi * np.sqrt(va3sq) / omega
        # azimuthal grid cells per MRI wavelength
        res3=np.fabs(lambda3/mydP)
        #
        # MRI wavelengths over the whole disk
        if hoverrwhich is not None:
            #ires2=np.fabs(lambda2)
            ires2=np.fabs(lambda2*divideavoidinf(r*(2.0*hoverrwhich)))
            #sumhoverrwhich=np.sum(hoverrwhich)
            #print("sumhoverrwhich")
            #print(sumhoverrwhich)
            #sumires2=np.sum(ires2)
            #print("sumires2")
            #print(sumires2)
            #ires2=np.fabs(lambda2/(r*(2.0*0.2)))
            ires2[np.fabs(hoverrwhich)<10^(-10)]=0
            ires2[hoverrwhich!=hoverrwhich]=0
            ires2[np.isnan(hoverrwhich)==1]=0
            ires2[np.isinf(hoverrwhich)==1]=0
        else:
            # h/r=1 set, so can use <h/r>_t later so more stable result
            ires2=np.fabs(lambda2*divideavoidinf(r*(2.0*1.0)))
        #
    return(res,res3,ires2)

def fieldcalc(gdetB1=None):
    """
    Computes the field vector potential
    """
    return(fieldcalcU(gdetB1))

def fieldcalcU(gdetB1=None):
    # 3D wacky in time-dep flow
    aphi=fieldcalcU2D(gdetB1=gdetB1)
    return(aphi)

# pure 2D version
def fieldcalcU2D(gdetB1=None):
    """
    Computes cell-centered vector potential
    """
    aphi=fieldcalcface(gdetB1)
    #center it properly in theta
    aphi[:,0:ny-1]=0.5*(aphi[:,0:ny-1]+aphi[:,1:ny])
    #special treatment for last cell since no cell at j = ny, and we know aphi[:,ny] should vanish
    aphi[:,ny-1] *= 0.5
    #and in r
    aphi[0:nx-1] = 0.5*(aphi[0:nx-1]  +aphi[1:nx])
    aphi/=(nz*_dx3)
    return(aphi)

def fieldcalcface(gdetB1=None):
    """
    Computes the field vector potential
    """
    global aphi
    if gdetB1 == None:
        gdetB1 = gdetB[1]
    #average in phi and add up
    daphi = (gdetB1).sum(-1)[:,:,None]/nz*_dx2
    aphi = np.zeros_like(daphi)
    aphi[:,1:ny//2+1]=(daphi.cumsum(axis=1))[:,0:ny//2]
    #sum up from the other pole
    aphi[:,ny//2+1:ny]=(-daphi[:,::-1].cumsum(axis=1))[:,::-1][:,ny//2+1:ny]
    return(aphi)

# compute integrated optical depth
def compute_taurad(domergeangles=True,radiussettau1zero=80):
        # uses uu[], KAPPAUSER, KAPPAESUSER, gv3, r
        #
        thetavel=0.0
        betasq=1.0-1.0/(uu[0]**2)
        betasq[betasq<0.0]=0.0
        betavel=np.sqrt(betasq)
        gamfactor=uu[0]*(1.0-betavel*np.cos(thetavel))
        #drco=_dx1*np.sqrt(np.fabs(gv3[1,1]))/(2.0*uu[0])
        drco=_dx1*np.sqrt(np.fabs(gv3[1,1]))*gamfactor
        dhco=_dx2*np.sqrt(np.fabs(gv3[2,2]))*uu[0]
        dphco=_dx3*np.sqrt(np.fabs(gv3[3,3]))*uu[0]
        #
        taurad1=(KAPPAUSER+KAPPAESUSER)*drco
        # http://arxiv.org/pdf/astro-ph/0408590.pdf equation~3
        tauradeff1=np.sqrt(3.0*KAPPAUSER*(KAPPAUSER+KAPPAESUSER))*drco
        #
        # FREE PARAMETER:
        #radiussettau1zero=80
        #
        ############# taurad1
        taurad1[r[:,0,0]>radiussettau1zero,:,:]=0 # to get rid of parts of flow that aren't in steady-state and wouldn't have contributed
        #np.set_printoptions(threshold=sys.maxint)
        #print("taurad1") ; sys.stdout.flush()
        #print(taurad1[:,0,0]) ; sys.stdout.flush()
        #print("r") ; sys.stdout.flush()
        #print(r[:,0,0]) ; sys.stdout.flush()
        ########################### taurad1 (i.e. from small radius)
        taurad1integrated=np.cumsum(taurad1,axis=0)
        #print("taurad1integrated") ; sys.stdout.flush()
        #print(taurad1integrated[:,0,0]) ; sys.stdout.flush()
        #
        ########################### taurad1flip (i.e. from large radius)
        taurad1flip=taurad1[::-1,:,:]
        taurad1flipintegrated=np.cumsum(taurad1flip,axis=0)
        taurad1flipintegrated=taurad1flipintegrated[::-1,:,:]
        #print("taurad1flipintegrated") ; sys.stdout.flush()
        #print(taurad1flipintegrated[:,0,0]) ; sys.stdout.flush()
        #
        ############# tauradeff1
        tauradeff1[r[:,0,0]>radiussettau1zero,:,:]=0 # to get rid of parts of flow that aren't in steady-state and wouldn't have contributed
        np.set_printoptions(threshold=sys.maxsize)
        #print("tauradeff1") ; sys.stdout.flush()
        #print(tauradeff1[:,0,0]) ; sys.stdout.flush()
        #print("r") ; sys.stdout.flush()
        #print(r[:,0,0]) ; sys.stdout.flush()
        ########################### tauradeff1 (i.e. from small radius)
        tauradeff1integrated=np.cumsum(tauradeff1,axis=0)
        #print("tauradeff1integrated") ; sys.stdout.flush()
        #print(tauradeff1integrated[:,0,0]) ; sys.stdout.flush()
        #
        ########################### tauradeff1flip (i.e. from large radius)
        tauradeff1flip=tauradeff1[::-1,:,:]
        tauradeff1flipintegrated=np.cumsum(tauradeff1flip,axis=0)
        tauradeff1flipintegrated=tauradeff1flipintegrated[::-1,:,:]
        #print("tauradeff1flipintegrated") ; sys.stdout.flush()
        #print(tauradeff1flipintegrated[:,0,0]) ; sys.stdout.flush()
        #
        ########################### taurad2 (from theta=0 pole)
        taurad2=(KAPPAUSER+KAPPAESUSER)*dhco
        taurad2integrated=np.cumsum(taurad2,axis=1)
#        for kk in np.arange(0,nz):
#                for ii in np.arange(0,nx):
#        for jj in np.arange(ny/2,ny):
#            taurad2integrated[:,jj,:]=0 #taurad2integrated[:,ny/2,:]
        #
        ########################### taurad2flip (from theta=pi pole)
        taurad2flip=taurad2[:,::-1,:]
        taurad2flipintegrated=np.cumsum(taurad2flip,axis=1)
        taurad2flipintegrated=taurad2flipintegrated[:,::-1,:]
        if domergeangles==True:
            ########################### merge taurad2's
            for jj in np.arange(0,ny//2):
                taurad2flipintegrated[:,jj,:]=taurad2integrated[:,jj,:]
            for jj in np.arange(ny//2,ny):
                taurad2integrated[:,jj,:]=taurad2flipintegrated[:,jj,:]
        #
        ########################### tauradeff2 (from theta=0 pole)
        tauradeff2=np.sqrt(KAPPAUSER*(KAPPAUSER+KAPPAESUSER))*dhco
        tauradeff2integrated=np.cumsum(tauradeff2,axis=1)
#        for kk in np.arange(0,nz):
#                for ii in np.arange(0,nx):
#        for jj in np.arange(ny/2,ny):
#            tauradeff2integrated[:,jj,:]=0 #tauradeff2integrated[:,ny/2,:]
        #
        ########################### tauradeff2flip (from theta=pi pole)
        tauradeff2flip=tauradeff2[:,::-1,:]
        tauradeff2flipintegrated=np.cumsum(tauradeff2flip,axis=1)
        tauradeff2flipintegrated=tauradeff2flipintegrated[:,::-1,:]
        ########################### merge tauradeff2's
        if domergeangles==True:
            for jj in np.arange(0,ny//2):
                tauradeff2flipintegrated[:,jj,:]=tauradeff2integrated[:,jj,:]
            for jj in np.arange(ny//2,ny):
                tauradeff2integrated[:,jj,:]=tauradeff2flipintegrated[:,jj,:]
        #
        ########################### taurad3
        taurad3=uu[0]*(KAPPAUSER+KAPPAESUSER)*dphco
        ########################### tauradeff3
        tauradeff3=uu[0]*np.sqrt(KAPPAUSER*(KAPPAUSER+KAPPAESUSER))*dphco
        #
        # so tauradintegrated (final version) is optical depth integrated from large radii and away from pole.
        tauradintegrated=np.maximum(taurad1flipintegrated,taurad2integrated)
        tauradeffintegrated=np.maximum(tauradeff1flipintegrated,tauradeff2integrated)
        #
        return(taurad1integrated,taurad1flipintegrated,taurad2integrated,taurad2flipintegrated,tauradintegrated,tauradeff1integrated,tauradeff1flipintegrated,tauradeff2integrated,tauradeff2flipintegrated,tauradeffintegrated)
        # use primarily: taurad1flipintegrated taurad2integrated and can use and'ed version

###################################
#
# Functions dealing with the avg2d.npy files
#
###################################
def loadavg():
    #
    avgmem = get2davg(usedefault=1,domerge=False)
    assignavg2dvars(avgmem)

def get2davg(usedefault=0,whichgroup=-1,whichgroups=-1,whichgroupe=-1,itemspergroup=20,domerge=False):
    if whichgroup >= 0:
        whichgroups = whichgroup
        whichgroupe = whichgroupe + 1
    elif whichgroupe < 0:
        whichgroupe = whichgroups + 1
    #check values for sanity
    if usedefault == 0 and (whichgroups < 0 or whichgroupe < 0 or whichgroups >= whichgroupe or itemspergroup <= 0):
        print(( "get2davg: whichgroups = %d, whichgroupe = %d, itemspergroup = %d not allowed"
               % (whichgroups, whichgroupe, itemspergroup) ));sys.stdout.flush()
        sys.stdout.flush()
        return None
    #
    if usedefault:
        fname = "avg2d.npy"
    else:
        fname = "avg2d%02d_%04d_%04d.npy" % (itemspergroup, whichgroups, whichgroupe)
    #
    print(("gdet2davg(): checking for fname=%s" % (fname))) ; sys.stdout.flush()
    #
    if os.path.isfile( fname ):
        print(( "File %s exists, loading from file..." % fname ));sys.stdout.flush()
        avgtot=np.load( fname )
        return( avgtot )
    #####################
    defaultfti,defaultftf=getdefaulttimes()
    avgfti = defaultfti
    avgftf = defaultftf
    ######################
    print(("whichgroups=%d whichgroupe=%d" % (whichgroups,whichgroupe))) ; sys.stdout.flush()
    ######################
    n2avg = 0
    nitems = 0
    myrange = np.arange(whichgroups,whichgroupe)
    numrange = myrange.shape[0]
    firstavgone=1
    firstavgoneused=1
    # defaults for ts,tf
    ts=0
    tf=0
    localdenom=0.0
    #####################
    for (i,g) in enumerate(myrange):
        print(("i=%d g=%d" % (i,g))) ; sys.stdout.flush()
        avgone=get2davgone( whichgroup = g, itemspergroup = itemspergroup )
        if avgone.any() == None:
            continue
        if firstavgone==1:
            avgtot = np.zeros_like(avgone)
            firstavgone=0
        tstry=avgone[0,0,0]
        tftry=avgone[0,1,0]
        if itemspergroup>1:
            localdt=tftry-tstry
        else:
            # degenerate case, then time considered irrelevant and average does occurs directly
            localdt=1.0
        #
        # use avg data if either start or finish of time used for averaging is within average period
        didntuse=1
        #if numrange>1:
        if domerge==True:
            # then merging
            # only include if no inital dump (in case small set of files happens to have tf late but ti=0)
            # or include if final dump non-zero (in case small set and includes ti=0 but tf late)
            if (tftry!=0 and tstry!=0) or (tftry!=0):
                if (tstry>avgfti and tstry<avgftf) or ((tftry>avgfti and tftry<avgftf)) or ((tstry<avgfti and tftry>avgftf)):
                    if firstavgoneused==1:
                        ts=avgone[0,0,0]
                        firstavgoneused=0
                    tf=avgone[0,1,0]
                    #
                    localdenom+=localdt
                    #
                    avgtot += avgone*localdt
                    nitems += avgone[0,2,0]
                    n2avg += 1
                    print(("USING: During merge: ts=%g tf=%g tstry=%g tftry=%g n2avg=%d" % (ts,tf,tstry,tftry,n2avg)));sys.stdout.flush()
                    didntuse=0
                    #
            # end if within averaging period of time
            if didntuse==1:
                print(("NOTUSING: During merge: tstry=%g tftry=%g n2avg=%d avgfti=%g avgftf=%g" % (tstry,tftry,n2avg,avgfti,avgftf)));sys.stdout.flush()
        else:
            # then not merging, normal assignments no matter what the time
            ts=avgone[0,0,0]
            tf=avgone[0,1,0]
            #
            localdenom+=localdt
            #
            avgtot += avgone*localdt
            nitems += avgone[0,2,0]
            n2avg += 1
            print(("During NONmerge: ts=%g tf=%g tstry=%g tftry=%g n2avg=%d" % (ts,tf,tstry,tftry,n2avg)));sys.stdout.flush()
    # end for loop over whichgroups->whichgroupe
    #
    # set final avg file times and number of items
    avgtot[0,0,0] = ts
    avgtot[0,1,0] = tf
    avgtot[0,2,0] = nitems
    print(("Final avg_ts=%g avg_te=%g nitems=%d localdenom=%g" % (avgtot[0,0,0],avgtot[0,1,0],avgtot[0,2,0],localdenom)));sys.stdout.flush()
    #get the average
    if n2avg == 0:
        print( "0 total files, so no data generated." ) ; sys.stdout.flush()
        return( None )
    #avoid renormalizing the header
    #avgtot[1:] /= n2avg
    avgtot[1:] /= (localdenom)
    #
    print("avgtot[3] for avg_bsq in merge") ; sys.stdout.flush()
    print((avgtot[3])) ; sys.stdout.flush()
    #
    #only save if more than 1 dump was to be processed (indicating merge)
    if numrange > 1:
        print(( "Saving data to file: n2avg=%d ..." % (n2avg) ));sys.stdout.flush()
        np.save( fname, avgtot )
    return( avgtot )

def assignavg2dvars(avgmem):
    # MEMMARK: currently 281 2D vars, so not much memory.
    global avg_ts,avg_te,avg_nitems
    global avg_rho,avg_ug,avg_bsq,avg_unb,avg_urad
    global avg_uu,avg_bu,avg_ud,avg_bd,avg_B,avg_gdetB,avg_omegaf2,avg_omegaf2b,avg_omegaf1,avg_omegaf1b,avg_rhouu,avg_rhobu,avg_rhoud,avg_rhobd
    global avg_absuu,avg_absbu,avg_absud,avg_absbd
    global avg_absomegaf2,avg_absomegaf2b,avg_absomegaf1,avg_absomegaf1b
    global avg_absrhouu,avg_absfdd
    global avg_uguu,avg_ugud,avg_Tud,avg_fdd,avg_rhouuud,avg_uguuud,avg_bsquuud,avg_bubd,avg_uuud
    global avg_TudEM, avg_TudMA, avg_mu, avg_sigma, avg_bsqorho, avg_absB, avg_absgdetB, avg_psisq
    global avg_TudPA, avg_TudEN, avg_TudRAD
    global avg_gamma,avg_pg,avg_pb,avg_beta,avg_betatot
    global avg_KAPPAUSER,avg_KAPPAESUSER,avg_tauradintegrated,avg_tauradeffintegrated
    global avg_vpot,avg_vpot2
    #avg defs
    i=0
    # 1
    # uses fake 2D space for some single numbers
    avg_ts=avgmem[i,0,:];
    avg_te=avgmem[i,1,:];
    print(( "assignavg2dvars: avg_ts=%d avg_te=%d" % (avg_ts[0],avg_te[0]))) ; sys.stdout.flush()
    avg_nitems=avgmem[i,2,:]
    i+=1
    #quantities
    # 4
    avg_rho=avgmem[i,:,:,None];i+=1 # i=1
    avg_ug=avgmem[i,:,:,None];i+=1  # i=2
    avg_urad=avgmem[i,:,:,None];i+=1  # i=3
    avg_bsq=avgmem[i,:,:,None];i+=1 # i=4
    avg_unb=avgmem[i,:,:,None];i+=1
    # 4*4=16
    n=4
    avg_uu=avgmem[i:i+n,:,:,None];i+=n
    avg_bu=avgmem[i:i+n,:,:,None];i+=n
    avg_ud=avgmem[i:i+n,:,:,None];i+=n
    avg_bd=avgmem[i:i+n,:,:,None];i+=n
    # 4*4=16
    n=4
    avg_absuu=avgmem[i:i+n,:,:,None];i+=n
    avg_absbu=avgmem[i:i+n,:,:,None];i+=n
    avg_absud=avgmem[i:i+n,:,:,None];i+=n
    avg_absbd=avgmem[i:i+n,:,:,None];i+=n
    #cell-centered magnetic field components
    # 3*2=6
    n=3;
    avg_B=avgmem[i:i+n,:,:,None];i+=n
    avg_gdetB=avgmem[i:i+n,:,:,None];i+=n
    # 3*2=6
    n=3
    avg_absB=avgmem[i:i+n,:,:,None];i+=n
    avg_absgdetB=avgmem[i:i+n,:,:,None];i+=n
    # 4
    avg_omegaf2=avgmem[i,:,:,None];i+=1
    avg_omegaf2b=avgmem[i,:,:,None];i+=1
    avg_omegaf1=avgmem[i,:,:,None];i+=1
    avg_omegaf1b=avgmem[i,:,:,None];i+=1
    #
    # 4
    avg_absomegaf2=avgmem[i,:,:,None];i+=1
    avg_absomegaf2b=avgmem[i,:,:,None];i+=1
    avg_absomegaf1=avgmem[i,:,:,None];i+=1
    avg_absomegaf1b=avgmem[i,:,:,None];i+=1
    #
    # 6*4=24
    n=4
    avg_rhouu=avgmem[i:i+n,:,:,None];i+=n
    avg_rhobu=avgmem[i:i+n,:,:,None];i+=n
    avg_rhoud=avgmem[i:i+n,:,:,None];i+=n
    avg_rhobd=avgmem[i:i+n,:,:,None];i+=n
    avg_uguu=avgmem[i:i+n,:,:,None];i+=n
    avg_ugud=avgmem[i:i+n,:,:,None];i+=n
    #
    # 1*4=4
    n=4
    avg_absrhouu=avgmem[i:i+n,:,:,None];i+=n
    #
    # 2*16=32
    n=16
    #energy fluxes and faraday
    avg_Tud=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    avg_fdd=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # 1*16=16
    n=16
    # faraday
    avg_absfdd=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # 5*16=80
    # part1: rho u^m u_l
    avg_rhouuud=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # part2: u u^m u_l
    avg_uguuud=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # part3: b^2 u^m u_l
    avg_bsquuud=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # part6: b^m b_l
    avg_bubd=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # u^m u_l
    #print( "i = %d, avgmem.shape[0] = %d " % (i, avgmem.shape[0]) )
    #sys.stdout.flush()
    avg_uuud=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # 2*16=32
    n=16
    #EM/MA
    avg_TudEM=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    avg_TudMA=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # 2*16=32
    n=16
    #P/IE
    avg_TudPA=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    avg_TudEN=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    avg_TudRAD=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # 3
    #mu,sigma
    n=1
    avg_mu=avgmem[i,:,:,None];i+=n
    avg_sigma=avgmem[i,:,:,None];i+=n
    avg_bsqorho=avgmem[i,:,:,None];i+=n
    # 1
    n=1
    avg_psisq=avgmem[i,:,:,None];i+=n
    #
    avg_KAPPAUSER=avgmem[i,:,:,None];i+=1 # i=1
    avg_KAPPAESUSER=avgmem[i,:,:,None];i+=1 # i=1
    avg_tauradintegrated=avgmem[i,:,:,None];i+=1 # i=1
    avg_tauradeffintegrated=avgmem[i,:,:,None];i+=1 # i=1
    #
    avg_vpot = avgmem[i,:,:,None];i+=1 # i=1
    avg_vpot2 = avgmem[i,:,:,None];i+=1 # i=1
    #
    # number of full 2D quantities
    nqtyavg=i
    global navg
    navg=getnqtyavg()
    if nqtyavg!=navg:
        print(("nqtyavg=%d while navg=%d" % (nqtyavg,navg))) ; sys.stdout.flush()
    #
    #
    ##########################
    #derived quantities
    avg_gamma=avg_uu[0]/(-gn3[0,0])**0.5
    avg_pg=((gam-1)*avg_ug)
    avg_pb=avg_bsq*0.5
    avg_prad=(4.0/3.0-1.0)*avg_urad
    avg_beta=avg_pg/avg_pb # gas only, ignores radiation
    avg_betatot=(avg_pg+avg_prad)/avg_pb

def getdefaulttimes():
    #
    defaultfti=1
    defaultftf=1e6
    #
    #
    return defaultfti,defaultftf

def get2davgone(whichgroup=-1,itemspergroup=20):
    """
    """
    global avg_ts,avg_te,avg_nitems,avg_rho,avg_ug,avg_urad,avg_bsq,avg_unb,avg_uu,avg_bu,avg_ud,avg_bd,avg_B,avg_gdetB,avg_omegaf2,avg_omegaf2b,avg_omegaf1,avg_omegaf1b,avg_rhouu,avg_rhobu,avg_rhoud,avg_rhobd,avg_uguu,avg_ugud,avg_Tud,avg_fdd,avg_rhouuud,avg_uguuud,avg_bsquuud,avg_bubd,avg_uuud
    global avg_TudEM, avg_TudMA, avg_mu, avg_sigma, avg_bsqorho, avg_absB, avg_absgdetB, avg_psisq
    global avg_TudPA, avg_TudEN,avg_TudRAD
    global avg_absuu,avg_absbu,avg_absud,avg_absbd
    global avg_absomegaf2,avg_absomegaf2b,avg_absomegaf1,avg_absomegaf1b
    global avg_absrhouu,avg_absfdd
    global avg_KAPPAUSER,avg_KAPPAESUSER,avg_tauradintegrated,avg_tauradeffintegrated
    global firstfieldlinefile
    global avg_vpot,avg_vpot2
    #
    if whichgroup < 0 or itemspergroup <= 0:
        print(( "get2davgone: whichgroup = %d, itemspergroup = %d not allowed" % (whichgroup, itemspergroup) )) ; sys.stdout.flush()
        return None
    #
    fname = "avg2d%02d_%02d.npy" % (itemspergroup, whichgroup)
    if os.path.isfile( fname ):
        print(( "File %s exists, loading from file..." % fname )) ; sys.stdout.flush()
        avgmem=np.load( fname )
        return( avgmem )
    tiny=np.finfo(t.dtype).tiny
    flist = glob.glob( os.path.join("dumps/", "fieldline*.bin") )
    sort_nicely(flist)
    firstfieldlinefile=flist[0]
    #flist.sort()
    #
    #print "Number of time slices: %d" % flist.shape[0] ; sys.stdout.flush()
    #store 2D data
    global navg
    navg=getnqtyavg()
    avgmem=np.zeros((navg,nx,ny),dtype=np.float32)
    assignavg2dvars(avgmem)
    ##
    ######################################
    ##
    ## NEED TO ADD vmin/vmax VELOCITY COMPONENTS
    ##
    ######################################
    ##
    #
    #########################################
    # get times for files to use
    itert=0
    for fldindex, fldname in enumerate(flist):
        if( whichgroup >=0 and itemspergroup > 0 ):
            if( fldindex / itemspergroup != whichgroup ):
                continue
        #
        print(( "Reading " + fldname + " ..." )) ; sys.stdout.flush()
        #rfdheaderonly("../"+fldname)
        rfdheaderonly(fldname)
        #
        if itert==0:
            # create array with 1 element
            localts=np.arange(0,1)
        else:
            # add an element
            localts=np.append(localts,np.arange(0,1))
        #
        localts[itert]=t
        itert=itert+1
    #
    if len(localts)>1:
        localdt=np.gradient(localts)
    else:
        # dummy dt if only 1 data file per avg file
        print("Setting dummy localdt") ; sys.stdout.flush()
        localdt=np.arange(0,1)
        localdt=0.0*localdt + 1.0
    #
    #
    #
    #print "Total number of quantities: %d" % (i)
    print("Doing %d-th group of %d items" % (whichgroup, itemspergroup)) ; sys.stdout.flush()
    #end avg defs
    itert=0
    for fldindex, fldname in enumerate(flist):
        if( whichgroup >=0 and itemspergroup > 0 ):
            if( fldindex / itemspergroup != whichgroup ):
                continue
        #
        print(( "Reading " + fldname + " ..." )) ; sys.stdout.flush()
        rfd("../"+fldname)
        #
        print(( "Computing get2davgone:" + fldname + " ..." )) ;  sys.stdout.flush()
        cvel()
        #
        #
        # which is *accepted* condition for non-jet part that's not floor contaminated.  So which==1 means ok to pass into averaging.  So use which==0 to clean-up things
        Tcalcud(maxbsqorho=maxbsqorhonear,which=condmaxbsqorho)
        faraday()
        #
        bsqo2rho = bsq/(2.0*rhounclean)
        bsqorho = bsq/(rhounclean)
        #
        #
        ##########################
        #if first item in group
        if fldindex == itemspergroup * whichgroup:
            avg_ts[0]=t
        #if last item in group
        # NO, number of files may not be evenly divisible by itemspergroup, so just always save te as when merging
        #if fldindex == itemspergroup * whichgroup + (itemspergroup - 1):
        avg_te[0]=t
        #
        # 1
        avg_nitems[0]+=1
        #
        ###################
        #quantities
        # 4
        avg_rho+=rhoclean.sum(-1)[:,:,None]*localdt[itert]
        avg_ug+=ugclean.sum(-1)[:,:,None]*localdt[itert]
        avg_urad+=urad.sum(-1)[:,:,None]*localdt[itert]
        avg_bsq+=bsq.sum(-1)[:,:,None]*localdt[itert]
        enth=1+ugclean*gam/rhoclean
        avg_unb+=(enth*ud[0]).sum(-1)[:,:,None]*localdt[itert]
        # 16
        avg_uu+=uu.sum(-1)[:,:,:,None]*localdt[itert]
        avg_bu+=bu.sum(-1)[:,:,:,None]*localdt[itert]
        avg_ud+=ud.sum(-1)[:,:,:,None]*localdt[itert]
        avg_bd+=bd.sum(-1)[:,:,:,None]*localdt[itert]
        # 16
        avg_absuu+=(np.fabs(uu)).sum(-1)[:,:,:,None]*localdt[itert]
        avg_absbu+=(np.fabs(bu)).sum(-1)[:,:,:,None]*localdt[itert]
        avg_absud+=(np.fabs(ud)).sum(-1)[:,:,:,None]*localdt[itert]
        avg_absbd+=(np.fabs(bd)).sum(-1)[:,:,:,None]*localdt[itert]
        #cell-centered magnetic field components
        # 3+3=6
        n=3;
        avg_B+=B[1:4].sum(-1)[:,:,:,None]*localdt[itert]
        avg_gdetB+=gdetB[1:4].sum(-1)[:,:,:,None]*localdt[itert]
        # 6
        n=3
        avg_absB += np.abs(B[1:4]).sum(-1)[:,:,:,None]*localdt[itert]
        avg_absgdetB += np.abs(gdetB[1:4]).sum(-1)[:,:,:,None]*localdt[itert]
        #
        # 4
        # omega has to be cleaned because of effects of floor near pole where effect on omegaf alot despite dynamically insignificance, but do it elsewhere
        avg_omegaf2+=omegaf2.sum(-1)[:,:,None]*localdt[itert]
        avg_omegaf2b+=omegaf2b.sum(-1)[:,:,None]*localdt[itert]
        avg_omegaf1+=omegaf1.sum(-1)[:,:,None]*localdt[itert]
        avg_omegaf1b+=omegaf1b.sum(-1)[:,:,None]*localdt[itert]
        # 4
        # omega has to be cleaned because of effects of floor near pole where effect on omegaf alot despite dynamically insignificance, but do it elsewhere
        avg_absomegaf2+=(np.fabs(omegaf2)).sum(-1)[:,:,None]*localdt[itert]
        avg_absomegaf2b+=(np.fabs(omegaf2b)).sum(-1)[:,:,None]*localdt[itert]
        avg_absomegaf1+=(np.fabs(omegaf1)).sum(-1)[:,:,None]*localdt[itert]
        avg_absomegaf1b+=(np.fabs(omegaf1b)).sum(-1)[:,:,None]*localdt[itert]
        #
        # 6*4=24
        n=4
        avg_rhouu+=(rhoclean*uu).sum(-1)[:,:,:,None]*localdt[itert]
        avg_rhobu+=(rhoclean*bu).sum(-1)[:,:,:,None]*localdt[itert]
        avg_rhoud+=(rhoclean*ud).sum(-1)[:,:,:,None]*localdt[itert]
        avg_rhobd+=(rhoclean*bd).sum(-1)[:,:,:,None]*localdt[itert]
        avg_uguu+=(ugclean*uu).sum(-1)[:,:,:,None]*localdt[itert]
        avg_ugud+=(ugclean*ud).sum(-1)[:,:,:,None]*localdt[itert]
        #
        # 1*4=4
        n=4
        avg_absrhouu+=(np.fabs(rhoclean*uu)).sum(-1)[:,:,:,None]*localdt[itert]
        #
        # 16*2=32
        n=16
        #energy fluxes and faraday
        avg_Tud+=Tud.sum(-1)[:,:,:,:,None]*localdt[itert]
        avg_fdd+=((fdd)).sum(-1)[:,:,:,:,None]*localdt[itert]
        #
        # 16*1=16
        n=16
        # faraday
        avg_absfdd+=(np.fabs(fdd)).sum(-1)[:,:,:,:,None]*localdt[itert] # take absolute value since oscillate around 0 near equator and would cancel out and give noise in fdd/fdd type calculations, such as for omegaf
        #
        #
        uuud=odot(uu,ud).sum(-1)[:,:,:,:,None]*localdt[itert]
        #
        # 16*5=80
        # part1: rho u^m u_l
        avg_rhouuud+=rhoclean.sum(-1)[:,:,None]*uuud*localdt[itert]
        # part2: u u^m u_l
        avg_uguuud+=ugclean.sum(-1)[:,:,None]*uuud*localdt[itert]
        # part3: b^2 u^m u_l
        avg_bsquuud+=bsq.sum(-1)[:,:,None]*uuud*localdt[itert]
        # part6: b^m b_l
        avg_bubd+=odot(bu,bd)[:,:,:,:,None].sum(-1)*localdt[itert]
        # u^m u_l
        avg_uuud+=uuud
        #
        # 16*2=32
        #EM/MA
        avg_TudEM+=TudEM.sum(-1)[:,:,:,:,None]*localdt[itert]
        avg_TudMA+=TudMA.sum(-1)[:,:,:,:,None]*localdt[itert]
        # 16*2=32
        #PA/IE (EM is B) -- for gammie plot
        avg_TudPA+=TudPA.sum(-1)[:,:,:,:,None]*localdt[itert]
        avg_TudEN+=TudEN.sum(-1)[:,:,:,:,None]*localdt[itert]
        avg_TudRAD+=TudRAD.sum(-1)[:,:,:,:,None]*localdt[itert]
        #
        # 3
        #mu,sigma
        avg_mu += (-Tud[1,0]/(rhoclean*uu[1])).sum(-1)[:,:,None]*localdt[itert]
        avg_sigma += (-TudEM[1,0]/TudMA[1,0]).sum(-1)[:,:,None]*localdt[itert]
        avg_bsqorho += (bsq/rhounclean).sum(-1)[:,:,None]*localdt[itert] # keep as unclean since want to know what bsqorho is
        #
        #
        avg_KAPPAUSER+=KAPPAUSER.sum(-1)[:,:,None]*localdt[itert]
        avg_KAPPAESUSER+=KAPPAESUSER.sum(-1)[:,:,None]*localdt[itert]
        avg_tauradintegrated+=tauradintegrated.sum(-1)[:,:,None]*localdt[itert]
        avg_tauradeffintegrated+=tauradeffintegrated.sum(-1)[:,:,None]*localdt[itert]
        # 1
        n=1
        aphi = fieldcalcface()
        avg_psisq += ((_dx3*aphi.sum(-1))**2)[:,:,None]*localdt[itert]
        #
        # 1 and 1
        n=1
        avg_vpot += (scaletofullwedge(nz*_dx3*fieldcalc(gdetB1=gdetB[0]))).sum(-1)[:,:,None]*localdt[itert]
        avg_vpot2 += (scaletofullwedge((nz*(_dx3*aphi)**2))**0.5).sum(-1)[:,:,None]*localdt[itert]
        #
        # iterate
        itert=itert+1
    #
    if avg_nitems[0] == 0:
        print( "No files found" ) ; sys.stdout.flush()
        return None
    #
    #
    denom=localdt.sum()
    #
    print(("denom=%g and localdt[0]=%g" % (denom,localdt[0]))) ; sys.stdout.flush()
    #
    print("avg_bsq when forming") ; sys.stdout.flush()
    print(avg_bsq) ; sys.stdout.flush()
    #
    ######################
    #divide all lines but the header line [which holds (ts,te,nitems)]
    #by the number of elements to get time averages
    #avgmem[1:]/=(np.float32(avg_nitems[0])*np.float32(nz))
    avgmem[1:]/=(np.float32(denom)*np.float32(nz))
    #
    print("avg_bsq when setting avgmem") ; sys.stdout.flush()
    print((avgmem[3])) ; sys.stdout.flush()
    #
    print( "Saving to file..." ) ; sys.stdout.flush()
    np.save( fname, avgmem )
    #
    print( "Done avgmem!" ) ; sys.stdout.flush()
    #
    return(avgmem)

def getnqtyavg():
    # was 206 before with Sasha's code, but apparently should have been 207
    #value=1 + 4 + 16 + 6 + 4 + 24 + 32 + 80 + 32 + 32 + 3 + 6 + 1
    # added&moved abs versions
    value=1 + 4 + 16*2 + 6*2 + 4*2 + 24+4 + 32+16 + 80 + 32 + 32 + 3 + 1
    value=value+16+4 # for TudRAD and KAPPAUSER and KAPPAESUSER and tauradintegrated and tauradeffintegrated
    value=value+1 # for urad
    value=value+2 # for avg_vpot and avg_vpot2
    return(value)

# allow to remove rho and ug component to remove floor effects
def Tcalcud(maxbsqorho=None, which=None):
    # MEMMARK: 16*5+5=85 full 3D vars.
    global Tud, TudEM, TudMA, TudPA, TudEN, TudRAD
    global mu, sigma
    global enth
    global unb, isunbound
    #
    bsqo2rho = bsq/(2.0*rho) # uses original rho, which is as desired
    bsqorho = bsq/(rho) # uses original rho, which is as desired
    #
    # below needed in case operating on averages rather than from rfd()
    #(rhoclean,ugclean,uu,maxbsqorhonear,maxbsqorhofar,condmaxbsqorho)=getrhouclean(rho,ug,uu)
    #ud = mdot(gv3,uu)                  #g_mn u^n
    # no, assume averages formed out of rfd reads that already include whatever cleaning wanted
    #
    #
    pg = (gam-1)*ugclean
    prad = (4.0/3.0-1)*urad
    w=rhoclean+ugclean+pg
    wnorhoclean=ugclean+pg
    eta=w+bsq
    Tud = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
    TudMA = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
    TudEM = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
    TudPA = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
    TudEN = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
    TudRAD = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
    for kapa in np.arange(4):
        for nu in np.arange(4):
            if(kapa==nu): delta = 1
            else: delta = 0
            TudEM[kapa,nu] = bsq*uu[kapa]*ud[nu] + 0.5*bsq*delta - bu[kapa]*bd[nu]
            TudMA[kapa,nu] = w*uu[kapa]*ud[nu]+pg*delta
            TudPA[kapa,nu] = rhoclean*uu[kapa]*ud[nu]
            TudEN[kapa,nu] = wnorhoclean*uu[kapa]*ud[nu]+pg*delta
            #Tud[kapa,nu] = eta*uu[kapa]*ud[nu]+(pg+0.5*bsq)*delta-bu[kapa]*bd[nu]
            TudRAD[kapa,nu] = (Erf/3.0)*(4.0*uradu[kapa]*uradd[nu]+delta)
            Tud[kapa,nu] = TudEM[kapa,nu] + TudMA[kapa,nu] + TudRAD[kapa,nu]
    #mu = -Tud[1,0]/(rhoclean*uu[1])
    mu = -Tud[1,0]*divideavoidinf(rhoclean*uu[1])
    sigma = TudEM[1,0]*divideavoidinf(TudMA[1,0])
    enth=1+ugclean*gam/rhoclean
    unb=enth*ud[0]
    # unbound here means *thermally* rather than kinetically (-u_t>1) or fully thermo-magnetically (\mu>1) unbound.
    isunbound=(-unb>1.0)
    #
    #
    print(("TudEM[1,0,5,0,0]=%g" % (TudEM[1,0,5,0,0]))) ; sys.stdout.flush()
    print(("bsq[5,0,0]=%g" % (bsq[5,0,0]))) ; sys.stdout.flush()
    print(("uu[1,5,0,0]=%g" % (uu[1,5,0,0]))) ; sys.stdout.flush()
    print(("ud[0,5,0,0]=%g" % (ud[0,5,0,0]))) ; sys.stdout.flush()
    print(("bu[1,5,0,0]=%g" % (bu[1,5,0,0]))) ; sys.stdout.flush()
    print(("bd[0,5,0,0]=%g" % (bd[0,5,0,0]))) ; sys.stdout.flush()
    #
    print(("uu[0,5,0,0]=%g" % (uu[0,5,0,0]))) ; sys.stdout.flush()
    print(("B[0,5,0,0]=%g" % (B[0,5,0,0]))) ; sys.stdout.flush()
    print(("uu[1,5,0,0]=%g" % (uu[1,5,0,0]))) ; sys.stdout.flush()
    print(("B[1,5,0,0]=%g" % (B[1,5,0,0]))) ; sys.stdout.flush()
    print(("uu[2,5,0,0]=%g" % (uu[2,5,0,0]))) ; sys.stdout.flush()
    print(("B[2,5,0,0]=%g" % (B[2,5,0,0]))) ; sys.stdout.flush()
    print(("uu[3,5,0,0]=%g" % (uu[3,5,0,0]))) ; sys.stdout.flush()
    print(("B[3,5,0,0]=%g" % (B[3,5,0,0]))) ; sys.stdout.flush()
    print(("udotB[5,0,0]=%g" % (ud[0,5,0,0]*B[0,5,0,0] + ud[1,5,0,0]*B[1,5,0,0] + ud[2,5,0,0]*B[2,5,0,0] + ud[3,5,0,0]*B[3,5,0,0]))) ; sys.stdout.flush()

def faraday():
    # MEMMARK: 32+4=36 full 3D vars
    global fdd, fuu, omegaf1, omegaf1b, omegaf2, omegaf2b
    # these are native values according to HARM
    fdd = np.zeros((4,4,nx,ny,nz),dtype=rho.dtype)
    #fdd[0,0]=0*gdet
    #fdd[1,1]=0*gdet
    #fdd[2,2]=0*gdet
    #fdd[3,3]=0*gdet
    fdd[0,1]=gdet*(uu[2]*bu[3]-uu[3]*bu[2]) # f_tr
    fdd[1,0]=-fdd[0,1]
    fdd[0,2]=gdet*(uu[3]*bu[1]-uu[1]*bu[3]) # f_th
    fdd[2,0]=-fdd[0,2]
    fdd[0,3]=gdet*(uu[1]*bu[2]-uu[2]*bu[1]) # f_tp
    fdd[3,0]=-fdd[0,3]
    fdd[1,3]=gdet*(uu[2]*bu[0]-uu[0]*bu[2]) # f_rp = gdet*B2
    fdd[3,1]=-fdd[1,3]
    fdd[2,3]=gdet*(uu[0]*bu[1]-uu[1]*bu[0]) # f_hp = gdet*B1
    fdd[3,2]=-fdd[2,3]
    fdd[1,2]=gdet*(uu[0]*bu[3]-uu[3]*bu[0]) # f_rh = gdet*B3
    fdd[2,1]=-fdd[1,2]
    #
    fuu = np.zeros((4,4,nx,ny,nz),dtype=rho.dtype)
    #fuu[0,0]=0*gdet
    #fuu[1,1]=0*gdet
    #fuu[2,2]=0*gdet
    #fuu[3,3]=0*gdet
    fuu[0,1]=-1/gdet*(ud[2]*bd[3]-ud[3]*bd[2]) # f^tr
    fuu[1,0]=-fuu[0,1]
    fuu[0,2]=-1/gdet*(ud[3]*bd[1]-ud[1]*bd[3]) # f^th
    fuu[2,0]=-fuu[0,2]
    fuu[0,3]=-1/gdet*(ud[1]*bd[2]-ud[2]*bd[1]) # f^tp
    fuu[3,0]=-fuu[0,3]
    fuu[1,3]=-1/gdet*(ud[2]*bd[0]-ud[0]*bd[2]) # f^rp
    fuu[3,1]=-fuu[1,3]
    fuu[2,3]=-1/gdet*(ud[0]*bd[1]-ud[1]*bd[0]) # f^hp
    fuu[3,2]=-fuu[2,3]
    fuu[1,2]=-1/gdet*(ud[0]*bd[3]-ud[3]*bd[0]) # f^rh
    fuu[2,1]=-fuu[1,2]
    #
    # these 2 are equal in degen electrodynamics when d/dt=d/dphi->0
    omegaf1=fdd[0,1]/fdd[1,3] # = ftr/frp
    omegaf2=fdd[0,2]/fdd[2,3] # = fth/fhp
    #
    B1hat=B[1]*np.sqrt(gv3[1,1])
    B2hat=B[2]*np.sqrt(gv3[2,2])
    B3nonhat=B[3]
    v1hat=uu[1]*np.sqrt(gv3[1,1])/uu[0]
    v2hat=uu[2]*np.sqrt(gv3[2,2])/uu[0]
    v3nonhat=uu[3]/uu[0]
    #
    aB1hat=np.fabs(B1hat)
    aB2hat=np.fabs(B2hat)
    av1hat=np.fabs(v1hat)
    av2hat=np.fabs(v2hat)
    #
    vpol=np.sqrt(av1hat**2 + av2hat**2)
    Bpol=np.sqrt(aB1hat**2 + aB2hat**2)
    #
    #omegaf1b=(omegaf1*aB1hat+omegaf2*aB2hat)/(aB1hat+aB2hat)
    #E1hat=fdd[0,1]*np.sqrt(gn3[1,1])
    #E2hat=fdd[0,2]*np.sqrt(gn3[2,2])
    #Epabs=np.sqrt(E1hat**2+E2hat**2)
    #Bpabs=np.sqrt(aB1hat**2+aB2hat**2)+1E-15
    #omegaf2b=Epabs/Bpabs
    #
    # assume field swept back so omegaf is always larger than vphi (only true for outflow, so put in sign switch for inflow as relevant for disk near BH or even jet near BH)
    # GODMARK: These assume rotation about z-axis
    omegaf2b=np.fabs(v3nonhat) + np.sign(uu[1])*(vpol/Bpol)*np.fabs(B3nonhat)
    #
    # below omega for the field
    omegaf1b=v3nonhat - B3nonhat*(v1hat*B1hat+v2hat*B2hat)/(B1hat**2+B2hat**2)

def checkiffullavgexists():
    #
    fname = "avg2d.npy"
    print(("checkiffullavgexists(): checking for fname=%s" % (fname))) ; sys.stdout.flush()
    #
    if os.path.isfile( fname ):
        print(( "File %s exists" % fname ));sys.stdout.flush()
        return(1)
    else:
        print(( "File %s does not exist" % fname ));sys.stdout.flush()
        return(0)


###################################
#
# Functions for reinterpolating the simulation grid
#
###################################
# 3D interp from Vorig to V (Vmetric usually)
# Requires input of full 3D Vorig and Vmetric
# TOO SLOW!
def reinterp3dspc(Vorig,Vmetric,vartointerp):
    #
    # grid the data.
    # first position args are original locations
    # second position args are new locations
    print(("reinterp3dspc shapes:",Vorig[1].shape, Vorig[2].shape, Vorig[3].shape, vartointerp.shape, Vmetric[1,].shape, Vmetric[2].shape, Vmetric[3].shape)) ; sys.stdout.flush()
    #varinterpolated = griddata((Vorig[1], Vorig[2], Vorig[3]), vartointerp, (Vmetric[1], Vmetric[2], Vmetric[3]), method='linear') # no cubic for 3D data.  Could do each slice cubiclaly.
    #
    if 1==0:
        pts = np.array((Vorig[1].ravel(),Vorig[2].ravel(),Vorig[3].ravel())).T
        varinterpolatedfun = sp.interpolate.LinearNDInterpolator(pts, vartointerp.ravel(), fill_value=0.0)
        ptsnew = np.array((Vmetric[1].ravel(),Vmetric[2].ravel(),Vmetric[3].ravel())).T
        varinterpolated=varinterpolatedfun(ptsnew)  #((Vmetric[1], Vmetric[2], Vmetric[3]))
    else:
        pts = np.array((Vorig[1].ravel(),Vorig[2].ravel(),Vorig[3].ravel())).T
        ptsnew = np.array((Vmetric[1,20,15,20].ravel(),Vmetric[2,20,15,20].ravel(),Vmetric[3,20,15,20].ravel())).T
        print("ptsnew") ; sys.stdout.flush()
        print(ptsnew) ; sys.stdout.flush()
        varinterpolatedfun1 = sp.interpolate.LinearNDInterpolator(pts, Vorig[1].ravel(), fill_value=0.0)
        print("duck1") ; sys.stdout.flush()
        result1=varinterpolatedfun1(ptsnew)  #((Vmetric[1], Vmetric[2], Vmetric[3]))
        print(("results: Vmetric: %g %g %g  :: Vorig1: %g\n" % (Vmetric[1,20,15,20],Vmetric[2,20,15,20],Vmetric[3,20,15,20],result1)))
        varinterpolatedfun2 = sp.interpolate.LinearNDInterpolator(pts, Vorig[2].ravel(), fill_value=0.0)
        print("duck2") ; sys.stdout.flush()
        result2=varinterpolatedfun2(ptsnew)  #((Vmetric[1], Vmetric[2], Vmetric[3]))
        print(("results: Vmetric: %g %g %g  :: Vorig2: %g\n" % (Vmetric[1,20,15,20],Vmetric[2,20,15,20],Vmetric[3,20,15,20],result2)))
        varinterpolatedfun3 = sp.interpolate.LinearNDInterpolator(pts, Vorig[3].ravel(), fill_value=0.0)
        print("duck3") ; sys.stdout.flush()
        result3=varinterpolatedfun3(ptsnew)  #((Vmetric[1], Vmetric[2], Vmetric[3]))
        print(("results: Vmetric: %g %g %g  :: Vorig3: %g\n" % (Vmetric[1,20,15,20],Vmetric[2,20,15,20],Vmetric[3,20,15,20],result3)))
        sys.stdout.flush()
    #
    exit
    #
    return(varinterpolated)

# this interpolates all things at once that one wants to interpolate.
# this also optimally uses fact that grid is just SPC (not competely general crazy grid) *and* that only performed rigid rotation *and* that dxdxp3[012]=0 *and* dxdxp12=0 (i.e. dr/dx2=0)
def reinterp3dspc_opt_all(Vorig,Vmetric,rho,ug,uu,B,gdetB=None):
    #
    # get starting time so can compute time differences
    start_time=datetime.now()
    print(("reinterp3dspc_opt_all(start) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
    #
    ######################################
    # Setup variables
    ######################################
    #
    rhoi=np.copy(rho)
    ugi=np.copy(ug)
    uui=np.copy(uu)
    Bi=np.copy(B)
    if gdetB!=None:
        gdetBi=np.copy(gdetB)
    #
    # tk is only axisymmetric information, so create fake nz-long tk called faketk
    faketk=np.zeros(nz,dtype=np.int)
    for kk in np.arange(0,nz):
        faketk[kk]=kk
    #
    # tj
    faketj=np.zeros(ny,dtype=np.int)
    for jj in np.arange(0,ny):
        faketj[jj]=jj
    #
    # ti
    faketi=np.zeros(nx,dtype=np.int)
    for ii in np.arange(0,nx):
        faketi[ii]=ii
    #
    # first determine which ph = Vmetric[\phi[x3[tk]]] = Vorig[kk]
    # Vorig already setup to be in Vmetric span of \theta,\phi
    # do for all ii,jj,kk at once
    # can do this outside loop because ph(x3) only for now
    # flatten Vorig[3] before passing to kofphfloat that takes 1D array
    # reshape result back into 3D array
    tkorigarray=kofphfloat(faketk,Vmetric[3,0,0,:],Vorig[3].view().reshape(-1)).reshape((nx,ny,nz))
    # integerize
    #inttkorigarray=tkorigarray.astype(int)
    inttkorigarray=np.round(tkorigarray).astype(int)
    #
    # can do this outside loop because r(x1) only for now
    if 0==1:
        tiorigarray=iofrfloat(faketi,Vmetric[1,:,0,0],Vorig[1])
        inttiorigarray=tiorigarray.astype(int)
    else:
        pass
        # infact, r doesn't change in rotation, so just assign
        #... just use tifake below
        #... just use ii directly below
    #
    print(("reinterp3dspc_opt_all(done with tk) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
    #
    ######################################
    # Interpolate (and transform from X->V for vectors)
    ######################################
    #
    # explore fast way:
    if 1==1:
        tjorigarray=np.copy(tkorigarray)
        for kk in np.arange(0,nz):
            for jj in np.arange(0,ny):
                for ii in np.arange(0,nx):
                    Vorig2array=np.array([Vorig[2,ii,jj,kk]])
                    # note that Vmetric[2] is function of r(ii) and jj and through inttkorigarray is function of kk, so can't feed in an array of V2orig2array.  Can only do 1 value at a time since list of theta changes for each ii,jj,kk
                    tjorigarray[ii,jj,kk]=jofhfloat(faketj[:],Vmetric[2,ii,:,inttkorigarray[ii,jj,kk]],Vorig2array)
        #
        #inttjorigarray=tjorigarray.astype(int)
        inttjorigarray=np.round(tjorigarray).astype(int)
        #
        print(("reinterp3dspc_opt_all(done with tjorigarray) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
        #
        rhoi[:,:,:]=rho[faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]
        ugi[:,:,:]=ug[faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]
        #
        print(("reinterp3dspc_opt_all(done with rho,ug) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
        #
        # do dxdxp first since needed for both uu and B.  Only need 4 full spatially dependent values of dxdxp
        if 1==1:
            dxdxpi=np.copy(uu)
            dxdxpi[0,:,:,:]=dxdxp[1,1,faketi[:,None,None],inttjorigarray[:,:,:],0] #inttkorigarray[:,:,:]]
            dxdxpi[1,:,:,:]=dxdxp[1,2,faketi[:,None,None],inttjorigarray[:,:,:],0] #inttkorigarray[:,:,:]]
            dxdxpi[2,:,:,:]=dxdxp[2,1,faketi[:,None,None],inttjorigarray[:,:,:],0] #inttkorigarray[:,:,:]]
            dxdxpi[3,:,:,:]=dxdxp[2,2,faketi[:,None,None],inttjorigarray[:,:,:],0] #inttkorigarray[:,:,:]]
        #
        print(("reinterp3dspc_opt_all(done with dxdxp11,12,21,22) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
        #
        if 1==1:
            uui[0,:,:,:]=uu[0,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]
            uui[1,:,:,:]=uu[1,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]
            uui[2,:,:,:]=uu[2,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]
            uui[3,:,:,:]=uu[3,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]

            uui = dxdxpverysimpletensordot(uui,dxdxpi,dxdxp)
        else:
            #uui[:,:,:,:] = dxdxpsimpletensordot(uu[:,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]],dxdxp[:,:,faketi[:,None,None],inttjorigarray[:,:,:],0],axes=[0,1])
            uui[:,:,:,:] = dxdxpsimpletensordot4uu(uu,dxdxp,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:])
        #
        print(("reinterp3dspc_opt_all(done with uu[0-4]) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
        #
        if 1==1:
            #Bi[0,:,:,:]=B[0,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]] # should still be zero
            Bi[0]=B[0]*0.0 # still zero
            Bi[1,:,:,:]=B[1,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]
            Bi[2,:,:,:]=B[2,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]
            Bi[3,:,:,:]=B[3,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]

            Bi = dxdxpverysimpletensordot(Bi,dxdxpi,dxdxp)
        else:
            #Bi[:,:,:,:] = dxdxpsimpletensordot(B[:,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]],dxdxp[:,:,faketi[:,None,None],inttjorigarray[:,:,:],0],axes=[0,1])
            Bi[:,:,:,:] = dxdxpsimpletensordot4B(B,dxdxp,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:])
        #
        print(("reinterp3dspc_opt_all(done with B[1-4]) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
        #
        if gdetB!=None:
            if 1==1:
                #gdetBi[0,:,:,:]=gdetB[0,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]] # should still be zero
                gdetBi[0]=gdetB[0]*0.0 # still zero
                gdetBi[1,:,:,:]=gdetB[1,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]
                gdetBi[2,:,:,:]=gdetB[2,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]
                gdetBi[3,:,:,:]=gdetB[3,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]]

                gdetBi = dxdxpverysimpletensordot(gdetBi,dxdxpi,dxdxp)
            else:
                # GODMARK: not quite right to transform gdetB this way (need to divide out gdet and put it back, but don't have gdet at faces)
                #gdetBi[:,:,:,:] = dxdxpsimpletensordot(gdetB[:,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:]],dxdxp[:,:,faketi[:,None,None],inttjorigarray[:,:,:],0],axes=[0,1])
                gdetBi[:,:,:,:] = dxdxpsimpletensordot4B(gdetB,dxdxp,faketi[:,None,None],inttjorigarray[:,:,:],inttkorigarray[:,:,:])

            #
            print(("reinterp3dspc_opt_all(done with gdetB[1-4]) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
            #
            #
    #
    # faster than generic interpolation, but still kinda slow
    if 1==0:
        #
        print(("reinterp3dspc_opt_all(start fast but slower interp) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
        #
        for kk in np.arange(0,nz):
            for jj in np.arange(0,ny):
                for ii in np.arange(0,nx):
                    #
                    # first determine which ph = Vmetric[\phi[x3[tk]]] = Vorig[kk]
                    # Vorig already setup to be in Vmetric span of \theta,\phi
                    if 0==1:
                        tkorig=kofphfloat(faketk,Vmetric[3,ii,jj,:],Vorig[3,ii,jj,kk])
                        inttkorig=tkorig.astype(int)
                    else:
                        inttkorig=inttkorigarray[ii,jj,kk]
                    # tkorig is tk we want to grab data from and stick into ii,jj,kk position -- if no interpolation
                    #
                    # now get radius, which is also independent of tj,tk.
                    if 0==1:
                        tiorig=iofrfloat(faketi,Vmetric[1,:,jj,inttkorig],Vorig[1,ii,jj,kk])
                        inttiorig=tiorig.astype(int)
                    else:
                        # just rotation, so ii same
                        #inttiorig=inttiorigarray[ii,jj,kk]
                        inttiorig=ii
                    #
                    # now get theta. theta(x1,x2) generally, so have to do this after getting tiorig.
                    if 0==1:
                        tjorig=jofhfloat(faketj,Vmetric[2,inttiorig,:,inttkorig],Vorig[2,ii,jj,kk])
                        inttjorig=tjorig.astype(int)
                    else:
                        # have to be inside loop to use correct r
                        #tjorig=jofhfloat(faketj,Vmetric[2,inttiorig,:,inttkorig],Vorig[2,ii,jj,kk])
                        # assume r(x1) too so avoid inttiorig[ii,jj,kk] by using fake[ii] or just ii directly
                        # still have to be inside loop to have correct inttkorig -- k does change on rotation and that will choose different \theta?
                        # No, same k too -- i.e. Vmetric[2,ii,kk] = Vmetric[2,inttiorig,inttkorig]
                        # Rotational dependence in \phi comes in via which k used for final primitive, not Vmetric[2]
                        #tjorig=jofhfloat(faketj,Vmetric[2,ii,:,inttkorig],Vorig[2,ii,jj,kk])
                        #
                        Vorig2array=np.array([Vorig[2,ii,jj,kk]])
                        tjorig=jofhfloat(faketj,Vmetric[2,inttiorig,:,inttkorig],Vorig2array)
                        #inttjorig=tjorig.astype(int)
                        # use round to be more consistent/accurate with nearest neighbors
                        inttjorig=np.round(tjorig).astype(int)
                    #
                    # DEBUG:
                    #print("REPORT: %d %d %d -> %d %d %d : jfloat=%g\n" % (ii,jj,kk, inttiorig,inttjorig,inttkorig,tjorig))
                    #print("Vorig123: %g %g %g\n" % (Vorig[1,ii,jj,kk],Vorig[2,ii,jj,kk],Vorig[3,ii,jj,kk]))
                    #print("Vmetric123: %g %g %g\n" % (Vmetric[1,ii,jj,kk],Vmetric[2,ii,jj,kk],Vmetric[3,ii,jj,kk]))
                    #print("Vmetric123orig: %g %g %g\n" % (Vmetric[1,inttiorig,inttjorig,inttkorig],Vmetric[2,inttiorig,inttjorig,inttkorig],Vmetric[3,inttiorig,inttjorig,inttkorig]))
                    #
                    # now tiorig,tjorig,tkorig (and associated integer locations) provide location for interpolation of primitives.
                    # However, no point to feed in entrire arrays for every ii,jj,kk -- wasteful.  So just do interpolation myself
                    # for now, try nearest neighbor, to get things going
                    rhoi[ii,jj,kk]=rho[inttiorig,inttjorig,inttkorig]
                    ug[ii,jj,kk]=ug[inttiorig,inttjorig,inttkorig]
                    uui[0,ii,jj,kk]=uu[0,inttiorig,inttjorig,inttkorig]
                    uui[1,ii,jj,kk]=uu[1,inttiorig,inttjorig,inttkorig]
                    uui[2,ii,jj,kk]=uu[2,inttiorig,inttjorig,inttkorig]
                    uui[3,ii,jj,kk]=uu[3,inttiorig,inttjorig,inttkorig]
                    #Bi[0,ii,jj,kk]=B[0,inttiorig,inttjorig,inttkorig] # should still be zero
                    Bi[1,ii,jj,kk]=B[1,inttiorig,inttjorig,inttkorig]
                    Bi[2,ii,jj,kk]=B[2,inttiorig,inttjorig,inttkorig]
                    Bi[3,ii,jj,kk]=B[3,inttiorig,inttjorig,inttkorig]
                    if gdetB!=None:
                        #gdetBi[0,ii,jj,kk]=gdetB[0,inttiorig,inttjorig,inttkorig] # should still be zero
                        gdetBi[1,ii,jj,kk]=gdetB[1,inttiorig,inttjorig,inttkorig]
                        gdetBi[2,ii,jj,kk]=gdetB[2,inttiorig,inttjorig,inttkorig]
                        gdetBi[3,ii,jj,kk]=gdetB[3,inttiorig,inttjorig,inttkorig]
                    #
                #
            #
        #
        print(("reinterp3dspc_opt_all(end fast but slower interp) time elapsed: %d" % (datetime.now()-start_time).seconds )) ; sys.stdout.flush()
        #
    ######################################
    # Return results
    ######################################
    #
    if(gdetB==None):
        return(rhoi,ugi,uui[0],uui[1],uui[2],uui[3],Bi[1],Bi[2],Bi[3])
    else:
        return(rhoi,ugi,uui[0],uui[1],uui[2],uui[3],Bi[1],Bi[2],Bi[3],gdetBi[1],gdetBi[2],gdetBi[3])

def reinterpxy(vartointerp,extent,ncell,domask=1,interporder='cubic'):
    global xi,yi,zi
    #grid3d("gdump")
    #rfd("fieldline0250.bin")
    xraw=r*np.sin(h)*np.cos(ph)
    yraw=r*np.sin(h)*np.sin(ph)
    #2 cells below the midplane
    x=xraw[:,ny//2,:].view().reshape(-1)
    y=yraw[:,ny//2,:].view().reshape(-1)
    var=vartointerp[:,ny//2,:].view().reshape(-1)
    # define grid.
    xi = np.linspace(extent[0], extent[1], ncell)
    yi = np.linspace(extent[2], extent[3], ncell)
    # grid the data.
    zi = griddata((x, y), var, (xi[None,:], yi[:,None]), method=interporder)
    #zi[interior] = np.ma.masked
    if domask!=0:
        interior = np.sqrt((xi[None,:]**2) + (yi[:,None]**2)) < (1+np.sqrt(1-a**2))*domask
        varinterpolated = ma.masked_where(interior, zi)
    else:
        varinterpolated = zi
    return(varinterpolated)

def reinterpxyz(vartointerp,b,ncell,domask=1,interporder='cubic'):
    # Max's attempt at 3d interp 1/11/21
    # As of 1/13/21 I tried using 'linear' instead of 'cubic' for bug-squashing purposes
    # todo: b is a new vairable in this so, add that in for every time this function is called

    global xi, yi, zi, ai
    xraw = r*np.sin(h)*np.cos(ph)
    yraw = r*np.sin(h)*np.sin(ph)
    zraw = r*np.cos(h)

    #data out to radius r=sqrt(3)*b
    rad = np.sqrt(3.0)*b #calculate the radius of a circle that a cube with side length 2*b will fit in
    irad  = int(iofr(rad)) #find the index corresponding to rad - maybe do a slightly larger cube in case griddata needs points outside the region of interest?
    x=xraw[0:irad,:,:].view().reshape(-1)
    y=yraw[0:irad,:,:].view().reshape(-1)
    z=zraw[0:irad,:,:].view().reshape(-1)
    var=vartointerp[0:irad,:,:].view().reshape(-1)

    extent = (-b,b,-b,b,-b,b)

    # define grid.
    xi = np.linspace(extent[0], extent[1], ncell)
    yi = np.linspace(extent[2], extent[3], ncell)
    zi = np.linspace(extent[4], extent[5], ncell)

    # grid the data, adjusted FOR 3D
    # points = np.array((x, y, z)).T # the first griddata argument, it looks like, must be in transposed like this
    print("If you can see this, then at least it got to linspace. -Max 1/14")
    ai = griddata((x,y,z), var, (xi[None,:,None], yi[:,None,None], zi[None,None,:]), method=interporder)

    if domask!=0:
        interior = np.sqrt((xi[None,:, None]**2) + (yi[:,None, None]**2)+ (zi[None,None,:]**2)) < (1+np.sqrt(1-a**2))*domask
        varinterpolated = ma.masked_where(interior, ai)
    else:
        varinterpolated = ai
    return(varinterpolated)

def reinterpxyhor(vartointerp,extent,ncell,domask=1,interporder='cubic'):
    '''function made to project qtys on the bh horizon down to the midplane - created by Megan 2/29/16'''
    global xi,yi,zi
    #grid3d("gdump")
    #rfd("fieldline0250.bin")
    rhor=1+(1-a**2)**0.5
    ihor=int(np.floor(iofr(rhor)+0.5))
    xraw=r*np.sin(h)*np.cos(ph)
    yraw=r*np.sin(h)*np.sin(ph)
    #restrict values to BH upper hemisphere
    x=xraw[ihor,0:ny//2,:].view().reshape(-1)
    y=yraw[ihor,0:ny//2,:].view().reshape(-1)
    var=vartointerp[ihor,0:ny//2,:].view().reshape(-1)
    #mirror
    if nz*_dx3*dxdxp[3,3,0,0,0] < 0.99 * 2 * np.pi:
        x=np.concatenate((-x,x))
        y=np.concatenate((-y,y))
        var=np.concatenate((var,var))
    # define grid.
    xi = np.linspace(extent[0], extent[1], ncell)
    yi = np.linspace(extent[2], extent[3], ncell)
    # grid the data.
    zi = griddata((x, y), var, (xi[None,:], yi[:,None]), method=interporder)
    #zi[interior] = np.ma.masked
    if domask!=0:
        interior = np.sqrt((xi[None,:]**2) + (yi[:,None]**2)) > (1+np.sqrt(1-a**2))*domask
        varinterpolated = ma.masked_where(interior, zi)
    else:
        varinterpolated = zi
    return(varinterpolated)

def velinterp_3d(fnumber, rng, extent, ncell):
    '''Max's attempt at a function which uses reinterpxyz() instead of reinterpxy()
    1/11/2021'''
    grid3d("gdump.bin",use2d=False)
    #load the fieldline file for a given time and compute standard quantities
    rfd("fieldline"+str(fnumber)+".bin")
    cvel()
    rhor=1+(1-a**2)**0.5
    ihor=np.floor(iofr(rhor)+0.5)
    #compute the 3-velocities in the equatorial slice
    vr = dxdxp[1,1]*uu[1]/uu[0]+dxdxp[1,2]*uu[2]/uu[0]
    vh = dxdxp[2,1]*uu[1]/uu[0]+dxdxp[2,2]*uu[2]/uu[0]
    vp = uu[3]/uu[0]*dxdxp[3,3]
    #
    vrnorm=vr
    vhnorm=vh*np.abs(r)
    vpnorm=vp*np.abs(r*np.sin(h))
    #
    vznorm=vrnorm*np.cos(h)-vhnorm*np.sin(h)
    vRnorm=vrnorm*np.sin(h)+vhnorm*np.cos(h)
    vxnorm=vRnorm*np.cos(ph)-vpnorm*np.sin(ph)
    vynorm=vRnorm*np.sin(ph)+vpnorm*np.cos(ph)
    vRhor=vhnorm*np.cos(h)
    vxhor=vRhor*np.cos(ph)-vpnorm*np.sin(ph)
    vyhor=vRhor*np.sin(ph)+vpnorm*np.cos(ph)
    #make uniform grid for velocity
    ivx=reinterpxyz(vxnorm,extent,ncell,domask=1,interporder='linear')
    ivx_h=reinterpxyhor(vxhor,extent,ncell,domask=1,interporder='linear')
    ivx[ivx.mask==True]=ivx_h[ivx.mask==True]
    ivy=reinterpxyz(vynorm,extent,ncell,domask=1,interporder='linear')
    ivy_h=reinterpxyhor(vyhor,extent,ncell,domask=1,interporder='linear')
    ivy[ivy.mask==True]=ivy_h[ivy.mask==True]

    return ivx, ivy

def velinterp(fnumber, rng, extent, ncell):
    grid3d("gdump.bin",use2d=True)
    #load the fieldline file for a given time and compute standard quantities
    rfd("fieldline"+str(fnumber)+".bin")
    cvel()
    rhor=1+(1-a**2)**0.5
    ihor=np.floor(iofr(rhor)+0.5)
    #compute the 3-velocities in the equatorial slice
    vr = dxdxp[1,1]*uu[1]/uu[0]+dxdxp[1,2]*uu[2]/uu[0]
    vh = dxdxp[2,1]*uu[1]/uu[0]+dxdxp[2,2]*uu[2]/uu[0]
    vp = uu[3]/uu[0]*dxdxp[3,3]
    #
    vrnorm=vr
    vhnorm=vh*np.abs(r)
    vpnorm=vp*np.abs(r*np.sin(h))
    #
    vznorm=vrnorm*np.cos(h)-vhnorm*np.sin(h)
    vRnorm=vrnorm*np.sin(h)+vhnorm*np.cos(h)
    vxnorm=vRnorm*np.cos(ph)-vpnorm*np.sin(ph)
    vynorm=vRnorm*np.sin(ph)+vpnorm*np.cos(ph)
    vRhor=vhnorm*np.cos(h)
    vxhor=vRhor*np.cos(ph)-vpnorm*np.sin(ph)
    vyhor=vRhor*np.sin(ph)+vpnorm*np.cos(ph)
    #make uniform grid for velocity
    ivx=reinterpxy(vxnorm,extent,ncell,domask=1,interporder='linear')
    ivx_h=reinterpxyhor(vxhor,extent,ncell,domask=1,interporder='linear')
    ivx[ivx.mask==True]=ivx_h[ivx.mask==True]
    ivy=reinterpxy(vynorm,extent,ncell,domask=1,interporder='linear')
    ivy_h=reinterpxyhor(vyhor,extent,ncell,domask=1,interporder='linear')
    ivy[ivy.mask==True]=ivy_h[ivy.mask==True]

    return ivx, ivy

def construct_cartesian():
    # takes the raw data from rf, hf, and phf to make an array in cartesian coords
    # first we call the functions to get rh, hf, and phf
    grid3d("gdump.bin",use2d = False)
    gridcellverts()

    x_cart = np.zeros_like(rf)
    y_cart = np.zeros_like(hf)
    z_cart = np.zeros_like(phf)

    x_cart=rf* np.sin(hf)*np.cos(phf)
    y_cart=rf* np.sin(hf)*np.sin(phf)
    z_cart=rf*np.cos(hf)

    print(x_cart.shape)

    return x_cart, y_cart, z_cart

def test_random_points():
    # compared inputs and outputs to this calculator: https://keisan.casio.com/exec/system/1359534351
    # CAVEAT: this site treats theta as the angle in the xy-plane and phi as the azimuthal

    # I tested five or so points like so, it looks like it works! But it doesn't seem to work anymore?
    # TO DO (Max): FIGURE OUT HOW IT BROKE
    grid3d("gdump.bin",use2d = False)
    gridcellverts()
    x_cart, y_cart, z_cart = construct_cartesian()
    digit = rnd.randint(0, 96)
    print("Digit chosen: " + str(digit))
    print("in spherical:")
    print(rf.shape)
    print(rf[0][digit], hf[0][digit], phf[0][digit])
    print("In cartesian:")
    print(x_cart[0][digit], y_cart[0][digit], z_cart[0][digit])

# initializing this in a global scope, right here, ensures that rfd() returns no errors
use2dglobal = False

def load_array_as_cartesian(x_array, y_array, z_array):
    # Takes x, y, z and (attempts) to load it into a yt framework
    # It hasn't worked yet
    import yt

    #return coords, conn
    coords, conn = yt.hexahedral_connectivity(x_array[0][0], y_array[0][0], z_array[0][0]) # This is what I likely messed up
    data = {"density" : rho} # make a dict of the densities (todo: add b-field components once this function works)
    ds = yt.load_hexahedral_mesh(data, conn, coords,
        bbox = np.array([[-10000.0, 10000.0], [-10000.0, 10000.0], [-10000.0, 10000.0]]),
        geometry = 'cartesian')
    return ds

def grid3d_rhph(dumpname,use2d=False,doface=False,usethetarot0=False): #read grid dump file: header and body
    # THIS IS A COPY of grid3d that only deals with r, h, and ph (Max 12/17)
    #
    #
    if usethetarot0==True:
        filename="dumps/gdump.THETAROT0.bin"
        dumpname="gdump.THETAROT0.bin" # override input
    else:
        filename="dumps/gdump.bin"
        # just use input dumpname
    #
    if os.path.isfile(filename):
        # only need header of true gdump.bin to get true THETAROT
        rfdheaderonly(filename)
    else:
        # if no gdump, use last fieldline file that is assumed to be consistent with gdump that didn't exist.
        # allows non-creation of gdump if restarting with tilt from non-tilt run.  So then enver have to have gdump.bin with THETAROT tilt.
        rfdheaderlastfile()
    #
    # for rfd() to use to see if different nz size
    global nzgdumptrue
    nzgdumptrue=nz

    # keeping only the global vars we need for calculations
    global nxgdump,nygdump,nzgdump,THETAROTgdump
    nxgdump=nx
    nygdump=ny
    nzgdump=nz
    THETAROTgdump=THETAROT

    realdumpname=dumpname
    #
    print(( "realdumpname=%s" % (realdumpname) )) ; sys.stdout.flush()
    #
    # load axisymmetric metric-grid data
    # this sets THETAROT=0 if THETAROT true is non-zero.  rfd() is responsible for setting THETAROT for each fieldline file so data inputted is transformed/interpolated correctly.
    grid3d_load(dumpname=realdumpname,doface=doface,loadsimple=False)
    #
    # get other things
    gridcellverts_rhph()
    #
    gc.collect() #try to release unneeded memory
    print( "Done grid3d!" ) ; sys.stdout.flush()

def grid3d_load_rhph(dumpname=None,use2d=False,doface=False,loadsimple=False): #read grid dump file: header and body
    # similar to grid3d_rhph, THIS IS A COPY of grid3d_load() but ONLY HELPS IN CALCULATING r, theta, and phi
    #The internal cell indices along the three axes: (ti, tj, tk)
    #The internal uniform coordinates, (x1, x2, x3), are mapped into the physical
    #non-uniform coordinates, (r, h, ph), which correspond to radius (r), polar angle (theta), and toroidal angle (phi).
    #There are more variables, e.g., dxdxp, which is the Jacobian of (x1,x2,x3)->(r,h,ph) transformation, that I can
    #go over, if needed.
    global Rin,Rout
    global nzgdump, lnz, dxdxp
    global r,h,ph
    # global ck,conn
    print(( "Reading grid from " + "dumps/" + dumpname + " ..." )) ; sys.stdout.flush()
    gin = open( "dumps/" + dumpname, "rb" )
    #
    #First line of grid dump file is a text line that contains general grid information:
    header = gin.readline().split()

    #Spherical polar radius of the innermost radial cell
    Rin=myfloatalt(float(header[14]))
    #Spherical polar radius of the outermost radial cell
    Rout=myfloatalt(float(header[15]))
    #read grid dump per-cell data
    #
    lnz = nz
    #
    print( "Done reading grid header" ) ; sys.stdout.flush()
    #
    ncols = 126
    print(( "Start reading grid as binary with lnz=%d" % (lnz) )) ; sys.stdout.flush()
    body = np.fromfile(gin,dtype=np.float64,count=ncols*nx*ny*lnz)
    gd = body.view().reshape((-1,nx,ny,lnz),order='F')
    gin.close()
    print(( "Done reading grid as binary with lnz=%d" % (lnz) )) ; sys.stdout.flush()
    gd=myfloat(gd)
    gc.collect()
    #
    print( "Done reading grid" ) ; sys.stdout.flush()
    #
    # always load ti,tj,tk,x1,x2,x3,r,h,ph
    # SUPERNOTEMARK: for use2d, note that tk depends upon \phi unlike all other things for a Kerr metric in standard coordinates
    r,h,ph = gd[6:9,:,:,:].view()

    dxdxp = gd[110:126].view().reshape((4,4,nx,ny,lnz), order='F').transpose(1,0,2,3,4)

def gridcellverts_rhph():
    # like the two functions above, it is a copy of gridcellverts() without tif,tjf,and tkf
    ##################################
    #CELL VERTICES:
    global rf,hf,phf
    #RADIAL:
    #add an extra dimension to rf container since one more faces than centers
    rf = np.zeros((r.shape[0]+1,r.shape[1]+1,r.shape[2]+1))
    #operate on log(r): average becomes geometric mean, etc
    rf[1:nx,0:ny,0:lnz] = (r[1:nx]*r[0:nx-1])**0.5 #- 0.125*(dxdxp[1,1,1:nx]/r[1:nx]-dxdxp[1,1,0:nx-1]/r[0:nx-1])*_dx1
    #extend in theta
    rf[1:nx,ny,0:lnz] = rf[1:nx,ny-1,0:lnz]
    #extend in phi
    rf[1:nx,:,lnz]   = rf[1:nx,:,lnz-1]
    #extend in r
    rf[0] = 0*rf[0] + Rin
    rf[nx] = 0*rf[nx] + Rout
    #ANGULAR:
    hf = np.zeros((h.shape[0]+1,h.shape[1]+1,h.shape[2]+1))
    hf[0:nx,1:ny,0:lnz] = 0.5*(h[:,1:ny]+h[:,0:ny-1]) #- 0.125*(dxdxp[2,2,:,1:ny]-dxdxp[2,2,:,0:ny-1])*_dx2
    hf[1:nx-1,1:ny,0:lnz] = 0.5*(hf[0:nx-2,1:ny,0:lnz]+hf[1:nx-1,1:ny,0:lnz])
    #populate ghost cells in r
    hf[nx,1:ny,0:lnz] = hf[nx-1,1:ny,0:lnz]
    #populate ghost cells in phi
    hf[:,1:ny,lnz] = hf[:,1:ny,lnz-1]
    #populate ghost cells in theta (note: no need for this since already initialized everything to zero)
    hf[:,0] = 0*hf[:,0] + 0
    hf[:,ny] = 0*hf[:,ny] + np.pi
    #TOROIDAL:
    phf = np.zeros((ph.shape[0]+1,ph.shape[1]+1,ph.shape[2]+1))
    phf[0:nx,0:ny,0:lnz] = ph[0:nx,0:ny,0:lnz] - dxdxp[3,3,0,0,0]*0.5*_dx3
    #extend in phi
    phf[0:nx,0:ny,lnz]   = ph[0:nx,0:ny,lnz-1] + dxdxp[3,3,0,0,0]*0.5*_dx3
    #extend in r
    phf[nx,0:ny,:]   =   phf[nx-1,0:ny,:]
    #extend in theta
    phf[:,ny,:]   =   phf[:,ny-1,:]

def make_simplified_array(fieldname):
    # should create the r h and ph array as an array of vertices
    grid3d_rhph('gdump.bin', use2d=False) # loads the data
    rfd(fieldname) # I call this to initialize rho
    gridcellverts_rhph() # converts to corners

    # splits the 3d arrays into their unique columns
    unique_r = rf[:,0,0]
    unique_h = hf[0,:,0]
    unique_ph = phf[0,0,:]

    # the other thing: truncate r at r = 50 if it's too big
    xf=int(iofr(50)) # the index at which we should truncate if necessary
    return unique_r, unique_h, unique_ph

def load_simplified_array(unique_r, unique_h, unique_ph):
    # takes the global rh, hf, phf variables and uses that to load to yt
    import yt
    yt.visualization.plot_modifications.ContourCallback._supported_geometries += ("spherical",)
    #return coords, conn
    coords, conn = yt.hexahedral_connectivity(unique_r, unique_h, unique_ph)
    data = {"density" : rho} # make a dict of the densities
    ds = yt.load_hexahedral_mesh(data, conn, coords,
        bbox = np.array([[0.0, 10000.0], [0.0, np.pi], [0.0, 2*np.pi]]),
        geometry = 'spherical')
    return ds

def render_isosurf_as_points(fieldname, rho_min):
    simplified_array = make_simplified_array(fieldname)
    global c_lo

    xraw = r*np.sin(h)*np.cos(ph)
    yraw = r*np.sin(h)*np.sin(ph)
    zraw = r*np.cos(h)

    # Connor came up with a way of limiting the number of points we load in.
    # I modified this slightly to depend on radius instead of number of points

    desired_max_rad = 40
    rad_index = int(iofr(40))

    # make sure these indices are correct
    x_short=xraw[0:rad_index,:,:].view().reshape(-1)
    y_short=yraw[0:rad_index,:,:].view().reshape(-1)
    z_short=zraw[0:rad_index,:,:].view().reshape(-1)
    rho_short=lrho[0:rad_index,:,:].view().reshape(-1)

    c_lo = min(rho_short)

    iso_rho = []
    iso_x = []
    iso_y = []
    iso_z = []

    for i in range(len(rho_short)):
        # there's probably a faster way of doing this
        # print(rho_short[i])
        if float(rho_short[i]) >= float(rho_min):
            iso_rho.append(rho_short[i])
            iso_x.append(x_short[i])
            iso_y.append(y_short[i])
            iso_z.append(z_short[i])

    # then create the 3d coordinate array
    coords = np.stack((iso_x, iso_y, iso_z), axis = -1)
    return coords, np.array(iso_rho)
'''
AS OF MONDAY NIGHT (12/21): I (Max) looked at the previous nine functions and here's what seems to be true:
    THE FIRST THREE:
        first function looks okay, the next two are broken but not that useful
    THE NEXT THREE:
        Helper functions, fine
    LAST THREE:
        are working!

Also, I found a list of color pallets: https://yt-project.org/doc/visualizing/colormaps/index.html
Jet is the matplotlib-style one, let's stick with that.

As for controling the radius, it looks like slc.zoom(zoomfactor) can help.
example of zoom function: https://yt-project.org/doc/visualizing/plots.html
more potentially helpful methods: https://yt-project.org/doc/cookbook/complex_plots.html
'''

def load_point_plot(coords, data):
    import pyvista as pv
    import vtk

    # fix the color map:
    c_hi = max(data)

    mesh = pv.PolyData(coords)
    mesh['density'] = data
    pv.set_plot_theme('night')
    mesh.plot(point_size = 1, screenshot = 'density.png', colormap = 'jet', clim = [c_lo,c_hi])

def render_and_load_iso_points(fieldname, rho_min):
    # this is merely a combination of the previous two Functions
    # I created it to make a more sensible color map:
    # rather than use magic numbers, I want the lower threshold of the color limit to be
    # equal to the lowest value of the FULL  array of densities, which required the load function
    # to have access to rho_short

    # initializes necessary global vars
    simplified_array = make_simplified_array(fieldname)

    xraw = r*np.sin(h)*np.cos(ph)
    yraw = r*np.sin(h)*np.sin(ph)
    zraw = r*np.cos(h)

    desired_max_rad = 40
    rad_index = int(iofr(40))

    x_short=xraw[0:rad_index,:,:].view().reshape(-1)
    y_short=yraw[0:rad_index,:,:].view().reshape(-1)
    z_short=zraw[0:rad_index,:,:].view().reshape(-1)
    rho_short=lrho[0:rad_index,:,:].view().reshape(-1)

    c_lo = min(rho_short) # lowest value on the colormap

    iso_rho = []
    iso_x = []
    iso_y = []
    iso_z = []

    for i in range(len(rho_short)):
        # there's probably a faster way of doing this
        # print(rho_short[i])
        if float(rho_short[i]) >= float(rho_min):
            iso_rho.append(rho_short[i])
            iso_x.append(x_short[i])
            iso_y.append(y_short[i])
            iso_z.append(z_short[i])

    # then create the 3d coordinate array
    coords = np.stack((iso_x, iso_y, iso_z), axis = -1)
    data = np.array(iso_rho)

    import pyvista as pv
    import vtk

    # highest value on the color map:
    c_hi = max(rho_short)

    mesh = pv.PolyData(coords)
    mesh['density'] = data
    pv.set_plot_theme('night')
    mesh.plot(point_size = 1, screenshot = 'density.png', colormap = 'jet', clim = [c_lo,c_hi])

# ATTEMPT TO LOAD THE FIELDLINES:
def load_fieldlines(ds):
    # takes the dataset loaded from yt, as in ds = yt.load_hexahedral_mesh(args)
    c = ds.domain_center # center of the sphere
    N = 100 # number of field lines
    scale = ds.domain_width[0]# scale of lines relative to boxsize
    pos_dx = np.random.random((N, 3))*scale-scale/2 # position relative to center (randomly deifined)
    pos = c+pos_dx # absolute location of fieldline pos
    from yt.visualization.api import Streamlines

    # create streamline of 3d vector velocity and integrate through boundary defined above
    streamlines = Streamlines(ds, pos, 'br', 'bh', 'bp', get_magnitude = True)
    streamlines.integrate_through_volume()

    for stream in streamlines.streamlines:
        stream = stream[np.all(stream != 0.0, axis=1)]
        ax.plot3D(stream[:,0], stream[:,1], stream[:,2], alpha=0.1)

    # Save the plot to disk.
    plt.savefig('streamlines.png')

def reinterp_3d_test():
    # WIP: see if it runs without errors
    grid3d_rhph('gdump.bin', use2d=False) # loads the data
    rfd('fieldline14926.bin') # I call this to initialize rho
    extent = (-25., 25., -25., 25., -25., 25.)
    extent2 = (-25., 25., -25., 25.)

    irho1 = reinterpxyz(rho, 40., 100, domask = 1, interporder = 'linear')
    irho2 = reinterpxy(rho, extent2, 100, domask = 1, interporder = 'linear')

    return irho1

def make_sphere_grid(coords, fieldname):
    import pyvista as pv
    simplified_array = make_simplified_array(fieldname)
    grid_scalar = pv.StructuredGrid(coords)

    # grid_scalar.cell_arrays["density"] = rho.flatten()

    p = pv.Plotter()
    p.add_mesh(pv.Sphere())
    p.add_mesh(grid_scalar)
    p.show()
