#!/bin/python

######################################################
## WRAPPER ADOPTING PYTHONTOOLS AS PART OF HARMETAL ##
## USAGE:                                           ##
## ipython -i --pylab="gtk" 1d.py                   ##
## CALLS INTERNALLY:                                ##
## run -i --pylab="gtk" ~/py/mread/__init__.py      ##
######################################################

import commands,sys,string,glob

## USER SPECS ##
TIMESTEP=sys.argv[1]
FIELDLINE_FILES=glob.glob("dumps/fieldline????.bin")
if TIMESTEP==-1 or TIMESTEP=="last":
    DATA_FILE=FIELDLINE_FILES[-1].split("/")[1:][0]
else:
    DATA_FILE="fieldline"+string.zfill(TIMESTEP,4)+".bin"

HOME=commands.getoutput("echo $HOME")

# get routines
# run -i --pylab="gtk" HOME/py/mread/__init__.py
execfile(HOME+"/py/mread/__init__.py")

# get data
#grid3d("gdump")
#rfd(DATA_FILE)

#rho = rho[:,0,0]
#x = r[:,0,0]

#plot(x,rho,'b.',label="Fast Shock")

plot_slice(5000,IC=True,stress=True)
