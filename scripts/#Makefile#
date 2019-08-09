##############
# Only makemoviec has MPI capability
#
# 3b) Run in MPI mode (only with makemoviec binary)
# cd <installdir>
# mpirun -np <numprocs> ./makemoviec <chunklist> <totalchunks> <DATADIR> <jobprefix>
# Where number of chunks in chunklist must be equal to numprocs.
# e.g.
# mpirun -np 4 ./makemoviec "1 2 3 4" 4 . "eoschunk"
#
#
# ensure no odd binaries are still running after mpirun failure: killall makemoviec
#
###########################

USEKIJMCK=0
USEKRAKEN=0
USENAUTILUS=0
USESTAMPEDE=1

#############
# Choose whether to use MPI
USEMPI=1

#############
# Choose whether to use Python (not working with Kraken yet)
USEPYTHON=0


all: makemoviec


ifeq ($(USEKRAKEN),1)
# KRAKEN

# Currently, for full compatibility, should load PrgEnv-gnu and unload PrgEnv-intel
ifeq ($(USEMPI),0)
# mpicc should work perfectly with linking as long as all paths are set as if source'ed relevant lib paths, include paths, etc.  Should be done by system!
DFLAGS=-DUSINGMPI=$(USEMPI) -DUSINGPYTHON=$(USEPYTHON)
# -no-ipo
COMP=cc $(DFLAGS)
ifeq ($(USEPYTHON),1)
CFLAGS=-I/lustre/scratch/proj/sw/python-cnl/2.7.1/cnl3.1_gnu4.6.2/include/python2.7/
LDFLAGS=-L/lustre/scratch/proj/sw/python-cnl/2.7.1/cnl3.1_gnu4.6.2/lib/ -lpython2.7
else
CFLAGS=
LDFLAGS=
endif
CC=$(COMP) $(CFLAGS) $(DFLAGS)
MCC=$(CC)
LINKER=$(COMP) $(LDFLAGS)
endif



ifeq ($(USEMPITRYING),1)
# mpicc should work perfectly with linking as long as all paths are set as if source'ed relevant lib paths, include paths, etc.  Should be done by system!
DFLAGS=-DUSINGMPI=$(USEMPI) -DUSINGPYTHON=$(USEPYTHON)
#DFLAGS=-DUSINGMPI=0
# -no-ipo
COMP=gcc $(DFLAGS)
ifeq ($(USEPYTHON),1)
#CFLAGS=-I/sw/xt/python/2.7.1/sles11.1_gnu4.3.2/include/python2.7/
#CFLAGS=-I/lustre/scratch/proj/sw/python-cnl/2.7.1/cnl3.1_gnu4.6.2/include/python2.7/ -I/opt/cray/mpt/5.3.5/xt/seastar/mpich2-gnu/46/include
CFLAGS=-I/lustre/scratch/proj/sw/python-cnl/2.7.1/.cnl2.2_gnu4.5.3.static/include/python2.7/ -I/opt/cray/mpt/5.3.5/xt/seastar/mpich2-gnu/46/include
#CFLAGS=-I/usr/include/python/
#LDFLAGS=-lstdc++ -L/usr/lib/python2.6 -lpython2.6
#LDFLAGS=-L/sw/xt/python/2.7.1/sles11.1_gnu4.3.2/lib/ -lpython2.7
#LDFLAGS=-L/lustre/scratch/proj/sw/python-cnl/2.7.1/cnl3.1_gnu4.6.2/lib/ -lpython2.7 -L/opt/cray/mpt/5.3.5/xt/seastar/mpich2-gnu/46/lib -lmpich_gnu
LDFLAGS=-L/lustre/scratch/proj/sw/python-cnl/2.7.1/.cnl2.2_gnu4.5.3.static/lib/ -static -lpython2.7 -L/opt/cray/mpt/5.3.5/xt/seastar/mpich2-gnu/46/lib -lmpich_gnu
else
CFLAGS=-I/opt/cray/mpt/5.3.5/xt/seastar/mpich2-gnu/46/include
LDFLAGS=-L/opt/cray/mpt/5.3.5/xt/seastar/mpich2-gnu/46/lib -lmpich_gnu
endif
CC=$(COMP) $(CFLAGS) $(DFLAGS)
MCC=$(CC)
LINKER=$(COMP) $(LDFLAGS)
endif

# -dynamic doesn't work with Kraken, says not compiled for compute nodes!  Unless use module load python/2.7.1-cnl specifically
ifeq ($(USEMPIBAD),1)
DFLAGS=-DUSINGMPI=$(USEMPI) -DUSINGPYTHON=$(USEPYTHON)
COMP=cc -dynamic $(DFLAGS)
ifeq ($(USEPYTHON),1)
CFLAGS=-I/lustre/scratch/proj/sw/python-cnl/2.7.1/cnl3.1_gnu4.6.2/include/python2.7/
LDFLAGS=-lpython2.7
else
CFLAGS=
LDFLAGS=
endif
CC=$(COMP) $(CFLAGS) $(DFLAGS)
MCC=$(CC)
LINKER=$(COMP) $(LDFLAGS)
endif

# can use below if not using python
ifeq ($(USEMPI),1)
DFLAGS=-DUSINGMPI=$(USEMPI) -DUSINGPYTHON=$(USEPYTHON)
COMP=cc $(DFLAGS)
ifeq ($(USEPYTHON),1)
CFLAGS=-I/lustre/scratch/proj/sw/python-cnl/2.7.1/.cnl2.2_gnu4.5.3.static/include/python2.7/
LDFLAGS=-L/lustre/scratch/proj/sw/python-cnl/2.7.1/.cnl2.2_gnu4.5.3.static/lib/  -lpython2.7
else
CFLAGS=
LDFLAGS=
endif
CC=$(COMP) $(CFLAGS) $(DFLAGS)
MCC=$(CC)
LINKER=$(COMP) $(LDFLAGS)
endif

endif





ifeq ($(USENAUTILUS),1)
# NAUTILUS (works)

# -fast causes lack of proper link
ifeq ($(USEMPI),1)
DFLAGS=-DUSINGMPI=$(USEMPI) -DUSINGPYTHON=$(USEPYTHON)
COMP=icc $(DFLAGS)  -no-ipo
CFLAGS=-I/sw/analysis/python/2.7.1/sles11.1_intel11.1/include/python2.7/
# mkl and openmp are Intel specific on Nautilus
# for GNU must use -fopenmp and cannot use -lmkl
LDFLAGS=-lstdc++ -L/sw/analysis/python/2.7.1/sles11.1_intel11.1/lib/ -lpython2.7 -lmpi -lmkl -lopenmp
CC=$(COMP) $(CFLAGS) $(DFLAGS)
MCC=$(CC)
LINKER=$(CC) $(LDFLAGS)
endif
endif




ifeq ($(USEKIJMCK),1)

# no MPI option yet -- problems
ifeq ($(USEMPI),0)
DFLAGS=-DUSINGMPI=$(USEMPI) -DUSINGPYTHON=$(USEPYTHON)
COMP=icc $(DFLAGS)  -no-ipo
# mpicc should work perfectly with linking as long as all paths are set as if source'ed relevant lib paths, include paths, etc.  Should be done by system!
LDFLAGS= -lstdc++ -lpython2.7
# below -I for ki-jmck
CFLAGS=-O2 -Wall -g -DUSINGMPI=$(USEMPI) -DUSINGPYTHON=$(USEPYTHON) -I/usr/include/python2.7/

MCC=$(COMP)
LINKER=$(COMP) $(LDFLAGS)
endif



# NOT WORKING:
ifeq ($(USEMPI),1)
DFLAGS=-DUSINGMPI=$(USEMPI) -DUSINGPYTHON=$(USEPYTHON)
# /data1/jmckinne/mpich2-install/bin/mpicc uses gcc, but fails to add libraries with -l
COMP=icc $(DFLAGS)
#COMP=/usr/bin/mpicc $(DFLAGS)
#LDFLAGS=-lstdc++ -lpython2.7
# can't get mpich2 to work
#LDFLAGS=-L /data1/jmckinne/mpich2-install/lib/ -lopa -lmpl  -lrt -lpthread -lmpich -lpython2.7
# use /usr/lib/libmpi instead (but code crashes upon run)
LDFLAGS=-lmpi -lpython2.7
CFLAGS=-O2 -Wall -g -DUSINGMPI=$(USEMPI) -DUSINGPYTHON=$(USEPYTHON) -I/usr/include/python2.7/ -I/data1/jmckinne/mpich2-install/include/

MCC=$(COMP)
LINKER=$(COMP) $(LDFLAGS)
endif

endif



ifeq ($(USESTAMPEDE),1)
# -fast causes lack of proper link
ifeq ($(USEMPI),1)
DFLAGS=-DUSINGMPI=$(USEMPI) -DUSINGPYTHON=$(USEPYTHON)
COMP=mpicc $(DFLAGS)  -no-ipo
#CFLAGS=-I/sw/analysis/python/2.7.1/sles11.1_intel11.1/include/python2.7/
# mkl and openmp are Intel specific on Nautilus
# for GNU must use -fopenmp and cannot use -lmkl
#LDFLAGS=-lstdc++ -L/sw/analysis/python/2.7.1/sles11.1_intel11.1/lib/ -lpython2.7 -lmpi -lmkl -lopenmp
LDFLAGS=-lstdc++
CC=$(COMP) $(CFLAGS) $(DFLAGS)
MCC=$(CC)
LINKER=$(CC) $(LDFLAGS)
endif
endif



makemoviec: jon_makemovie_programstartc.o
	$(LINKER) -o makemoviec jon_makemovie_programstartc.o

clean:
	rm -f makemoviec jon_makemovie_programstartc.o

jon_makemovie_programstartc.o: jon_makemovie_programstartc.c Makefile
	$(MCC)  -c $(CFLAGS)  jon_makemovie_programstartc.c



# end of Makefile
