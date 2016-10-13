#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things


###############################################################
# SPECIAL NOTES:
#
# 1) On kraken, have to have source ~/setuppython27 before running makeallmovie.sh or makemovie.sh so that makemoviec compiled properly and python run properly on compute nodes.
#   Can't source here because "module" program doesn't exist in bash
# 
###############################################################

#user="jmckinne"
#userbatch="jmckinn"
#emailaddr="pseudotensor@gmail.com"

# whether to wait and check on runs (if run many jobs, some head nodes won't fork that much)
dowait=0


user=$USER
userbatch=${USER:2:6}
emailaddr="pseudotensor@gmail.com"
ACCOUNT="TG-PHY120005"

EXPECTED_ARGS=18
E_BADARGS=65

#if [ $# -ne $EXPECTED_ARGS ]
if [ $# -lt $(($EXPECTED_ARGS)) ]
then
    echo "Usage: `basename $0` {modelname make1d makemerge makeplot makemontage makepowervsmplots makespacetimeplots makefftplot makespecplot makeinitfinalplot makethradfinalplot makeframes makemovie makeavg makeavgmerge makeavgplot} <system> <parallel> <dirname>"
    echo "only dirname is optional"
    echo "e.g. sh makemovie.sh thickdisk7 1 1 1 1 1 1 1 0 0 0 0    3 0 /data1/$user/thickdisk7/fulllatest14/"
    exit $E_BADARGS
fi

echo "HOSTNAME=$HOSTNAME"


modelname=$1
make1d=$2
makemerge=$3

makeplot=$4

makemontage=$5

# below several part of makeplot options
makepowervsmplots=${6}
makespacetimeplots=${7}
makefftplot=${8}
makespecplot=${9}
makeinitfinalplot=${10}
makethradfinalplot=${11}

makeframes=${12}
makemovie=${13}
makeavg=${14}
makeavgmerge=${15}
makeavgplot=${16}



system=${17}
parallel=${18}

# get optional dirname
if [ $# -eq $(($EXPECTED_ARGS+1))  ]
then
    dirname=${19}
else
    # assume just local directory if not given
    # should be full path
    dirname=`pwd`
fi



###########################################
#
# parameters one can set
#
###########################################

jobprefix=''
testrun=0
rminitfiles=0


# can run just certain runi values
useoverride=0
ilistoverride=`seq 24 31`
runnoverride=128
#############################################
# setup tasks, cores, and nodes for make2davg
#itemspergroup=$(( 1 )) # MAVARA
itemspergroup=$(( 4 ))


numfilesforavg=`find dumps/ -name "fieldline*.bin"|wc -l`

echo "NUMFILES=$numfilesforavg"

# catch too small number of files
# must match __init__.py
if [ $numfilesforavg -le $itemspergroup ]
then
    if [ $numfilesforavg -eq 1 ]
    then
        itemspergroup=1
    else
        itemspergroup=$(( $numfilesforavg - 1))
    fi
fi

itemspergrouptext=`printf "%02d"  "$itemspergroup"`




#########################################################################
# define unique suffix so know which thing is running in batch system
# below list obtained from __init__.py and then processed for bash
if [ $modelname == "thickdisk7" ]
then
    jobprefix="td7"
    jobsuffix="jy"       
elif [ $modelname == "thickdisk8" ]
then
    jobsuffix="jb"       
elif [ $modelname == "thickdisk11" ]
then
    jobsuffix="jc"       
elif [ $modelname == "thickdisk12" ]
then
    jobsuffix="jd"       
elif [ $modelname == "thickdisk13" ]
then
    jobsuffix="je"       
elif [ $modelname == "run.like8" ]
then
    jobsuffix="jf"       
elif [ $modelname == "thickdiskrr2" ]
then
    jobsuffix="jg"       
elif [ $modelname == "run.liker2butbeta40" ]
then
    jobsuffix="jh"       
elif [ $modelname == "run.liker2" ]
then
    jobsuffix="ji"       
elif [ $modelname == "thickdisk16" ]
then
    jobsuffix="jj"       
elif [ $modelname == "thickdisk5" ]
then
    jobsuffix="jk"       
elif [ $modelname == "thickdisk14" ]
then
    jobsuffix="jl"       
elif [ $modelname == "thickdiskr1" ]
then
    jobsuffix="jm"       
elif [ $modelname == "run.liker1" ]
then
    jobsuffix="jn"       
elif [ $modelname == "thickdiskr2" ]
then
    jobsuffix="jo"       
elif [ $modelname == "thickdisk9" ]
then
    jobsuffix="jp"       
elif [ $modelname == "thickdiskr3" ]
then
    jobsuffix="jq"       
elif [ $modelname == "thickdisk17" ]
then
    jobsuffix="jr"       
elif [ $modelname == "thickdisk10" ]
then
    jobsuffix="js"       
elif [ $modelname == "thickdisk15" ]
then
    jobsuffix="jt"       
elif [ $modelname == "thickdiskr15" ]
then
    jobsuffix="ju"       
elif [ $modelname == "thickdisk2" ]
then
    jobsuffix="jv"       
elif [ $modelname == "thickdisk3" ]
then
    jobsuffix="jw"       
elif [ $modelname == "thickdiskhr3" ]
then
    jobsuffix="jx"       
elif [ $modelname == "runlocaldipole3dfiducial" ]
then
    jobsuffix="ja"       
elif [ $modelname == "blandford3d_new" ]
then
    jobsuffix="ka"       
elif [ $modelname == "a0hr07" ]
then
    jobsuffix="kb"       
elif [ $modelname == "sasham9" ]
then
    jobsuffix="kc"       
elif [ $modelname == "sasham9full2pi" ]
then
    jobsuffix="kd"       
elif [ $modelname == "sasham5" ]
then
    jobsuffix="ke"       
elif [ $modelname == "sasham2" ]
then
    jobsuffix="kf"       
elif [ $modelname == "sasha0" ]
then
    jobsuffix="kg"       
elif [ $modelname == "sasha1" ]
then
    jobsuffix="kh"       
elif [ $modelname == "sasha2" ]
then
    jobsuffix="ki"       
elif [ $modelname == "sasha5" ]
then
    jobsuffix="kj"       
elif [ $modelname == "sasha9b25" ]
then
    jobsuffix="kk"       
elif [ $modelname == "sasha9b50" ]
then
    jobsuffix="kl"       
elif [ $modelname == "sasha9b100" ]
then
    jobsuffix="km"       
elif [ $modelname == "sasha9b200" ]
then
    jobsuffix="kn"       
elif [ $modelname == "sasha99" ]
then
    jobsuffix="jz"       
elif [ $modelname == "thickdiskr7" ]
then
    jobsuffix="ko"
elif [ $modelname == "sashaa99t0.15" ]
then
    jobsuffix="aa"
elif [ $modelname == "sashaa99t0.3" ]
then
    jobsuffix="ab"
elif [ $modelname == "sashaa99t0.6" ]
then
    jobsuffix="ac"
elif [ $modelname == "sashaa99t1.5708" ]
then
    jobsuffix="ad"
elif [ $modelname == "sashaa9b100t0.15" ]
then
    jobsuffix="ba"
elif [ $modelname == "sashaa9b100t0.3" ]
then
    jobsuffix="bb"
elif [ $modelname == "sashaa9b100t0.6" ]
then
    jobsuffix="bc"
elif [ $modelname == "sashaa9b100t1.5708" ]
then
    jobsuffix="bd"
elif [ $modelname == "sashaam9full2pit0.15" ]
then
    jobsuffix="ca"
elif [ $modelname == "sashaam9full2pit0.3" ]
then
    jobsuffix="cb"
elif [ $modelname == "sashaam9full2pit0.6" ]
then
    jobsuffix="cc"
elif [ $modelname == "sashaam9full2pit1.5708" ]
then
    jobsuffix="cd"
elif [ $modelname == "thickdiskfull3d7tilt0.35" ]
then
    jobsuffix="da"
elif [ $modelname == "thickdiskfull3d7tilt0.7" ]
then
    jobsuffix="db"
elif [ $modelname == "thickdiskfull3d7tilt1.5708" ]
then
    jobsuffix="dc"
elif [ $modelname == "rad1" ]
then
    jobsuffix="dd"
else
    jobsuffix="uk"
fi


# defaults
numtasksplot=1
runnplot=1
numnodesplot=1
numcorespernodeplot=1







# runn is number of runs (and in parallel, should be multiple of numcorespernode)




#############################################
# system dependent settings


##############################################
# orange
if [ $system -eq 1 ]
then
    # up to 768 cores (96*2*4).  That is, 8cores/node
    # but only 4GB/core.  Seems to work, but maybe much slower than would be if used 6 cores to allow 5.3G/core?
    # ok, now need 5 cores
    # ok, now 3 required for thickdisk7
    # need 2 for thickdisk3 where final qty file is 12GB right now
    numcorespernode=3
    #
    numnodes=$((180/$numcorespernode)) # so 180 cores total
    thequeue="kipac-ibq"
    # first part of name gets truncated, so use suffix instead for reliability no matter how long the names are
fi

##############################################
# orange-gpu
if [ $system -eq 2 ]
then
    numcorespernode=8
    numnodes=3
    # there are 16 cores, but have to use 8 for kipac-gpuq too since only 48GB memory and would use up to 5.3GB/core
    # only 3 free nodes for kipac-gpuq (rather than 4 since one is head node)
    thequeue="kipac-gpuq"
fi

#############################################
# ki-jmck
if [ $system -eq 3 ]
then
    # 4 for thickdisk7 (until new memory put in)
    numcorespernode=16  # MAVARA
    numnodes=1
    thequeue="none"
fi


#############################################
# Nautilus or Kraken (partially)
if [ $system -eq 4 ] ||
    [ $system -eq 5 ]
then
    # go to directory where "dumps" directory is
    # required for Nautilus, else will change to home directory when job starts
    numnodes=1
    thequeue="analysis"
    #
    if [ "$modelname" == "thickdisk7" ] ||
        [ "$modelname" == "thickdiskr7" ] ||
        [ "$modelname" == "thickdiskhr3" ]
    then
        # 24 hours is good enough for these if using 450 files (taking 18 hours for thickdisk7), but not much more.
        timetot="24:00:00"
        numcorespernode=80
        # grep "memoryusage" python*full.out | sed 's/memoryusage=/ /g'|awk '{print $2}' | sort -g
        # thickdisk7 needs at least 11GB/core according to memory usage print outs, so request 12GB
        # sasha99 needs 7GB/core, so give 8GB
        # sasha9b100 needs 4GB/core, so give 6GB
        # runs like sasha5 need 2GB/core, so give 4GB
        # This will increase the number of cores when qsub called, but numcorespernode is really how many tasks.
        memtot=$((16 + $numcorespernode * 14)) # so real number of cores charged will be >3X numcorespernode.
    elif [ "$modelname" == "sasha99" ]
    then
        # takes 3*6281/1578~12 hours for sasha99 movie
        timetot="24:00:00"
        numcorespernode=80
        memtot=$((8 + $numcorespernode * 8))
    elif [ "$modelname" == "sasham9full2pi" ] ||
        [ "$modelname" == "sasha9b100" ]
    then
        timetot="24:00:00"
        numcorespernode=80
        memtot=$((8 + $numcorespernode * 6))
    elif [ "$modelname" == "sasham9" ] ||
        [ "$modelname" == "sasham5" ] ||
        [ "$modelname" == "sasham2" ] ||
        [ "$modelname" == "sasha0" ] ||
        [ "$modelname" == "sasha1" ] ||
        [ "$modelname" == "sasha2" ] ||
        [ "$modelname" == "sasha5" ] ||
        [ "$modelname" == "sasha9b25" ] ||
        [ "$modelname" == "sasha9b50" ] ||
        [ "$modelname" == "sasha9b200" ] ||
        [ "$modelname" == "runlocaldipole3dfiducial" ] ||
        [ "$modelname" == "blandford3d_new" ] ||
        [ "$modelname" == "a0hr07" ]
    then
        timetot="24:00:00"
        numcorespernode=80
        memtot=$((4 + $numcorespernode * 4))
    elif [ "$modelname" == "sashaam9full2pit0.15" ] ||
        [ "$modelname" == "sashaa9b100t0.15" ] ||
        [ "$modelname" == "sashaa99t0.15" ] ||
        [ "$modelname" == "sashaam9full2pit0.3" ] ||
        [ "$modelname" == "sashaa9b100t0.3" ] ||
        [ "$modelname" == "sashaa99t0.3" ] ||
        [ "$modelname" == "sashaam9full2pit0.6" ] ||
        [ "$modelname" == "sashaa9b100t0.6" ] ||
        [ "$modelname" == "sashaa99t0.6" ] ||
        [ "$modelname" == "sashaam9full2pit1.5708" ] ||
        [ "$modelname" == "sashaa9b100t1.5708" ] ||
        [ "$modelname" == "sashaa99t1.5708" ]
    then
        numnodes=6 # overwrite numnodes to 6
        timetot="24:00:00" # if use numnodes=6, only need ~12 hours actually for 5194 images
        numcorespernode=80
        # for new script with reinterp3dspc, uses up to 10GB per core, so give 11GB per core
        memtot=$((11 + $numcorespernode * 11))
    elif [ "$modelname" == "thickdiskfull3d7tilt0.35" ] ||
        [ "$modelname" == "thickdiskfull3d7tilt0.7" ] ||
        [ "$modelname" == "thickdiskfull3d7tilt1.5708" ]
    then
        timetot="24:00:00"
        numcorespernode=80
        # give an extra 2GB/core for these models compared to without reinterp3dspc
        memtot=$((18 + $numcorespernode * 16))
    else
        # default for lower res thick disk poloidal and toroidal runs
        echo "Ended up in default for timetot, numcorespernode, and memtot in makemovie.sh"
        timetot="4:00:00"
        numcorespernode=1024
        memtot=$((4 + $numcorespernode * 2))
    fi
    #
    # for makeplot part or makeplotavg part
    numcorespernodeplot=1
    numnodesplot=1
    # new analysis can take a long time.
    timetotplot="8:00:00"
    timetotavgplot=$timetotplot
    # don't always need so much memory.
    memtotplot=32
    # interactive use for <1hour:
    # ipython -pylab -colors=LightBG
    #
    # long normal interactive job:
    # qsub -I -A $ACCOUNT -q analysis -l ncpus=8,walltime=24:00:00,mem=32GB
    # Note that this gives you a node for <=24 hours, so you can run 8 processes in parallel.  If you want to open a few xterm's from that window with (xterm &), in that terminal you will have to set the DISPLAY variable to whatever it was in the login node of nautilus (or else, the display variable is empty, and the new xterm windows refuse to spawn).
fi


#############################################
# stampede
if [ $system -eq 7 ]
then
    # go to directory where "dumps" directory is
    # required for Nautilus, else will change to home directory when job starts
    thequeue="normal"
    #
    # CHOOSE total time
    #timetot="4:30:00"
    #timetot="24:00:00"
    #
    # makeavg:
    #timetot="00:30:00" # makeavg2d takes 0:17:00 for 512 tasks for all files
    # make1d: 0:44:30 for numtasks=512 and all files.
    #timetot="01:00:00"
    #timetot="04:00:00" # took 4 hours
    # makemovie
    timetot="01:00:00"
    #
    # numtasks set equal to total number of time slices, so each task does only 1 fieldline file
    # CHOOSE below or set numtasks to some number <= number of field lines
    numtasks=`ls dumps/fieldline*.bin |wc -l`  # true total number of tasks
    #
    #numtasks=$(($numtasks/2))
    
    # for makeavg2d only needs 1 hour for 512 tasks
    #numtasks=512
    # for make1d only needs 1 hour for 512 tasks
    #numtasks=512
    # for makemovie only needs 1 hour for 210 tasks
    #numtasks=210
    #
    #
    numtaskscorr=$(($numtasks))
    # choose below number of cores per node (16 maximum for stampede, probably less if each fieldline file needs more memory than system has)
    # choose 2 because thickdisk7 needs 12GB/core and only have 32GB per node
    # stampede with radtma0.8 has resident max of 2GB/task, so can't quite have 16 tasks per node, so go with 14.
    numtaskspernode=16
    numtotalnodes=$((($numtaskscorr+$numtaskspernode-1)/$numtaskspernode))
    apcmd="ibrun "

    # -n $numtasks # how many actual MPI processes there are.
    # -N $numtotalnodes # number of nodes requested such that really have access to numnodes*16 total cores even if not using them.
    

    #################
    # setup fake setup
    numcorespernode=$numtasks
    numnodes=1
    #
    DATADIR=$dirname
    


    ##############################
    # setup plotting part
    if [ $parallel -ge 2 ]
    then
        numtasksplot=$numtaskspernode
    else
        numtasksplot=1
    fi



    numnodesplot=1
    numtotalnodesplot=1
    numcorespernodeplot=16
    # this gives 16GB free for plotting (temp vars + qty2.npy file has to be smaller than this or swapping will occur)
    numtotalcoresplot=$numcorespernodeplot
    thequeueplot="normal"  # try using this as "serial" instead of takes too long for makeplot or makeavgplot steps.
    apcmdplot="ibrun "
    # only took 6 minutes for thickdisk7 doing 458 files inside qty2.npy!  Up to death at point when tried to resample in time.
    timetotplot="1:00:00" # for normal can go up to 48 hours.  For serial up to 12 hours.
    timetotavgplot=$timetotplot


    #############################################
    # setup tasks, cores, and nodes for make2davg
    numtasksavg0=`ls dumps/fieldline*.bin |wc -l`  # true total number of tasks
    numtasksavg=$((($numtasksavg0)/($itemspergroup)))
    # +1 below for general case when above floors, if no remainder than +1 doesn't hurt -- just not optimal
    numtasksavg=$(($numtasksavg+1))
    numtaskscorravg=$(($numtasksavg))
    numtotalnodesavg=$((($numtaskscorravg+$numtaskspernode-1)/$numtaskspernode))
    echo "numtasksavg=$numtasksavg numtotalnodesavg=$numtotalnodesavg"
    apcmdavg="ibrun "


fi
#############################################
# pfe (very similar to stampede, but apcmd different and numtaskspernode can be >16 if not using model=san
if [ $system -eq 8 ]
then
    # go to directory where "dumps" directory is
    # required for Nautilus, else will change to home directory when job starts
    thequeue="normal"
    #
    # CHOOSE total time
    #timetot="4:30:00"
    #timetot="24:00:00"
    #
    # makeavg:
    #timetot="00:30:00" # makeavg2d takes 0:17:00 for 512 tasks for all files
    # make1d: 0:44:30 for numtasks=512 and all files.
    #timetot="01:00:00"
    #timetot="04:00:00" # took 4 hours

    if [ $make1d -ge 1 ]
    then
        # 50 files
        #timetot="00:20:00"
        # 500 files
        # even 1000 files with 512 tasks
        timetot="00:25:00"
    elif [ $makeframes -ge 1 ]
    then
        # 50 files
        #timetot="00:20:00"
        # 500 files
        timetot="00:50:00"
        # 1000 files and 512 tasks, only up to 20 minutes
        timetot="00:25:00"
    elif [ $makeavg -ge 1 ]
    then
        # even with 500 files or 1000 files(with 512 cores)
        timetot="00:15:00"
    else
        # shouldn't matter then as not calling expensive routie that uses this timetot
        timetot="00:50:00"
    fi
    #
    numfieldlinefiles=`ls dumps/fieldline*.bin |wc -l`  # true total number of fieldline files
    #
    # numtasks set equal to total number of time slices, so each task does only 1 fieldline file
    # CHOOSE below or set numtasks to some number <= number of field lines
    numtasks=`ls dumps/fieldline*.bin |wc -l`  # true total number of tasks
    numtasks=512
    #
    #numtasks=$(($numtasks/2))
    
    # for makeavg2d only needs 1 hour for 512 tasks
    #numtasks=512
    # for make1d only needs 1 hour for 512 tasks
    #numtasks=512
    # for makemovie only needs 1 hour for 210 tasks
    #numtasks=210
    #
    #
    numtaskscorr=$(($numtasks))
    # choose below number of cores per node (16 maximum for stampede, probably less if each fieldline file needs more memory than system has)
    # choose 2 because thickdisk7 needs 12GB/core and only have 32GB per node
    # stampede with radtma0.8 has resident max of 2GB/task, so can't quite have 16 tasks per node, so go with 14.
    numtaskspernode=16
    numtotalnodes=$((($numtaskscorr+$numtaskspernode-1)/$numtaskspernode))

    # -n $numtasks # how many actual MPI processes there are.
    # -N $numtotalnodes # number of nodes requested such that really have access to numnodes*16 total cores even if not using them.
    

    #################
    # setup fake setup
    numcorespernode=$numtasks
    numnodes=1
    #
    DATADIR=$dirname
    


    ##############################
    # setup plotting part
    if [ $parallel -ge 2 ]
    then
        numtasksplot=$numtaskspernode
    else
        numtasksplot=1
    fi


    numnodesplot=1
    numtotalnodesplot=1
    numcorespernodeplot=16
    # this gives 16GB free for plotting (temp vars + qty2.npy file has to be smaller than this or swapping will occur)
    numtotalcoresplot=$numcorespernodeplot
    thequeueplot="normal"  # try using this as "serial" instead of takes too long for makeplot or makeavgplot steps.

    # only took 6 minutes for thickdisk7 doing 458 files inside qty2.npy!  Up to death at point when tried to resample in time.
    #timetotplot="1:00:00" # for normal can go up to 48 hours.  For serial up to 12 hours.
    #timetotplot="0:45:00" # for normal can go up to 48 hours.  For serial up to 12 hours.
    # if stick to parallel==2 mode, then only need about 10-20 minutes depending upon resolution
    #
    #
    if [ $numfieldlinefiles -lt 200 ]
    then
        timetotplot="0:20:00"
        timetotavgplot=$timetotplot
    else # esp for jonharmrad17
        timetotplot="0:30:00"
        timetotavgplot=$timetotplot
    fi

    #############################################
    # setup tasks, cores, and nodes for make2davg
    numtasksavg0=`ls dumps/fieldline*.bin |wc -l`  # true total number of tasks
    numtasksavg=$((($numtasksavg0)/($itemspergroup)))
    # +1 below for general case when above floors, if no remainder than +1 doesn't hurt -- just not optimal
    numtasksavg=$(($numtasksavg+1))
    numtaskscorravg=$(($numtasksavg))
    numtotalnodesavg=$((($numtaskscorravg+$numtaskspernode-1)/$numtaskspernode))
    echo "numtasksavg=$numtasksavg numtotalnodesavg=$numtotalnodesavg"


    apcmd="mpiexec -np $numtaskscorr "
    apcmdplot="mpiexec -np $numtasksplot "
    apcmdavg="mpiexec -np $numtasksavg "


fi


# Nautilus fix
# this starts a bunch of single node jobs
if [ $system -eq 4 ]
then
    numcorespernodeeff=$(($memtot / 4))
fi

# Kraken
# this starts a bunch of single node jobs
if [ $system -eq 5 ] &&
    [ $parallel -eq 1 ]
then
    thequeue="small"
    timetot="14:00:00" # probably don't need all this is 1 task per fieldline file

    # Kraken only has 16GB per node of 12 cores
    # so determine how many nodes need based upon Nautilus/Kraken memtot above

    # numtasks set equal to total number of time slices, so each task does only 1 fieldline file
    numtasks=`ls dumps/fieldline*.bin |wc -l`
    memtotnaut=$memtot
    numcorespernodenaut=$numcorespernode

    # total memory required is old memtot/numcorespernode from Nautilus multiplied by total number of tasks for Kraken
    memtotpercore=$((1+$memtotnaut/$numcorespernodenaut))
    
    numcorespernode=$((16/$memtotpercore))

    if [ $numcorespernode -eq 0 ]
    then
        echo "Not enough memory per core to do anything"
        exit
    fi

    numnodes=$(($numtasks/$numcorespernode))
    # total number of cores used by these nodes
    #numtotalcores=$(($numnodes*12))
    # numtotalcores is currently what is allocated per node because each job is for each node
    numtotalcores=12


    # setup plotting part
    numnodesplot=1
    numcorespernodeplot=12
    # this gives 16GB free for plotting
    numtotalcoresplot=$numnodesplot
    thequeueplot="small"
    timetotplot="8:00:00"
    timetotavgplot=$timetotplot

    #############################################
    # setup tasks, cores, and nodes for make2davg
    numtasksavg0=`ls dumps/fieldline*.bin |wc -l`  # true total number of tasks
    numtasksavg=$((($numtasksavg0)/($itemspergroup)))
    # +1 below for general case when above floors, if no remainder than +1 doesn't hurt -- just not optimal
    numtasksavg=$(($numtasksavg+1))
    numtaskscorravg=$(($numtasksavg))
    numtotalnodesavg=$((($numtaskscorravg+$numtaskspernode-1)/$numtaskspernode))
    echo "numtasksavg=$numtasksavg numtotalnodesavg=$numtotalnodesavg"

    apcmd="aprun -n 1 -d 12 -cc none -a xt"
    apcmdplot="aprun -n 1 -d 12 -cc none -a xt"
    apcmdavg="aprun -n 1 -d 12 -cc none -a xt"


fi


# Kraken using makemoviec to start fake-MPI job
# this starts a single many-node-core job
# script views this as 1 node with many cores
if [ $system -eq 5 ] &&
    [ $parallel -ge 2 ]
then
    # Kraken only has 16GB per node of 12 cores
    # so determine how many nodes need based upon Nautilus/Kraken memtot above
    memtotnaut=$memtot
    numcorespernodenaut=$numcorespernode

    # numtasks set equal to total number of time slices, so each task does only 1 fieldline file
    numtasks=`ls dumps/fieldline*.bin |wc -l`

    # total memory required is old memtot/numcorespernode from Nautilus multiplied by total number of tasks for Kraken
    memtotpercore=$((1+$memtotnaut/$numcorespernodenaut))
    numcorespernode=$((16/$memtotpercore))

    if [ $numcorespernode -eq 0 ]
    then
        echo "Not enough memory per core to do anything"
        exit
    fi

    if [ $numcorespernode -eq 2 ] ||
        [ $numcorespernode -eq 4 ] ||
        [ $numcorespernode -eq 6 ] ||
        [ $numcorespernode -eq 8 ] ||
        [ $numcorespernode -eq 10 ] ||
        [ $numcorespernode -eq 12 ]
    then
        numcorespersocket=$(($numcorespernode/2))
        apcmd="aprun -n $numtasks -S $numcorespersocket"
    else
        # odd, so can't use socket version
        apcmd="aprun -n $numtasks -N $numcorespernode"
    fi

    if [ $numcorespernode -eq 12 ]
    then
	numnodes=$((0 + $numtasks/$numcorespernode))
    else
	numnodes=$((1 + $numtasks/$numcorespernode))
    fi

    numtotalcores=$(($numnodes * 12)) # always 12 for Kraken

    if [ $numtotalcores -le 512 ]
        then
        thequeue="small"
    elif [ $numtotalcores -le 8192 ]
        then
        thequeue="medium"
    elif [ $numtotalcores -le 49536 ]
        then
        thequeue="large"
    fi

    # GODMARK: 458 thickdisk7 files only took 1:45 on Kraken
    timetot="3:00:00" # probably don't need all this is 1 task per fieldline file

    echo "PART1: $numcorespernode $numcorespersocket $numnodes $numtotalcores $thequeue $timetot"
    echo "PART2: $apcmd"

    # setup fake setup
    numcorespernode=$numtasks
    numnodes=1
    #
    DATADIR=$dirname
    



    # setup plotting part
    numtasksplot=16 # cores in single node

    numnodesplot=1
    numcorespernodeplot=12
    # this gives 16GB free for plotting (temp vars + qty2.npy file has to be smaller than this or swapping will occur)
    numtotalcoresplot=$numcorespernodeplot
    thequeueplot="small"
    apcmdplot="aprun -n $numtasksplot"
    # only took 6 minutes for thickdisk7 doing 458 files inside qty2.npy!  Up to death at point when tried to resample in time.
    timetotplot="8:00:00"
    timetotavgplot=$timetotplot


    #############################################
    # setup tasks, cores, and nodes for make2davg
    numtasksavg0=`ls dumps/fieldline*.bin |wc -l`  # true total number of tasks
    numtasksavg=$((($numtasksavg0)/($itemspergroup)))
    # +1 below for general case when above floors, if no remainder than +1 doesn't hurt -- just not optimal
    numtasksavg=$(($numtasksavg+1))
    numtaskscorravg=$(($numtasksavg))
    numtotalnodesavg=$((($numtaskscorravg+$numtaskspernode-1)/$numtaskspernode))
    echo "numtasksavg=$numtasksavg numtotalnodesavg=$numtotalnodesavg"

    if [ $numcorespernode -eq 2 ] ||
        [ $numcorespernode -eq 4 ] ||
        [ $numcorespernode -eq 6 ] ||
        [ $numcorespernode -eq 8 ] ||
        [ $numcorespernode -eq 10 ] ||
        [ $numcorespernode -eq 12 ]
    then
        numcorespersocket=$(($numcorespernode/2))
        apcmdavg="aprun -n $numtasksavg -S $numcorespersocket"
    else
        # odd, so can't use socket version
        apcmdavg="aprun -n $numtasksavg -N $numcorespernode"
    fi


    echo "PART1P: $numcorespernodeplot $numnodesplot $numtotalcoresplot $thequeueplot $timetotplot $timetotavgplot"
    echo "PART2P: $apcmdplot"

fi






#### chunk lists
chunklisttype=0
chunklist=\"`seq -s " " 1 $numtasks`\"

chunklisttypeplot=0
chunklistplot=\"`seq -s " " 1 $numtasksplot`\"

chunklistavg=\"`seq -s " " 1 $numtasksavg`\"




#############################################
# physics-179
if [ $system -eq 6 ]
then
    numcorespernode=16
    numnodes=1
    thequeue="none"
fi




if [ $useoverride -eq 0 ]
then
    runnglobal=$(( $numcorespernode * $numnodes ))
    runnglobalplot=$(( $numcorespernodeplot * $numnodesplot ))
else
    runnglobal=${runnoverride}
    runnglobalplot=${runnoverride}
fi
echo "runnglobal=$runnglobal"


# for orange systems:
#http://kipac.stanford.edu/collab/computing/hardware/orange/overview
# 96 cnodes, 2 CPUs, 4cores/CPU = 768 cores.
#http://kipac.stanford.edu/collab/computing/docs/orange
#bqueues | grep kipac 
#kipac-ibq       125  Open:Active       -    -    1    -   254     0   254     0
#kipac-gpuq      122  Open:Active       -    -    -    -     0     0     0     0
#kipac-xocmpiq   121  Open:Active       -    -    -    -     0     0     0     0
#kipac-xocq      120  Open:Active       -    -    -    -    41     5    36     0
#kipac-testq      62  Open:Active       -    -    1    -     0     0     0     0
#kipac-ibidleq    15  Open:Active       -    -    1    -     0     0     0     0
#kipac-xocguestq  10  Open:Active       -    -    -    -     0     0     0     0
#

# If you want the movie to contain the bottom panel with Mdot
# vs. time, etc. you need to pre-generate the file, which I call
# qty2.npy: This file contains 1d information for every frame.  I
# generate qty2.npy by running generate_time_series().  


# 1) ensure binary files in place

# see scripts/createlinks.sh

# Sasha: I don't think you need to modify anything (unless you changed
# the meaning of columns in output files or added extra columns to
# fieldline files which would be interpreted by my script as 3 extra
# columns which I added to output gdetB^i's).

# The streamline code does not need gdet B^i's.  It can use B^i's.

# Requirements:
# A) Currently python script requires fieldline0000.bin to exist for getting parameters.  Can search/replace this name for another that actually exists if don't want to include that first file.

# Options: A)
#
# mklotsopanels() has hard-coded indexes of field line files that it
# shows in different panels.  What you see is python complaining it
# cannot find an element with index ti, which is one of the 4 indices
# defined above, findexlist=(0,600,1225,1369) I presume you have less
# than 1369 frames, hence there are problem.  Try resetting findexlist
# to (0,1,2,3), and hopefully the macro will work.

# Options: B)
#
#  In order to run mkavgfigs(), you need to have generated 2D
# average dump files beforehand (search for 2DAVG), created a combined
# average file (for which you run the script with 2DAVG section
# enabled, generate 2D average files, and then run the same script
# again with 3 arguments: start end step, where start and end set the
# starting and ending number of fieldline file group that you average
# over, and each group by default contains 20 fieldline files) and
# named it avg2d.npy .
#
#We can talk more about what you need to do in order to run each of
#the sections of the file.  I have not put any time into making those
#sections portable: as I explained, the code is quite raw!  You might
#want to familiarize yourself with the tutorial first (did it work out
#for you in the end?) and basic python operations.  I am afraid, in
#order to figure out what is going on the script file, you have to be
#able to read python and see what is done.
#
#One thing that can help you, is to enable python debugger.  For this, you run from inside ipython shell:
#
#pdb
#
#Then, whenever there is an error, you get into debugger, where you can evaluate variables, make plots, etc.

# 2) Change "False" to "True" in the section of __main__ that runs
# generate_time_series()


###################
#
# some python setup stuff
#
###################
# copy over python script path since supercomputers (e.g. Kraken) can't access home directory while running.
#rm -rf $dirname/py/
#echo "dirname=$dirname"
#cp -a $HOME/py $dirname/
#cd $dirname
#ln -s $dirname/py $dirname/py
# setup py path
MYPYTHONPATH=$dirname/py/
MREADPATH=$MYPYTHONPATH/mread/
initfile=$MREADPATH/__init__.py
echo "initfile=$initfile"
myrand=${RANDOM}
echo "RANDOM=$myrand"

# assumes chose correct python library setup and system in Makefile
if [ $parallel -ge 2 ]
then
    # create makemoviec for local use
    oldpath=`pwd`
    cd $MYPYTHONPATH/scripts/

    makemoviecfullfile=$MYPYTHONPATH/scripts/makemoviec
    if [ -e makemoviec ]
    then
        echo "nothing to do"
    else

        if [ $system -eq 3 ]
        then
            sed -e 's/USEKIJMCK=0/USEKIJMCK=1/g' -e 's/USEKRAKEN=1/USEKRAKEN=0/g' -e 's/USENAUTILUS=1/USENAUTILUS=0/g' -e 's/USEMPI=1/USEMPI=0/g' Makefile > Makefile.temp
            cp Makefile.temp Makefile
        elif [ $system -eq 4 ]
        then
            sed -e 's/USEKIJMCK=1/USEKIJMCK=0/g' -e 's/USEKRAKEN=1/USEKRAKEN=0/g' -e 's/USENAUTILUS=0/USENAUTILUS=1/g' -e 's/USEMPI=0/USEMPI=1/g' Makefile > Makefile.temp
            cp Makefile.temp Makefile
        elif [ $system -eq 5 ]
        then
            sed -e 's/USEKIJMCK=1/USEKIJMCK=0/g' -e 's/USEKRAKEN=0/USEKRAKEN=1/g' -e 's/USENAUTILUS=1/USENAUTILUS=0/g' -e 's/USEMPI=0/USEMPI=1/g' Makefile > Makefile.temp
            cp Makefile.temp Makefile
        elif [ $system -eq 6 ]
        then
            sed -e 's/USEKIJMCK=0/USEKIJMCK=1/g' -e 's/USEKRAKEN=1/USEKRAKEN=0/g' -e 's/USENAUTILUS=1/USENAUTILUS=0/g' -e 's/USEMPI=1/USEMPI=0/g' Makefile > Makefile.temp
            cp Makefile.temp Makefile
        elif [ $system -eq 7 ] # leave as default stampede
        then
            sed -e 's/USEKIJMCK=1/USEKIJMCK=0/g' -e 's/USEKRAKEN=1/USEKRAKEN=0/g' -e 's/USENAUTILUS=1/USENAUTILUS=0/g' -e 's/USESTAMPEDE=0/USESTAMPEDE=1/g' -e 's/USEMPI=0/USEMPI=1/g' Makefile > Makefile.temp
            cp Makefile.temp Makefile
        elif [ $system -eq 8 ] # leave as default stampede
        then
            sed -e 's/USEKIJMCK=1/USEKIJMCK=0/g' -e 's/USEKRAKEN=1/USEKRAKEN=0/g' -e 's/USENAUTILUS=1/USENAUTILUS=0/g' -e 's/USESTAMPEDE=0/USESTAMPEDE=1/g' -e 's/USEMPI=0/USEMPI=1/g' Makefile > Makefile.temp
            cp Makefile.temp Makefile
        else
            echo "Not setup for system=$sytem"
            exit
        fi

        make clean ; make
        chmod ug+rx $makemoviecfullfile
    fi
    cd $oldpath

fi



localpath=`pwd`


#avoid questions that would stall things
alias cp='cp'
alias rm='rm'
alias mv='mv'

############################
#
# Make time series (vs. t and vs. r)
#
############################

# runtypes:
# 0 : make1d
# 1 : makeavg
# 2 : makeframe
# 3 : makeplot
# 4 : makeavgplot
for runtypes in `seq 0 5`
do

    if [ $make1d -ge 1 ] &&
        [ $runtypes -eq 0 ]
    then
        runtype=3
        jobpre="md"
        myapcmd=$apcmd
        extracmdraw=""
        mynumtasks=$numtasks
        mynumtotalnodes=$numtotalnodes
        mynumcorespernode=$numcorespernode
        mythequeue=$thequeue
        mytimetot=$timetot
        mymemtot=$memtot
        mychunklisttype=$chunklisttype
        mychunklist=$chunklist
        myrunnglobal=$runnglobal
    elif [ $makeavg -ge 1 ] &&
        [ $runtypes -eq 1 ]
    then
        #takes average of $itemspergroup fieldline files per avg file created
        runtype=2
        jobpre="ma"
        myapcmd=$apcmdavg
        extracmdraw=$itemspergroup
        mynumtasks=$numtasksavg
        mynumtotalnodes=$numtotalnodesavg
        mynumcorespernode=$numcorespernode
        mythequeue=$thequeue
        mytimetot=$timetot
        mymemtot=$memtot
        mychunklisttype=$chunklisttype
        mychunklist=$chunklistavg
        myrunnglobal=$runnglobal
    elif [ $makeframes -ge 1 ] &&
        [ $runtypes -eq 2 ]
    then
        runtype=4
        jobpre="mv"
        myapcmd=$apcmd
        extracmdraw=""
        mynumtasks=$numtasks
        mynumtotalnodes=$numtotalnodes
        mynumcorespernode=$numcorespernode
        mythequeue=$thequeue
        mytimetot=$timetot
        mymemtot=$memtot
        mychunklisttype=$chunklisttype
        mychunklist=$chunklist
        myrunnglobal=$runnglobal
    elif [ $makeplot -ge 1 ] &&
        [ $runtypes -eq 3 ]
    then

        jobpre="pl"
        myapcmd=$apcmdplot
        extracmdraw="$makepowervsmplots $makespacetimeplots $makefftplot $makespecplot $makeinitfinalplot $makethradfinalplot"
        mynumtasks=$numtasksplot
        mynumtotalnodes=$numtotalnodesplot
        mynumcorespernode=$numcorespernodeplot
        mythequeue=$thequeueplot
        mytimetot=$timetotplot
        mymemtot=$memtotplot
        mychunklisttype=$chunklisttypeplot
        mychunklist=$chunklistplot

        
        if [ $makeplot -eq 1 ]
        then
            runtype=11 # non-parallel plotting mode
            # could do runtype==3 because args differentiate it in python script
            myrunnglobal=1
            # whichmode==0 in pyton script
        elif [ $parallel -ge 2 ] &&
            [ $makeplot -eq 100 ]
        then
            # then full parallel mode
            runtype=10
            myrunnglobal=$runnglobalplot
        else
            myrunnglobal=1
            # then assume don't want parallel mode and want to select certain plots to make
            if [ $makeplot -eq 2 ]
            then
                runtype=12
            fi
            if [ $makeplot -eq 3 ]
            then
                runtype=13
            fi
            if [ $makeplot -eq 4 ]
            then
                runtype=14
            fi
            if [ $makeplot -eq 5 ]
            then
                runtype=15
            fi
            if [ $makeplot -eq 6 ]
            then
                runtype=16
            fi
            if [ $makeplot -eq 7 ]
            then
                runtype=17
            fi
            if [ $makeplot -eq 8 ]
            then
                runtype=18
            fi
        fi
    elif [ $makeavgplot -ge 1 ] &&
        [ $runtypes -eq 4 ]
    then

        jobpre="pa"
        myapcmd=$apcmdplot
        extracmdraw=""
        mynumtasks=$numtasksplot
        mynumtotalnodes=$numtotalnodesplot
        mynumcorespernode=$numcorespernodeplot
        mythequeue=$thequeueplot
        mytimetot=$timetotavgplot
        mymemtot=$memtotplot
        mychunklisttype=$chunklisttypeplot
        mychunklist=$chunklistplot
        myrunnglobal=$runnglobal

        if [ $makeavgplot -eq 1 ]
        then
            runtype=21 # non-parallel plotting mode
            myrunnglobal=1
            # whichmode==0 in pyton script
        elif [ $parallel -ge 2 ] &&
            [ $makeavgplot -eq 100 ]
        then
            # then assume want parallel mode
            runtype=20
            myrunnglobal=$runnglobalplot
        else
            myrunnglobal=1
            # then assume don't want parallel mode and want to select
            if [ $makeavgplot -eq 2 ]
            then
                runtype=22
            fi
            if [ $makeavgplot -eq 3 ]
            then
                runtype=23
            fi
            if [ $makeavgplot -eq 4 ]
            then
                runtype=24
            fi
            if [ $makeavgplot -eq 5 ]
            then
                runtype=25
            fi
            if [ $makeavgplot -eq 6 ]
            then
                runtype=26
            fi
            if [ $makeavgplot -eq 7 ]
            then
                runtype=27
            fi
            if [ $makeavgplot -eq 8 ]
            then
                runtype=28
            fi
            if [ $makeavgplot -eq 9 ]
            then
                runtype=29
            fi
            if [ $makeavgplot -eq 10 ]
            then
                runtype=30
            fi
            if [ $makeavgplot -eq 11 ]
            then
                runtype=31
            fi
            if [ $makeavgplot -eq 12 ]
            then
                runtype=32
            fi
            if [ $makeavgplot -eq 13 ]
            then
                runtype=33
            fi
            if [ $makeavgplot -eq 14 ]
            then
                runtype=34
            fi
            if [ $makeavgplot -eq 15 ]
            then
                runtype=35
            fi
            if [ $makeavgplot -eq 16 ]
            then
                runtype=36
            fi
        fi

    else
        echo "skipping runtypes=$runtypes"
        continue
    fi

    echo "setup: runtypes=$runtypes runtype=$runtype jobpre=$jobpre"

    
    runn=${myrunnglobal}
    echo "runn=$runn"
    numparts=1


    myinitfile=$localpath/__init__.py.$runtype.$myrand
    echo "myinitfile="${myinitfile}
    cp $initfile $myinitfile

   
    je=$(( $numparts - 1 ))
    # above two must be exactly divisible
    itot=$(( $runn/$numparts ))
    echo "itot,runn,numparts: $itot $runn $numparts"
    ie=$(( $itot -1 ))

    resid=$(( $runn - $itot * $numparts ))
    
    echo "Running with $itot cores simultaneously"
    
    # LOOP:
    
    for j in `seq 0 $numparts`
    do

	    if [ $j -eq $numparts ]
	    then
	        if [ $resid -gt 0 ]
	        then
		        residie=$(( $resid - 1 ))
		        ilist=`seq 0 $residie`
		        doilist=1
	        else
		        doilist=0
	        fi
	    else
            if [ $useoverride -eq 1 ]
            then
                ilist=$ilistoverride
            else
	            ilist=`seq 0 $ie`
            fi
	        doilist=1
	    fi

	    if [ $doilist -eq 1 ]
	    then
            echo "runtype=$runtype Starting simultaneous run of $itot jobs for group $j"
            for i in $ilist
	        do

	      ############################################################
	      ############# BEGIN WITH RUN IN PARALLEL OR NOT

              # for parallel -ge 1, do every numcorespernode starting with i=0
	            modi=$(($i % $mynumcorespernode))

	            dowrite=0
	            #if [ $parallel -eq 0 ]
		        #then
		        #    dowrite=1
	            #else
		            if [ $modi -eq 0 ]
		            then
		                dowrite=1
		            fi
	            #fi

	            if [ $dowrite -eq 1 ]
		        then
                  # create script to be run
		            thebatch="sh${runtype}_python_${i}_${mynumcorespernode}_${runn}.sh"
		            rm -rf $thebatch
		            echo "j=$j" >> $thebatch
		            echo "itot=$itot" >> $thebatch
		            echo "i=$i" >> $thebatch
		            echo "runn=$runn" >> $thebatch
		            echo "runtype=$runtype" >> $thebatch
		            echo "extracmdraw=\"$extracmdraw\"" >> $thebatch
		            echo "mynumcorespernode=$mynumcorespernode" >> $thebatch
		            echo "mynumcorespernode=$mynumcorespernode" >> $thebatch
		            echo "system=$system" >> $thebatch
		            echo "parallel=$parallel" >> $thebatch
	            fi
	            
	            #if [ $parallel -eq 0 ]
		        #then
		        #    myruni='$(( $i + $itot * $j ))'
		        #    echo "cor=0" >> $thebatch
	            #else
		            echo "i=$i mynumcorespernode=$mynumcorespernode modi=$modi"
		            if [ $modi -eq 0 ]
		            then
 		                myruni='$(( $cor - 1 + $i + $itot * $j ))'
		                myseq='`seq 1 $mynumcorespernode`'
		                if [ $dowrite -eq 1 ]
			            then
			                echo "for cor in $myseq" >> $thebatch
			                echo "do" >> $thebatch
		                fi
		            fi
	            #fi

	            
	            if [ $dowrite -eq 1 ]
		        then
                    echo "cd $dirname" >> $thebatch
                    echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $thebatch
                    if [ $system -eq 4 ]
                    then
                        echo "unset MPLCONFIGDIR" >> $thebatch
                    else                        
                        rm -rf $dirname/matplotlibdir.$runtype/
                        echo "export MPLCONFIGDIR=$dirname/matplotlibdir.$runtype/" >> $thebatch
                    fi
		            echo "runi=$myruni" >> $thebatch
		            echo "textrun=\"runtype=\$runtype: Running i=\$i j=\$j giving runi=\$runi with runn=\$runn\"" >> $thebatch
		            echo "echo \$textrun" >> $thebatch
		            echo "sleep 1" >> $thebatch
		            cmdraw="python $myinitfile $system $parallel $runtype $modelname "'$runi $runn ${extracmdraw}'
		            cmdfull='((nohup $cmdraw 2>&1 1>&3 | tee python_${runi}_${cor}_${runn}.${runtype}.stderr.out) 3>&1 1>&2 | tee python_${runi}_${cor}_${runn}.${runtype}.out) > python_${runi}_${cor}_${runn}.${runtype}.full.out 2>&1'
		            echo "cmdraw=\"$cmdraw\"" >> $thebatch
		            echo "cmdfull=\"$cmdfull\"" >> $thebatch
		            echo "echo \"\$cmdfull\" > torun_$thebatch.\$cor.sh" >> $thebatch
		            echo "nohup sh torun_$thebatch.\$cor.sh &" >> $thebatch
	            fi


	            #if [ $parallel -ge 1 ]
		        #then
		            if [ $dowrite -eq 1 ]
		            then
		                echo "done" >> $thebatch
		            fi
	            #fi

	            if [ $dowrite -eq 1 ]
		        then
		            echo "wait" >> $thebatch
		            chmod a+x $thebatch
	            fi
	      #
	            if [ $parallel -eq 0 ]
		        then
		            if [ $dowrite -eq 1 ]
		            then
 		                if [ $testrun -eq 1 ]
		                then
		                    echo $thebatch
		                else
		                    sh ./$thebatch
		                fi
                    fi
	            else
		            if [ $dowrite -eq 1 ]
		            then
    	              # run bsub on batch file
                        jobcheck=${jobpre}.$jobsuffix
		                jobname=$jobprefix${i}${jobcheck}
		                outputfile=$jobname.${runtype}.out
		                errorfile=$jobname.${runtype}.err
                        rm -rf $outputfile
                        rm -rf $errorfile
                        #
                        if [ $system -eq 4 ]
                        then
		                    bsubcommand="qsub -S /bin/bash -A $ACCOUNT -l mem=${mymemtot}GB,walltime=$mytimetot,ncpus=$mynumcorespernodeeff -q $mythequeue -N $jobname -o $outputfile -e $errorfile -M $emailaddr ./$thebatch"
                        elif [ $system -eq 7 ]
                        then
                            # -n $mynumtasks # how many actual MPI processes there are.
                            # -N $mynumtotalnodes # number of nodes requested such that really have access to numnodes*16 total cores even if not using them.
                            superbatch=superbatchfile.$thebatch
                            rm -rf $superbatch
                            echo "#!/bin/bash" >> $superbatch
                            echo "cd $dirname" >> $superbatch
                            echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $superbatch
                            rm -rf $dirname/matplotlibdir.$runtype/
                            echo "export MPLCONFIGDIR=$dirname/matplotlibdir.$runtype/" >> $superbatch
		                    fakeruni=99999999999999
                            if [ $parallel -eq 1 ]
                            then
                                echo "$myapcmd ./$thebatch" >> $superbatch
                            else
		                        cmdraw="$makemoviecfullfile $mychunklisttype $mychunklist $runn $DATADIR $jobcheck $myinitfile $system $parallel $runtype $modelname $fakeruni $runn ${extracmdraw}"
                                echo "$myapcmd $cmdraw" >> $superbatch
                            fi
                            localerrorfile=python_${fakeruni}_${runn}.${runtype}.stderr.out
                            localoutputfile=python_${fakeruni}_${runn}.${runtype}.out
                            rm -rf $localerrorfile
                            rm -rf $localoutputfile
		                    bsubcommand="sbatch -A $ACCOUNT -t $mytimetot -p $mythequeue -n $mynumtasks -N $mynumtotalnodes -J $jobname -o $outputfile -e $errorfile --mail-user=$emailaddr --mail-type=begin --mail-type=end ./$superbatch"

                        elif [ $system -eq 5 ]
                        then
                            superbatch=superbatchfile.$thebatch
                            rm -rf $superbatch
                            echo "cd $dirname" >> $superbatch
                            cat ~/setuppython27 >> $superbatch
                            echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $superbatch
                            rm -rf $dirname/matplotlibdir.$runtype/
                            echo "export MPLCONFIGDIR=$dirname/matplotlibdir.$runtype/" >> $superbatch
		                    fakeruni=99999999999999
                            if [ $parallel -eq 1 ]
                            then
                                echo "$myapcmd ./$thebatch" >> $superbatch
                            else
		                        cmdraw="$makemoviecfullfile $mychunklisttype $mychunklist $runn $DATADIR $jobcheck $myinitfile $system $parallel $runtype $modelname $fakeruni $runn ${extracmdraw}"
                                echo "$myapcmd $cmdraw" >> $superbatch
                            fi
                            localerrorfile=python_${fakeruni}_${runn}.${runtype}.stderr.out
                            localoutputfile=python_${fakeruni}_${runn}.${runtype}.out
                            rm -rf $localerrorfile
                            rm -rf $localoutputfile
		                    bsubcommand="qsub -S /bin/bash -A $ACCOUNT -l walltime=$mytimetot,size=$numtotalcores -q $mythequeue -N $jobname -o $localoutputfile -e $localerrorfile -M $emailaddr -m be ./$superbatch"
                        elif [ $system -eq 8 ]
                        then
                            superbatch=superbatchfile.$thebatch
                            rm -rf $superbatch
                            echo "cd $dirname" >> $superbatch
                            #cat ~/setuppython27 >> $superbatch
                            echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $superbatch
                            rm -rf $dirname/matplotlibdir.$runtype/
                            echo "export MPLCONFIGDIR=$dirname/matplotlibdir.$runtype/" >> $superbatch
		                    fakeruni=99999999999999
                            if [ $parallel -eq 1 ]
                            then
                                echo "$myapcmd ./$thebatch" >> $superbatch
                            else
		                        cmdraw="$makemoviecfullfile $mychunklisttype $mychunklist $runn $DATADIR $jobcheck $myinitfile $system $parallel $runtype $modelname $fakeruni $runn ${extracmdraw}"
                                echo "$myapcmd $cmdraw" >> $superbatch
                            fi
                            localerrorfile=python_${fakeruni}_${runn}.${runtype}.stderr.out
                            localoutputfile=python_${fakeruni}_${runn}.${runtype}.out
                            rm -rf $localerrorfile
                            rm -rf $localoutputfile
		                    bsubcommand="qsub -S /bin/bash -l walltime=$mytimetot,select=$mynumtotalnodes:ncpus=16:model=san -W group_list=s1497 -q $mythequeue -N $jobname -o $localoutputfile -e $localerrorfile -M $emailaddr -m be ./$superbatch"
                        else
                            # probably specifying ptile below is not necessary
		                    bsubcommand="bsub -n 1 -x -R span[ptile=$mynumcorespernode] -q $mythequeue -J $jobname -o $outputfile -e $errorfile ./$thebatch"
                        fi
                        #
		                if [ $testrun -eq 1 ]
			            then
			                echo $bsubcommand
		                else
			                echo $bsubcommand
			                echo "$bsubcommand" > bsubshtorun_${runtype}_$thebatch
			                chmod a+x bsubshtorun_${runtype}_$thebatch
			                sh bsubshtorun_${runtype}_$thebatch
		                fi
		            fi
	            fi

	      ############# END WITH RUN IN PARALLEL OR NOT
	      ############################################################
	            
	        done

	    # waiting game
            if [ $dowait -eq 1 ]
            then
	        if [ $parallel -eq 0 ]
		    then
		        wait
                
	        else
                # Wait to move on until all jobs done
		        totaljobs=$runn
		        firsttimejobscheck=1
		        while [ $totaljobs -gt 0 ]
		        do
                    if [ $system -eq 7 ]
                    then
		                pendjobs=`showq -u $user | grep $userbatch | grep -v "SUMMARY OF JOBS"  2>&1 | grep $jobcheck | grep $userbatch | grep "Waiting" | wc -l`
		                runjobs=`showq -u $user | grep $userbatch | grep -v "SUMMARY OF JOBS" 2>&1 | grep $jobcheck | grep $userbatch | grep "Running" | wc -l`
                    elif [ $system -eq 4 ] ||
                        [ $system -eq 5 ] ||
                        [ $system -eq 8 ]
                    then
		                pendjobs=`qstat -e $mythequeue 2>&1 | grep $jobcheck | grep $userbatch | grep " Q " | wc -l`
		                runjobs=`qstat -e $mythequeue 2>&1 | grep $jobcheck | grep $userbatch | grep " R " | wc -l`
                    else
		                pendjobs=`bjobs -u all -q $mythequeue 2>&1 | grep $jobcheck | grep $userbatch | grep PEND | wc -l`
		                runjobs=`bjobs -u all -q $mythequeue 2>&1 | grep $jobcheck | grep $userbatch | grep RUN | wc -l`
                    fi
		            totaljobs=$(($pendjobs+$runjobs))
		            
		            if [ $totaljobs -gt 0 ]
		            then
		                echo "PEND=$pendjobs RUN=$runjobs TOTAL=$totaljobs ... waiting ..."
		                sleep 300
                        firsttimejobscheck=0
		            else
		                if [ $firsttimejobscheck -eq 1 ]
			            then
			                totaljobs=$runn
			                echo "waiting for jobs to get started..."
			                sleep 300
		                else
			                echo "DONE!"		      
		                fi
		            fi
		        done

	        fi
            fi

	        echo "runtype=$runtype: Ending simultaneous run of $itot jobs for group $j"
	    fi
    done
    
    if [ $dowait -eq 1 ]
    then
        wait
        if [ $rminitfiles -eq 1 ]
	    then
            # remove created file
	        rm -rf $myinitfile
        fi
    fi


done








###################################
#
# Merge npy files
#
####################################
if [ $makemerge -eq 1 ]
then

    # runn should be same as when creating files
    runn=${runnglobal}


    myinitfile2=$localpath/__init__.py.2.$myrand
    echo "myinitfile2="${myinitfile2}
    runtype=3
    cp $initfile $myinitfile2


    echo "Merge to single file"
    if [ $testrun -eq 1 ]
	then
	    echo "((nohup python $myinitfile2 $system $parallel $runtype $modelname $runn $runn 2>&1 1>&3 | tee python_${runn}_${runn}.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.out) > python_${runn}_${runn}.full.out 2>&1"
    else
	    ((nohup python $myinitfile2 $system $parallel $runtype $modelname $runn $runn 2>&1 1>&3 | tee python_${runn}_${runn}.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.out) > python_${runn}_${runn}.full.out 2>&1
    fi

    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile2
    fi

fi




###################################
#
# Make plots that depend upon makeplot being done
#
####################################
#########################################################
makemontage=$makemontage
if [ $makemontage -eq 1 ]
then

    # create montage of t vs. r and t vs. h plots
    files=`ls -rt plot*.png`
    montage -geometry 300x600 $files montage_plot.png
    # use -density to control each image size
    # e.g. default for montage is:
    # montage -density 72 $files montage.png
    # but can get each image as 300x300 (each tile) if do:
    # montage -geometry 300x300 -density 300 $files montage.png
    #
    # to display, do:
    # display montage.png
    # if want to have smaller window and pan more do (e.g.):
    # display -geometry 1800x1400 montage.png

    files=`ls -rt powervsm*.png`
    montage -geometry 500x500 $files montage_powervsm.png

fi





###################################
#
# MOVIE File
#
####################################
if [ $makemovie -eq 1 ]
then

    #  now can create an avi with:
    
    if [ $testrun -eq 1 ]
	then
	    echo "make movie files"
    else

        fps=25
        #
        #ffmpeg -i lrho%04d_Rzxym1.png -r $fps lrho.mp4
        #ffmpeg -fflags +genpts -i lrho%04d_Rzxym1.png -r $fps -sameq lrho.$modelname.avi

	    if [ 1 -eq 0 ]
	    then
        # high quality 1 minute long no matter what framerate (-t 60 doesn't work)
	        ffmpeg -y -fflags +genpts -i lrho%04d_Rzxym1.png -r 25 -sameq -qmax 5 -vcodec mjpeg lrho25.$modelname.avi 
        # now set frame rate (changes duration)
	        ffmpeg -y -i lrho25.$modelname.avi -f image2pipe -vcodec copy - </dev/null | ffmpeg -r $fps -f image2pipe -vcodec mjpeg -i - -vcodec copy -an lrho.$modelname.avi
	        
        # high quality 1 minute long no matter what framerate (-t 60 doesn't work)
	        ffmpeg -y -fflags +genpts -i lrhosmall%04d_Rzxym1.png -r 25 -sameq -qmax 5 -vcodec mjpeg lrhosmall25.$modelname.avi 
        # now set frame rate (changes duration)
	        ffmpeg -y -i lrhosmall25.$modelname.avi -f image2pipe -vcodec copy - </dev/null | ffmpeg -r $fps -f image2pipe -vcodec mjpeg -i - -vcodec copy -an lrhosmall.$modelname.avi
	    fi

        if [ 1 -eq 0 ]
        then
        # Sasha's command:
	        ffmpeg -y -fflags +genpts -r $fps -i lrho%04d_Rzxym1.png -vcodec mpeg4 -qscale 0 lrho.$modelname.avi
	        ffmpeg -y -fflags +genpts -r $fps -i lrhosmall%04d_Rzxym1.png -vcodec mpeg4 -qscale 0 lrhosmall.$modelname.avi
	        
        # for Roger (i.e. any MAC)
	        #ffmpeg -y -fflags +genpts -r $fps -i lrho%04d_Rzxym1.png -sameq -qmax 5 lrho.$modelname.mov
	        #ffmpeg -y -fflags +genpts -r $fps -i lrhosmall%04d_Rzxym1.png -sameq -qmax 5 lrhosmall.$modelname.mov
	    fi
        
        if [ 1 -eq 0 ]
        then
            cp ~/py/scripts/makelinkimagenew2.sh .
            sh ./makelinkimagenew2.sh
        fi

        if [ 1 -eq 1 ]
        then
            cp ~/py/scripts/makelinkimagenew3.sh .
            sh ./makelinkimagenew3.sh $modelname
        fi


    fi


#	        ffmpeg -y -fflags +genpts -r $fps -i initfinal%04d.png -vcodec mpeg4 -sameq -qmax 5 initfinal.$modelname.avi
#	        ffmpeg -y -fflags +genpts -r $fps -i stream%04d.png -vcodec mpeg4 -sameq -qmax 5 stream.$modelname.avi



    echo "Now do: mplayer -loop 0 lrho.$modelname.avi OR lrhosmall.$modelname.avi"

fi







###################################
#
# Merge avg npy files
#
####################################
if [ $makeavgmerge -eq 1 ]
then

    # should be same as when creating avg files
    runn=${runnglobal}


    myinitfile6=$localpath/__init__.py.6.$myrand
    echo "myinitfile6="${myinitfile6}

    runtype=2
    cp $initfile $myinitfile6


    echo "Merge avg files to single avg file"
    # <index of first avg file to use> <index of last avg file to use> <step=1>
    # must be step=1, or no merge occurs
    step=1
    whichgroups=$(( 0 ))
    numavg2dmerge=`ls -vrt | egrep "avg2d"${itemspergrouptext}"_[0-9]*\.npy"|wc|awk '{print $1}'`
    #whichgroupe=$(( $itemspergroup * $runn ))
    whichgroupe=$numavg2dmerge

    groupsnum=`printf "%04d" "$whichgroups"`
    groupenum=`printf "%04d" "$whichgroupe"`

    echo "GROUPINFO: $step $itemspergroup $whichgroups $whichgroupe $groupsnum $groupenum"

    if [ $testrun -eq 1 ]
	then
	    echo "((nohup python $myinitfile6 $system $parallel $runtype $modelname $whichgroups $whichgroupe $step 2>&1 1>&3 | tee python_${runn}_${runn}.avg.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.avg.out) > python_${runn}_${runn}.avg.full.out 2>&1"
    else
	    ((nohup python $myinitfile6 $system $parallel $runtype $modelname $whichgroups $whichgroupe $step $itemspergroup 2>&1 1>&3 | tee python_${runn}_${runn}.avg.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.avg.out) > python_${runn}_${runn}.avg.full.out 2>&1

        # copy resulting avg file to avg2d.npy
	    avg2dmerge=`ls -vrt avg2d${itemspergrouptext}_${groupsnum}_${groupenum}.npy | head -1`    
	    cp $avg2dmerge avg2d.npy

    fi

    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile6
    fi




fi





# to clean-up bad start, use:
# rm -rf sh*.sh bsub*.sh __init* py*.out torun*.sh j1*.err j1*.out *.npy
#
