#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things

# Steps:
#
# 1) Ensure followed createlinks.sh for each directory.
# 2) use the script

# order of models as to appear in final table

dirshow='thickdisk7 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3  thickdisk17 thickdisk10 thickdisk15 thickdisk15r thickdisk2 thickdisk3 runlocaldipole3dfiducial blandford3d_new sasham9 sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'



EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {moviedirname doshowmontage doshowmovie}"
    echo "e.g. sh showallmontage.sh moviefinal1 1 0"
    exit $E_BADARGS
fi


# name of movie directory in each dirshow
moviedirname=$1
doshowmontage=$2
doshowmovie=$3

# On ki-jmck in /data2/jmckinne/
cd /data2/jmckinne/


###################################
for thedir in $dirshow
do

    # goto movie directory
    fulldir=/data2/jmckinne/${thedir}/$moviedirname
    echo $fulldir
    cd $fulldir

    
    if [ $doshowmontage -eq 1 ]
    then
        echo "Doing montage for: "$thedir
        
        display -geometry 1800x1400 montage.png

        echo "Done with montage"
    fi

    if [ $doshowmovie -eq 1 ]
    then
        echo "Doing movie for: "$thedir
        
        mplayer -loop 0 lrho.avi

        echo "Done with movie"
    fi

done




echo "Done with all stages"
