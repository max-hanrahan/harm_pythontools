#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things

# Steps:
#
# 1) Ensure followed createlinks.sh for each directory.
# 2) use the script

# order of models as to appear in final table
dirshow='thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3  thickdisk17 thickdisk10 thickdisk15 thickdisk15r thickdisk2 thickdisk3 runlocaldipole3dfiducial blandford3d_new sasham9 sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'



EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {moviedirname doshowfft doshowspec}"
    echo "e.g. sh showallfft.sh moviefinal1 1 0"
    exit $E_BADARGS
fi


# name of movie directory in each dirshow
moviedirname=$1
doshowfft=$2
doshowspec=$3

# On ki-jmck in /data2/jmckinne/
cd /data2/jmckinne/


###################################
for thedir in $dirshow
do

    # goto movie directory
    fulldir=/data2/jmckinne/${thedir}/$moviedirname
    echo $fulldir
    cd $fulldir

    
    if [ $doshowfft -eq 1 ]
    then
        echo "Doing fft for: "$thedir
        
        #display -resize 1024x1024 fft.png
        display -geometry 1600x1200 fft1.png

        echo "Done with fft"
    fi

    if [ $doshowspec -eq 1 ]
    then
        echo "Doing spec for: "$thedir
        
        #display -resize 1024x1024 spec.png
        display -geometry 1600x1200 plotspec1.png

        echo "Done with spec"
    fi


done




echo "Done with all stages"
