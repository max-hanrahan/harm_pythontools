#!/bin/bash
#clean-up old analysis directories

dirclean1='thickdisk7 sasham9full2pi sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'

dirclean2='thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new'

alias lsd='ls -d */'
alias lsdir='ls -la | egrep "^d"'
alias lsh='ls -Flagt $@ | head'
alias lssdir='ls -ap | grep / | sed "s/\///"'
alias lssdir2='ls -ap| grep / | tail -n +3 | sed "s/\///"'
alias dudirs='for fil in `lssdir2`; do du -s $fil; done'
alias dud='dudirs | sort -n'


if [ 1 -eq 1 ]
then
    cd /data2/jmckinne/
    for blob in $dirclean2
    do
        dirorig=`pwd`
        echo "doing dir=$blob"
        cd $blob
        tokill=`lssdir | grep -v fulllatest14 | grep -v fulllatest15`
        for fil in $tokill
        do
            echo "Removing: $fil"
            rm -rf $fil
        done
        cd $dirorig
    done
fi


if [ 1 -eq 1 ]
then
    cd /data2/jmckinne/
    for blob in $dirclean1
    do
        dirorig=`pwd`
        echo "doing dir=$blob"
        cd $blob
        tokill=`lssdir | grep -v fulllatest13 | grep -v fulllatest14`
        echo "tokill=$tokill"
        for fil in $tokill
        do
            echo "Removing: $fil"
            rm -rf $fil
        done
        cd $dirorig
    done
fi

