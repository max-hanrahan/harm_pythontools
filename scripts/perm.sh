#!/bin/bash
#give others permissions to data dirs

# same as dircollect in makeallmovie.sh
#dirperm='thickdisk7part1 thickdisk7part2 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskr7 thickdiskrr2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9full2pi sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'
dirperm='thickdiskr15'

alias ls='ls'
alias lsd='ls -d */'
alias lsdir='ls -la | egrep "^d"'
alias lsh='ls -Flagt $@ | head'
alias lssdir='ls -ap | grep / | sed "s/\///"'
alias lssdir2='ls -ap| grep / | tail -n +3 | sed "s/\///"'
alias dudirs='for fil in `lssdir2`; do du -s $fil; done'
alias dud='dudirs | sort -n'

overalldirs='/data2/jmckinne/ /data1/jmckinne/'

for overalldir in $overalldirs
do
    cd $overalldir
    chmod a+rx .
    dirbase=`pwd`

    for thedir0 in $dirperm
    do
        odddir=0
        thedir=$thedir0
        if [ "$thedir0" == "thickdisk7part1" ]
        then
            thedir="thickdisk7"
            odddir=1
        elif [ "$thedir0" == "thickdisk7part2" ]
        then
            thedir="thickdisk7"
        fi


        if [ $odddir -eq 1 ]
        then
            cd $thedir
        fi

        dirorig=`pwd`
        echo "doing dirperm=$thedor0 with dir=$thedir at dirorig=$dirorig"

        
        rm -rf dirs${thedir}.txt
        list0=`lssdir  | grep ${thedir} | grep -v nextnextnextnext | grep -v nextnextnext | grep -v nextnext | grep -v next`
        list1=`lssdir  | grep ${thedir} | grep next | grep -v nextnextnextnext | grep -v nextnextnext | grep -v nextnext`
        list2=`lssdir  | grep ${thedir} | grep nextnext | grep -v nextnextnextnext | grep -v nextnextnext`
        list3=`lssdir  | grep ${thedir} | grep nextnextnext | grep -v nextnextnextnext`
        list4=`lssdir  | grep ${thedir} | grep nextnextnextnext`
        listfinal=`echo $list0 $list1 $list2 $list3 $list4`
        list=`echo $listfinal | sed 's/'${thedir}'\///g' | sed 's/'${thedir}' //g'`
        
        listfinalnum=`echo $listfinal | wc -w`
        listnum=`echo $list | wc -w`
        
        echo "begin echo of list"
        echo $list
        echo "end echo of list"
        for fil in $list ; do echo $fil >> dirs${thedir}.txt ; done


        predir=`pwd`
        
        if [ 1 -eq 1 ]
        then
            for mydir in `cat $dirorig/dirs${thedir}.txt`
            do
                echo $mydir
                cd $mydir

                chmod a+rx .
                chmod og-w .
                chmod a+r coordparms* nprlist*
                chmod og-wx coordparms* nprlist*

                chmod a+rx dumps
                chmod og-w dumps
                chmod a+r dumps/fieldline*
                chmod og-wx dumps/fieldline*
                chmod a+r dumps/gdump*
                chmod og-wx dumps/gdump*
                chmod a+r dumps/dump0000*
                chmod og-wx dumps/dump0000*
                
                cd $predir
            done
        fi
        
        cd $dirbase
        
    done
done
