#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things




# ALL:
dircollectnontilt='thickdisk7 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskr7 thickdiskrr2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9full2pi sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'

dircollecttilt="sashaam9full2pit0.15 sashaa9b100t0.15 sashaa99t0.15 sashaam9full2pit0.3 sashaa9b100t0.3 sashaa99t0.3 sashaam9full2pit0.6 sashaa9b100t0.6 sashaa99t0.6 sashaam9full2pit1.5708 sashaa9b100t1.5708 sashaa99t1.5708 thickdiskfull3d7tilt0.35 thickdiskfull3d7tilt0.7 thickdiskfull3d7tilt1.5708"

#################
# choose:
dircollect=$dircollecttilt


# note that thickdisk1 is actually bad, so ignore it.
# can choose so do runs in different order than collection.
#dirruns=$dircollect
# do expensive thickdisk7 and sasha99 last so can test things
#dirruns='run.like8 thickdisk8 thickdisk11 thickdisk12 thickdisk13 thickdiskr7 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9full2pi sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

dirrunsnontilt='thickdisk7 run.like8 thickdisk8 thickdisk11 thickdisk12 thickdisk13 thickdiskr7 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9full2pi sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'

dirrunstilt="sashaam9full2pit0.15 sashaa9b100t0.15 sashaa99t0.15 sashaam9full2pit0.3 sashaa9b100t0.3 sashaa99t0.3 sashaam9full2pit0.6 sashaa9b100t0.6 sashaa99t0.6 sashaam9full2pit1.5708 sashaa9b100t1.5708 sashaa99t1.5708 thickdiskfull3d7tilt0.35 thickdiskfull3d7tilt0.7 thickdiskfull3d7tilt1.5708"

###################
# choose
dirruns=$dirrunstilt


#dirruns='thickdisk17'

#dirruns='thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3'

#dirruns='blandford3d_new'
#dirruns='thickdisk8'


#dirruns='sasham9full2pi sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'

#dirruns='sasha9b25'
#dirruns='sasha9b100'

#dirruns='thickdisk8'

#dirruns='run.like8'

#dirruns='thickdisk8'



#dirruns='thickdisk8'

#dirruns='thickdisk15 thickdiskr15 thickdisk2 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9 sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

#dirruns='thickdiskhr3'

#dirruns='thickdisk8'

#dirruns='sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

#dirruns='thickdisk7 sasha99'
#dirruns='runlocaldipole3dfiducial'
#dirruns='sasha99'



EXPECTED_ARGS=1
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` <moviedirname>"
    echo "e.g. sh collectframes.sh twoframesonly1"
    exit $E_BADARGS
fi




# /Luste/ki/orange/jmckinne/thickdisk7/movie6
# sh makemovie.sh thickdisk7 1 1 1 0 1 1 0 0 0 0 0 
# jobstokill=`bjobs -u jmckinne -q kipac-ibq | awk '{print $1}'`
# for fil in $jobstokill ; do bkill -r $fil ; done 

# 
# /u/ki/jmckinne/nfsslac2/thickdisk7/movie6b
# sh makemovie.sh thickdisk7 1 1 1 0 1 1 0 0 0 0 0 
# jobstokill=`bjobs -u jmckinne -q kipac-gpuq | awk '{print $1}'`
# for fil in $jobstokill ; do bkill -r $fil ; done 

# name of movie directory in each dirrun
moviedirname=$1

basedir=`pwd`

##############################################
#
echo "Now collect some data"

rm -rf globuslist.txt

for thedir in $dircollect
do
	echo "Doing remote collection for: "$thedir
    #
    list=`ls -v ${basedir}/${thedir}/$moviedirname/ | egrep 'lrho.*\.png' | grep -v "lrho0000_Rzxym1.png" | grep -v "lrhosmall0000_Rzxym1.png"`
    if [ 1 -eq 0 ]
    then
        first=`echo "$list" | head -2`
        last=`echo "$list" | tail -2`
        echo "first=$first  last=$last"
        allfiles="echo $first $last"
    fi

    # use -s 2 in case same file size
    for realfile in $list
    do
        echo "xsede#nautilus/${basedir}/${thedir}/$moviedirname/$realfile pseudotensor#ki-jmck/home/jmckinne/collect2frames/${thedir}/$moviedirname/$realfile" >> globuslist.txt
    done
done


ssh pseudotensor@cli.globusonline.org transfer -s 1 < globuslist.txt





echo "Done with all stages"
