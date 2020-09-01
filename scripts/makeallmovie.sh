#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things


user="megan"
#userbatch="jmckinn"
emailaddr="meganmarshall20@gmail.com"
#remotehost="ki-jmck.slac.stanford.edu"
#globushost="pseudotensor#ki-jmck"
#globusremote="pseudotensor@cli.globusonline.org"

# note that ubuntu defaults to dash after update.  Also causes \b to appear as ^H unlike bash.  Can't do \\b either -- still appears as ^H

# Steps:
#
# 1) Ensure followed createlinks.sh for each directory.
# 2) use the script

# thickdisk7 on Nautilus:
#
# sh makeallmovie.sh fulllatest14 1 1 1 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 /lustre/medusa/$user/data1/$user/$user/



# order of models as to appear in final table
# SKIP thickdisk15 thickdisk2 since not run long enough

# POLOIDAL:
#dircollect='thickdisk7 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 runlocaldipole3dfiducial blandford3d_new sasham9 sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'
# TOROIDAL:
#dircollecttoroidal='thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3'

# ALL:
dircollectnontilt='thickdisk7 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskr7 thickdiskrr2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9full2pi sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'

dircollecttilt="thickdiskfull3d7tilt0.35 thickdiskfull3d7tilt0.7 thickdiskfull3d7tilt1.5708 sashaam9full2pit0.15 sashaa9b100t0.15 sashaa99t0.15 sashaam9full2pit0.3 sashaa9b100t0.3 sashaa99t0.3 sashaam9full2pit0.6 sashaa9b100t0.6 sashaa99t0.6 sashaam9full2pit1.5708 sashaa9b100t1.5708 sashaa99t1.5708"


#################
# choose:
dircollect="rad1"


# note that thickdisk1 is actually bad, so ignore it.
# can choose so do runs in different order than collection.
#dirruns=$dircollect
# do expensive thickdisk7 and sasha99 last so can test things
#dirruns='run.like8 thickdisk8 thickdisk11 thickdisk12 thickdisk13 thickdiskr7 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9full2pi sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

dirrunsnontilt='thickdisk7 run.like8 thickdisk8 thickdisk11 thickdisk12 thickdisk13 thickdiskr7 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9full2pi sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'

dirrunstilt="sashaam9full2pit0.15 sashaa9b100t0.15 sashaa99t0.15 sashaam9full2pit0.3 sashaa9b100t0.3 sashaa99t0.3 sashaam9full2pit0.6 sashaa9b100t0.6 sashaa99t0.6 sashaam9full2pit1.5708 sashaa9b100t1.5708 sashaa99t1.5708 thickdiskfull3d7tilt0.35 thickdiskfull3d7tilt0.7 thickdiskfull3d7tilt1.5708"

###################
# choose
dirruns="rad1"


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

# number of files to keep
numkeep=50 # test


EXPECTED_ARGS=20
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {moviedirname docleanexist dolinks dofiles make1d makemerge makeplot makemontage makepowervsmplots makespacetimeplots makefftplot makespecplot makeinitfinalplot makethradfinalplot makeframes makemovie makeavg makeavgmerge makeavgplot collect} <full path dirname>"
    echo "e.g. sh makeallmovie.sh moviefinal1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 /data2/$user/"
    exit $E_BADARGS
fi




# /Luste/ki/orange/$user/thickdisk7/movie6
# sh makemovie.sh thickdisk7 1 1 1 0 1 1 0 0 0 0 0 
# jobstokill=`bjobs -u $user -q kipac-ibq | awk '{print $1}'`
# for fil in $jobstokill ; do bkill -r $fil ; done 

# 
# /u/ki/$user/nfsslac2/thickdisk7/movie6b
# sh makemovie.sh thickdisk7 1 1 1 0 1 1 0 0 0 0 0 
# jobstokill=`bjobs -u $user -q kipac-gpuq | awk '{print $1}'`
# for fil in $jobstokill ; do bkill -r $fil ; done 

# name of movie directory in each dirrun
moviedirname=$1
docleanexist=$2
dolinks=$3
dofiles=$4
make1d=$5
makemerge=$6
makeplot=$7
makemontage=$8
makepowervsmplots=$9
makespacetimeplots=${10}
makefftplot=${11}
makespecplot=${12}
makeinitfinalplot=${13}
makethradfinalplot=${14}
makeframes=${15}
makemovie=${16}
makeavg=${17}
makeavgmerge=${18}
makeavgplot=${19}
collect=${20}


if [ $# -eq $(($EXPECTED_ARGS+1)) ]
then
    dirname=${21}
else
    # On ki-jmck in /data2/$user/
    # assume run from /data2/$user or wherever full list of links/dirs are.
    dirname=`pwd`
fi



#based upon head node name
isorange=`echo $HOSTNAME | grep "orange" | wc -l`
isorangegpu=`echo $HOSTNAME | grep "orange-gpu" | wc -l`
isjmck=`echo $HOSTNAME | grep "ki-jmck" | wc -l`
isnautilus=`echo $HOSTNAME | egrep 'conseil|arronax' | wc -l`
iskraken=`echo $HOSTNAME | grep "kraken" | wc -l`
isphysics179=`echo $HOSTNAME | grep "physics-179" | wc -l`
isstampede=`echo $HOSTNAME | grep "stampede" | wc -l`

# parallel>=1 sets to use batch sysem if long job *and* if multiple jobs then submit all of them to batch system
# 1 = orange
# 2 = orange-gpu
# 3 = ki-jmck
# 4 = Nautilus
# 5 = Kraken
# 6 = physics-179
# 7 = stampede
if [ $isorange -eq 1  ]
then
    system=1
    parallel=1
elif [ $isorangegpu -eq 1  ]
then
    system=2
    parallel=1
elif [ $isjmck -eq 1 ]
then
    system=3
    parallel=0
elif [ $isnautilus -eq 1  ]
then
    system=4
    parallel=1
elif [ $iskraken -eq 1 ]
then
    system=5
    #parallel=1
    parallel=2 # so uses makemoviec instead of python with appropriate arg changes
elif [ $isphysics179 -eq 1 ]
then
    system=6
    parallel=0
elif [ $isstampede -eq 1 ]
then
    system=7
    #parallel=1
    parallel=2 # so uses makemoviec instead of python with appropriate arg changes
else
    # CHOOSE
    system=3
    parallel=0
fi


echo "system=$system parallel=$parallel"



# change to directory where processing occurs
cd $dirname



###################################
if [ $dolinks -eq 1 ]
then

    echo "Doing Links"

    for thedir in $dirruns
    do
        
        echo "Doing links for: "$thedir


    # make movie directory
        mkdir -p $dirname/${thedir}/$moviedirname/dumps/
        cd $dirname/${thedir}/$moviedirname


        echo "create new links (old links are removed) for: "$thedir


    # remove old links
        if [ $docleanexist -eq 1 ]
        then
            if [ "$moviedirname" != "" ] &&
                [ "$moviedirname" != "/" ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/fieldline*.bin
                # this won't remove .npz's -- which is correct.
            fi
        fi

    # FUCKMARK
        #ln -s $dirname/${thedir}/fulllatest11/avg*.npy .

        echo "Linking base files"
        alias cp='cp'
        cp ~/py/scripts/createlinksaltquiet.sh .
        echo "sh createlinksaltquiet.sh 1 $dirname/${thedir} ./"
        sh createlinksaltquiet.sh 1 $dirname/${thedir} ./

        echo "now remove some fraction of links (only keep about 50 for averaging period, first one, and last one): "$thedir
        cd $dirname/${thedir}/$moviedirname/dumps/
    # avoid @ symbol on soft links
        alias ls='ls'
    # get list of files in natural human order
        fieldlinelist=`ls -v | egrep 'fieldline[0-9]+\.bin' | grep -v "npz"`
        firstfieldlinefile=`ls -v | egrep 'fieldline[0-9]+\.bin' | grep -v "npz" | head -1`
        lastfieldlinefile=`ls -v | egrep 'fieldline[0-9]+\.bin' | grep -v "npz" | tail -1`
        numfiles=`echo $fieldlinelist | wc | awk '{print $2}'`
    #
    # set 1/2 to keep since average over roughly latter half in time of data
    #
    #default
    # most steady by 8000 but run till only 13000 for dipoley runs
    # or for toroidal runs, ran for 2X when was steady.  So also good.
        useend=1
        usefirst=1
    #
    ######################
    # default
        factor=1
        #keepfilesstart=$(( (1 * $numfiles) / $factor ))
	keepfilesstart=0
        keepfilesend=$(( $numfiles ))
    #
    ########################
    # different than default
    #
    # 
        if [ "$thedir" == "run.like8" ] ||
            [ "$thedir" == "thickdisk8" ] ||
            [ "$thedir" == "thickdisk11" ] ||
            [ "$thedir" == "thickdisk12" ] ||
            [ "$thedir" == "thickdisk13" ] ||
            [ "$thedir" == "thickdiskrr2" ] ||
            [ "$thedir" == "run.liker2butbeta40" ] ||
            [ "$thedir" == "run.liker2" ] ||
            [ "$thedir" == "thickdisk16" ] ||
            [ "$thedir" == "thickdisk5" ] ||
            [ "$thedir" == "thickdisk14" ] ||
            [ "$thedir" == "thickdiskr1" ] ||
            [ "$thedir" == "thickdiskr2" ] ||
            [ "$thedir" == "run.liker1" ]
        then
            keepfilesstart=$(( 1+3995 ))
            keepfilesend=$(( $numfiles ))
        fi
    #
    #
        if [ "$thedir" == "thickdisk9" ]
        then
            keepfilesstart=$(( 1+3995 ))
            keepfilesend=$(( $numfiles ))
        fi
    #####
        if [ "$thedir" == "thickdisk7" ]
        then
            keepfilesstart=$(( 1+2398 ))
            keepfilesend=$(( $numfiles ))
        fi
    #####
        if [ "$thedir" == "thickdiskr7" ]
        then
            keepfilesstart=$(( 1+2398 ))
            keepfilesend=$(( $numfiles ))
        fi
    #####
        if [ "$thedir" == "sasham9full2pi" ] ||
            [ "$thedir" == "sasham5" ] ||
            [ "$thedir" == "sasham2" ] ||
            [ "$thedir" == "sasha0" ] ||
            [ "$thedir" == "sasha1" ] ||
            [ "$thedir" == "sasha2" ] ||
            [ "$thedir" == "sasha5" ] ||
            [ "$thedir" == "sasha9b25" ] ||
            [ "$thedir" == "sasha9b100" ]
        then
            keepfilesstart=$(( 1+1598 ))
            keepfilesend=$(( $numfiles ))
        fi
        if [ "$thedir" == "sasha9b200" ]
        then
            keepfilesstart=$(( 1+3198 ))
            keepfilesend=$(( $numfiles ))
        fi
        if [ "$thedir" == "sasha99" ]
        then
            keepfilesstart=$(( 1+2998 ))
            keepfilesend=$(( $numfiles ))
        fi
    #
    ##########
        if [ "$thedir" == "thickdiskr3" ]
        then
            keepfilesstart=$(( 1+24200 )) # tstart=58000
            keepfilesend=$(( $numfiles ))
        fi
        if [ "$thedir" == "thickdisk17" ]
        then
            keepfilesstart=$(( 1+30930 )) # tstart=80000
            keepfilesend=$(( $numfiles ))
        fi
        if [ "$thedir" == "thickdisk10" ]
        then
            keepfilesstart=$(( 1+24900 )) # tstart=58000
            keepfilesend=$(( $numfiles ))
        fi
        if [ "$thedir" == "thickdiskr15" ]
        then
            keepfilesstart=$(( 1+24200 )) # tstart=80000
            keepfilesend=$(( $numfiles ))
        fi
        if [ "$thedir" == "thickdisk3" ]
        then
            keepfilesstart=$(( 1+25500 )) # tstart=58000
            keepfilesend=$(( $numfiles ))
        fi
        if [ "$thedir" == "thickdiskhr3" ]
        then
            keepfilesstart=$(( 1+25460 )) # tstart=58000
            keepfilesend=$(( $numfiles ))
        fi
    #
    ########
        if [ "$thedir" == "blandford3d_new" ]
        then
            keepfilesstart=$(( 1+748 ))  #tstart=1500
            keepfilesend=$(( $numfiles ))
        fi
    #
    #
    # don't want to go till very end with this model
        if [ "$thedir" == "runlocaldipole3dfiducial" ]
        then
            #keepfilesstart=`echo "result=(900/5662)*$numfiles;scale=0;puke=result/=1;puke" | bc -l`  #tstart=2000
            #keepfilesend=`echo "result=(2500/5662)*$numfiles;scale=0;puke=result/=1;puke" | bc -l`  #tend=3000
            keepfilesstart=$(( 1+998 ))  #tstart=2000
            keepfilesend=$(( 1+1502 )) #tend=3000
        fi
    #
#       # THETAROT simulations
#        if [ "$thedir" == "sashaam9full2pit0.15" ] ||
#            [ "$thedir" == "sashaa9b100t0.15" ] ||
#            [ "$thedir" == "sashaa99t0.15" ] ||
#            [ "$thedir" == "sashaam9full2pit0.3" ] ||
#            [ "$thedir" == "sashaa9b100t0.3" ] ||
#            [ "$thedir" == "sashaa99t0.3" ] ||
#            [ "$thedir" == "sashaam9full2pit0.6" ] ||
#            [ "$thedir" == "sashaa9b100t0.6" ] ||
#            [ "$thedir" == "sashaa99t0.6" ] ||
#            [ "$thedir" == "sashaam9full2pit1.5708" ] ||
#            [ "$thedir" == "sashaa9b100t1.5708" ] ||
#            [ "$thedir" == "sashaa99t1.5708" ] ||
#            [ "$thedir" == "thickdiskfull3d7tilt0.35" ] ||
#            [ "$thedir" == "thickdiskfull3d7tilt0.7" ] ||
#            [ "$thedir" == "thickdiskfull3d7tilt1.5708" ]
#        then
#            # keep all files
#            #keepfilesstart=$(( 1 ))
#            #keepfilesend=$(( $numfiles ))
#            #
#            # if only want to get first and last file images, then just do below that avoids keeping anything.  But then later first and last file force to be kept separately.
#            keepfilesstart=$(( 3390 ))
#            keepfilesend=$(( 5023 ))
#        fi
#
        if  [ "$thedir" == "sashaa99t0.15" ]
        then
            keepfilesstart=$(( 4390 ))
            keepfilesend=$(( 5023 ))
        fi

        if  [ "$thedir" == "sashaa99t0.3" ]
        then
            keepfilesstart=$(( 4390 ))
            keepfilesend=$(( 5016 ))
        fi

        if  [ "$thedir" == "sashaa99t0.6" ]
        then
            keepfilesstart=$(( 4390 ))
            keepfilesend=$(( 5192 ))
        fi
        if  [ "$thedir" == "sashaa99t1.5708" ]
        then
            keepfilesstart=$(( 4390 ))
            keepfilesend=$(( 5737 ))
        fi
##############
        if  [ "$thedir" == "sashaam9full2pit0.15" ]
        then
            keepfilesstart=$(( 3390 ))
            keepfilesend=$(( 4655 ))
        fi

        if  [ "$thedir" == "sashaam9full2pit0.3" ]
        then
            keepfilesstart=$(( 3390 ))
            keepfilesend=$(( 4418 ))
        fi

        if  [ "$thedir" == "sashaam9full2pit0.6" ]
        then
            keepfilesstart=$(( 3390 ))
            keepfilesend=$(( 4079 ))
        fi
        if  [ "$thedir" == "sashaam9full2pit1.5708" ]
        then
            keepfilesstart=$(( 3390 ))
            keepfilesend=$(( 4715 ))
        fi
##############
        if  [ "$thedir" == "sasha9b100t0.15" ]
        then
            keepfilesstart=$(( 3390 ))
            keepfilesend=$(( 4057 ))
        fi

        if  [ "$thedir" == "sasha9b100t0.3" ]
        then
            keepfilesstart=$(( 3390 ))
            keepfilesend=$(( 4516 ))
        fi

        if  [ "$thedir" == "sasha9b100t0.6" ]
        then
            keepfilesstart=$(( 3390 ))
            keepfilesend=$(( 4201 ))
        fi
        if  [ "$thedir" == "sasha9b100t1.5708" ]
        then
            keepfilesstart=$(( 3390 ))
            keepfilesend=$(( 4876 ))
        fi
    ##############
	if  [ "$thedir" == "thickdiskfull3d7tilt0.35" ]
	then
            keepfilesstart=$(( 3898 ))
            keepfilesend=$(( 5493 ))
	fi
	if [ "$thedir" == "thickdiskfull3d7tilt0.7" ]
	then
            keepfilesstart=$(( 3898 ))
            keepfilesend=$(( 5063 ))
	fi
	if [ "$thedir" == "thickdiskfull3d7tilt1.5708" ]
	then
            keepfilesstart=$(( 3898 ))
            keepfilesend=$(( 4932 ))
	fi
    #################
        if [ $keepfilesstart -le $keepfilesend ]
        then

            keepfilesdiff=$(( $keepfilesend - $keepfilesstart ))
            lastfilesdiff=$(( $numfiles - $keepfilesend ))
    #
            echo "keepfilesstart=$keepfilesstart keepfilesend=$keepfilesend keepfilesdiff=$keepfilesdiff"
    # if above 1/2 kills more than want to keep, then avoid kill of 1/2
            if [ $keepfilesdiff -lt $numkeep ]
	        then
	            keepfilesstart=0
	            keepfilesend=$numfiles
	            keepfilesdiff=$(( $keepfilesend - $keepfilesstart ))
                echo "keepfilesdiff=$keepfilesdiff -lt numkeep=$numkeep"
            else
            #keepfieldlinelist=`ls -v | grep "fieldline" | tail -$keepfilesstart | head -$keepfilesdiff`
	            rmfieldlinelist1=`ls -v | egrep 'fieldline[0-9]+\.bin' | grep -v "npz" | head -$keepfilesstart`
	            for fil in $rmfieldlinelist1
	            do
                #echo "removing $dirname/${thedir}/$moviedirname/dumps/$fil"   #DEBUG
	                rm -rf $dirname/${thedir}/$moviedirname/dumps/$fil
	            done
	            rmfieldlinelist2=`ls -v | egrep 'fieldline[0-9]+\.bin' | grep -v "npz" | tail -$lastfilesdiff`
	            for fil in $rmfieldlinelist2
	            do
                #echo "removing $dirname/${thedir}/$moviedirname/dumps/$fil"   #DEBUG
	                rm -rf $dirname/${thedir}/$moviedirname/dumps/$fil
	            done
            fi
    #
        ###############
            echo "now trim every so a file so only about numkeep+2 files in the end: "$thedir
            fieldlinelist=`ls -v | egrep 'fieldline[0-9]+\.bin' | grep -v "npz"`
            numfiles=`echo $fieldlinelist | wc | awk '{print $2}'`
    #
            skipfactor=$(( $numfiles / $numkeep ))
            echo "skipfactor=$skipfactor"
    #
            if [ $skipfactor -eq 0 ]
            then
	            resid=$(( $numfiles - $numkeep ))
	            echo "keeping bit extra: "$resid
            fi
        #
            if [ $skipfactor -gt 0 ]
            then
	            iiter=0
	            for fil in $fieldlinelist
	            do
	                mymod=$(( $iiter % $skipfactor ))
                #echo "fil=$fil mymod=$mymod iter=$iiter"#DEBUG
	                if [ $mymod -ne 0 ]
	                then
		                rm -rf $dirname/${thedir}/$moviedirname/dumps/$fil
	                fi
	                iiter=$(( $iiter + 1 ))
	            done
            fi

        else
            # remove all fieldline files
		    rm -rf $dirname/${thedir}/$moviedirname/dumps/fieldline*.bin
            # this won't remove .npz's -- which is correct.
        fi
        #



        #############
        echo "Ensure fieldline0000.bin and last fieldline files exist: "$thedir
        cd $dirname/${thedir}/$moviedirname/dumps/
        if [ $usefirst -eq 1 ]
        then
            echo "usefirst: $dirname/${thedir}/dumps/$firstfieldlinefile"
            ln -s $dirname/${thedir}/dumps/$firstfieldlinefile .
        fi
        if [ $useend -eq 1 ]
        then
            echo "useend: $dirname/${thedir}/dumps/$lastfieldlinefile"
            ln -s $dirname/${thedir}/dumps/$lastfieldlinefile .
        fi



    done

    echo "Done with links"

fi


###################################
if [ $dofiles -eq 1 ]
then

    echo "Doing files"

    for thedir in $dirruns
    do

        echo "Doing files for: "$thedir


        echo "cp makemovie.sh: "$thedir
        cd $dirname/${thedir}/$moviedirname/
        cp ~/py/scripts/makemovie.sh .

        echo "edit makemovie.sh: "$thedir

        ##################
        #
        # NO LONGER using "export runn" in makemovie.sh:
        #
    #in makemovie.sh:
    # for thickdisk7 runn=5
    # for run.like8 run.liker1 run.liker2 runn=12
    # for runlocaldipole3dfiducial runn=12
    # more like runn=4 for thickdisk7 for avg creation.
        if [ "$thedir" == "thickdisk7" ] ||
            [ "$thedir" == "thickdiskfull3d7tilt0.35" ] ||
            [ "$thedir" == "thickdiskfull3d7tilt0.7" ] ||
            [ "$thedir" == "thickdiskfull3d7tilt1.5708" ]
	    then
	        sed -e 's/export runn=[0-9]*/export runn=4/g' makemovie.sh > makemovielocal.temp.sh
        elif [ "$thedir" == "sasha99" ]
	    then
	        sed -e 's/export runn=[0-9]*/export runn=8/g' makemovie.sh > makemovielocal.temp.sh
        elif [ "$thedir" == "sashaam9full2pit0.15" ] ||
            [ "$thedir" == "sashaa9b100t0.15" ] ||
            [ "$thedir" == "sashaa99t0.15" ] ||
            [ "$thedir" == "sashaam9full2pit0.3" ] ||
            [ "$thedir" == "sashaa9b100t0.3" ] ||
            [ "$thedir" == "sashaa99t0.3" ] ||
            [ "$thedir" == "sashaam9full2pit0.6" ] ||
            [ "$thedir" == "sashaa9b100t0.6" ] ||
            [ "$thedir" == "sashaa99t0.6" ] ||
            [ "$thedir" == "sashaam9full2pit1.5708" ] ||
            [ "$thedir" == "sashaa9b100t1.5708" ] ||
            [ "$thedir" == "sashaa99t1.5708" ]
        then
	        sed -e 's/export runn=[0-9]*/export runn=4/g' makemovie.sh > makemovielocal.temp.sh
        else
	        sed -e 's/export runn=[0-9]*/export runn=12/g' makemovie.sh > makemovielocal.temp.sh
        fi

        # force use of local __init__.py file (GODMARK: doesn't actually do anything due to export vs. no export part of bash line in makemovie.sh)
        #sed 's/export initfile=\$MREADPATH\/__init__.py/export initfile=\/data2\/$user\/'${thedir}'\/'${moviedirname}'\/__init__.local.py/g' makemovielocal.temp.sh > makemovielocal.sh
        cat makemovielocal.temp.sh > makemovielocal.sh
        rm -rf makemovielocal.temp.sh

        echo "cp  __init__.py to __init__.local.py: "$thedir
        cp ~/py/mread/__init__.py __init__.local.temp.py

    #if [ "$thedir" == "runlocaldipole3dfiducial" ]
    #then
	#nothing=1
        #in __init__.local.py:
        # for runlocaldipole3dfiducial myMB09=0 -> myMB09=1
	#sed 's/myMB09=0/myMB09=1/g' __init__.local.temp.py > __init__.local.py

	# MB09 no longer needs this with myMB09 switch
	#rm -rf titf.txt
	#echo "#" >> titf.txt
	#echo "0 1 2000 100000" >> titf.txt

	# switch no longer present or required as long as model name correct

    #else
	#    cp __init__.local.temp.py __init__.local.py
    #fi

	    cp __init__.local.temp.py __init__.local.py

        rm -rf __init__.local.temp.py


        if [ "$thedir" == "thickdiskhr3" ]
	    then
            ln -s $dirname/thickdisk3/$moviedirname/qty2.npy $dirname/${thedir}/$moviedirname/qty2_thickdisk3.npy
        fi


    done

    echo "Done with files"


fi




##############################################
make1d2dormovie=$(( $make1d + $makemerge + $makeplot + $makemontage + $makepowervsmplots + $makespacetimeplots + $makefftplot + $makespecplot + $makeinitfinalplot + $makethradfinalplot + $makeframes + $makemovie + $makeavg + $makeavgmerge + $makeavgplot ))
if [ $make1d2dormovie -gt 0 ]
then

    echo "Doing makemovie.sh stuff"

    for thedir in $dirruns
    do
	    echo "Doing makemovielocal for: "$thedir

	    cd $dirname/${thedir}/$moviedirname

    #################
        if [ $docleanexist -eq 1 ]
        then
            
            echo "clean: "$thedir
        # only clean what one is redoing and isn't fully overwritten
            if [ $make1d -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/qty2_[0-9]*_[0-9]*.npy
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.stderr.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.full.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.out
                rm -rf $dirname/${thedir}/$moviedirname/python_u_[0-9]*_[0-9]*_[0-9]*.stderr.out
                rm -rf $dirname/${thedir}/$moviedirname/python_u_[0-9]*_[0-9]*_[0-9]*.full.out
                rm -rf $dirname/${thedir}/$moviedirname/python_u_[0-9]*_[0-9]*_[0-9]*.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*_[0-9]*.stderr.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*_[0-9]*.full.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*_[0-9]*.out
            fi
            if [ $makemerge -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/qty2.npy
            fi
            if [ $makeplot -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/*.pdf
                rm -rf $dirname/${thedir}/$moviedirname/python.plot.stderr.out
                rm -rf $dirname/${thedir}/$moviedirname/python.plot.full.out
                rm -rf $dirname/${thedir}/$moviedirname/python.plot.out
                rm -rf $dirname/${thedir}/$moviedirname/aphi.png
                rm -rf $dirname/${thedir}/$moviedirname/aphi.pdf
                rm -rf $dirname/${thedir}/$moviedirname/aphi.eps
                rm -rf $dirname/${thedir}/$moviedirname/datavsr*.txt
                rm -rf $dirname/${thedir}/$moviedirname/datavsh*.txt
                rm -rf $dirname/${thedir}/$moviedirname/datavst*.txt
            fi
        #
            if [ $makemontage -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/montage*.png
                rm -rf $dirname/${thedir}/$moviedirname/montage*.eps
                rm -rf $dirname/${thedir}/$moviedirname/montage*.png
            fi
            if [ $makepowervsmplots -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/powervsm*.png
                rm -rf $dirname/${thedir}/$moviedirname/powervsm*.eps
                rm -rf $dirname/${thedir}/$moviedirname/powervsm*.png
            fi
            if [ $makespacetimeplots -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/plot*.png
                rm -rf $dirname/${thedir}/$moviedirname/plot*.eps
                rm -rf $dirname/${thedir}/$moviedirname/plot*.png
            fi
            if [ $makefftplot -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/fft?.png
                rm -rf $dirname/${thedir}/$moviedirname/fft?.eps
                rm -rf $dirname/${thedir}/$moviedirname/fft?.png
            fi
            if [ $makespecplot -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/spec?.png
                rm -rf $dirname/${thedir}/$moviedirname/spec?.eps
                rm -rf $dirname/${thedir}/$moviedirname/spec?.png
            fi
            if [ $makeinitfinalplot -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/init1.png
                rm -rf $dirname/${thedir}/$moviedirname/init1.eps
                rm -rf $dirname/${thedir}/$moviedirname/init1.png
                rm -rf $dirname/${thedir}/$moviedirname/middle1.png
                rm -rf $dirname/${thedir}/$moviedirname/middle1.eps
                rm -rf $dirname/${thedir}/$moviedirname/middle1.png
                rm -rf $dirname/${thedir}/$moviedirname/final1.png
                rm -rf $dirname/${thedir}/$moviedirname/final1.eps
                rm -rf $dirname/${thedir}/$moviedirname/final1.png
                rm -rf $dirname/${thedir}/$moviedirname/init1_stream.png
                rm -rf $dirname/${thedir}/$moviedirname/init1_stream.eps
                rm -rf $dirname/${thedir}/$moviedirname/init1_stream.png
                rm -rf $dirname/${thedir}/$moviedirname/middle1_stream.png
                rm -rf $dirname/${thedir}/$moviedirname/middle1_stream.eps
                rm -rf $dirname/${thedir}/$moviedirname/middle1_stream.png
                rm -rf $dirname/${thedir}/$moviedirname/final1_stream.png
                rm -rf $dirname/${thedir}/$moviedirname/final1_stream.eps
                rm -rf $dirname/${thedir}/$moviedirname/final1_stream.png
            fi
            if [ $makethradfinalplot -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/plot0qvsth_.png
            fi
        #
        #
            if [ $makeframes -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.movieframes.stderr.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.movieframes.full.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.movieframes.out
                rm -rf $dirname/${thedir}/$moviedirname/*.png
                rm -rf $dirname/${thedir}/$moviedirname/*.eps
            fi
            if [ $makemovie -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/*.avi
            fi
            if [ $makeavg -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.stderr.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.full.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.out
            fi
            if [ $makeavg -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/avg2d[0-9]*_[0-9]*.npy
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.stderr.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.full.out
                rm -rf $dirname/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.out
            fi
            if [ $makeavgmerge -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/avg2d[0-9]*_[0-9]*_[0-9]*.npy
                rm -rf $dirname/${thedir}/$moviedirname/avg2d.npy
            fi
            if [ $makeavgplot -eq 1 ]
            then
                rm -rf $dirname/${thedir}/$moviedirname/fig2.png
            fi
        fi
    ###############
        # shouldn't need ${thedir} at end of command
        cmdraw="sh makemovielocal.sh ${thedir} $make1d $makemerge $makeplot $makemontage $makepowervsmplots $makespacetimeplots $makefftplot $makespecplot $makeinitfinalplot $makethradfinalplot $makeframes $makemovie $makeavg $makeavgmerge $makeavgplot ${system} ${parallel}"

        echo "cmdraw = $cmdraw"

	    if [ $parallel -eq 0 ]
        then
	        $cmdraw
        else
            rm -rf makemovielocal_${thedir}.stderr.out
            rm -rf makemovielocal_${thedir}.out
            rm -rf makemovielocal_${thedir}.full.out
            echo "((nohup $cmdraw 2>&1 1>&3 | tee makemovielocal_${thedir}.stderr.out) 3>&1 1>&2 | tee makemovielocal_${thedir}.out) > makemovielocal_${thedir}.full.out 2>&1" > batch_makemovielocal_${thedir}.sh
            chmod a+x ./batch_makemovielocal_${thedir}.sh
            nohup ./batch_makemovielocal_${thedir}.sh &
            #(nohup $cmdraw &> makemovielocal_${thedir}.full.out &)
        fi

    done

    echo "Done with makemovie.sh stuff"

fi


##############################################
#
echo "Now collect Latex results"

if [ $collect -eq 1 ] &&
    [ $system -ne 3 ] &&
    [ $system -ne 5 ] &&
    [ $system -ne 6 ] &&
    [ $system -ne 7 ]
then
    # then copy over results
    for thedir in $dircollect
    do
	    echo "Doing remote collection for: "$thedir
        #
        if [ $system -eq 4 ]
        then
            # use -s 2 in case same file size
            ssh $globusremote scp -D -r -s 2 xsede#nautilus:$dirname/${thedir}/$moviedirname $globushost:/data2/$user/${thedir}/
        elif [ $system -eq 5 ] ||
            [ $system -eq 7 ]
        then
            ssh $globusremote scp -D -r -s 2 xsede#kraken:$dirname/${thedir}/$moviedirname $globushost:/data2/$user/${thedir}/
        else
            #scp $dirname/${thedir}/$moviedirname/qty*.npy $dirname/${thedir}/$moviedirname/*.txt $dirname/${thedir}/$moviedirname/*.png $dirname/${thedir}/$moviedirname/*.eps $dirname/${thedir}/$moviedirname/python.plot.*.out $user@$remotehost:/data2/$user/${thedir}/$moviedirname/
            scp -rp $dirname/${thedir}/$moviedirname $user@$remotehost:/data2/$user/${thedir}/
        fi

    done
fi



if [ $collect -eq 1 ] &&
    [ $system -eq 5 ] ||
    [ $collect -eq 1 ] &&
    [ $system -eq 7 ]
then
# below only appears updated if also do powervsm stuff.
    pythonlatexfile="python_u_3_0_1.stdout.out"
# below won't have updated Q's
#    pythonlatexfile="python_u_3_0_0.stdout.out"
else
    pythonlatexfile="python.plot.out"
fi


if [ $collect -eq 1 ] &&
    [ $system -eq 3 ] ||
    [ $collect -eq 1 ] &&
    [ $system -eq 6 ] ||
    [ $collect -eq 1 ] &&
    [ $system -eq 5 ] ||
    [ $collect -eq 1 ] &&
    [ $system -eq 7 ]
then

    cd $dirname/

    echo "Doing collection"

    # refresh tables.tex
    rm  -rf tables$moviedirname.tex

    iiter=1
    for thedir in $dircollect
    do
	    echo "Doing collection for: "$thedir
        if [ $iiter -eq 1 ]
        then
		    cat $dirname/${thedir}/$moviedirname/$pythonlatexfile | grep "HLatex" >> tables$moviedirname.tex
		    echo "HLatex: \hline" >> tables$moviedirname.tex
        fi
		cat $dirname/${thedir}/$moviedirname/$pythonlatexfile | grep "VLatex" >> tables$moviedirname.tex
		echo "$dirname $thedir $moviedirname $pythonlatexfile : $iiter"

        iiter=$(( $iiter+1))
    done


    # temporary fix to model names:
    #cat tables$moviedirname.tex | sed 's/0_C/0\_C/g' | sed 's/A94BfN40\\_C5/A-94BfN10\\_Cx/g' | sed 's/A-94BfN10\\_C1/A94BfN40\\_C5/g' | sed 's/A-94BfN10\\_Cx/A-94BfN10\\_C1/g' | sed 's/MB09_D /MB09\\_D /g'  > tables$moviedirname.tex.tmp
    #mv tables$moviedirname.tex.tmp tables$moviedirname.tex

    ##############################################
    #
    # Tables:
    numtbls=17

    for numtbl in `seq 1 $numtbls`
    do
	    echo "Doing Table #: "$numtbl


        ###############################
        fname=table$numtbl$moviedirname.tex
        rm -rf $fname
        #
        echo "\begin{table*}" >> $fname
        if [ $numtbl -eq 1 ]
        then
            echo "\caption{Physical Model Parameters}" >> $fname
        fi
        if [ $numtbl -eq 2 ]
        then
            echo "\caption{Numerical Model Parameters}" >> $fname
        fi
        if [ $numtbl -eq 3 ]
        then
            echo "\caption{Grid Cells across Half-Thickness at Horizon, Half-Thicknesses of Disk, and Location for Interfaces for Disk-Corona and Corona-Jet}" >> $fname
        fi
        if [ $numtbl -eq 16 ]
        then
            echo "\caption{Grid Cells across Half-Thickness at Horizon, Half-Thicknesses of Disk, and Location for Interfaces for Disk-Corona and Corona-Jet}" >> $fname
        fi
        if [ $numtbl -eq 17 ]
        then
            echo "\caption{Grid Cells across Half-Thickness at Horizon, Half-Thicknesses of Disk, and Location for Interfaces for Disk-Corona and Corona-Jet}" >> $fname
        fi
        if [ $numtbl -eq 4 ]
        then
            echo "\caption{Viscosities, Grid Cells per Correlation lengths and MRI Wavelengths, MRI Wavelengths per full Disk Height, and Radii for MRI Suppression}" >> $fname
        fi
        if [ $numtbl -eq 5 ]
        then
            echo "\caption{Rest-Mass Accretion and Ejection Rates}" >> $fname
        fi
        if [ $numtbl -eq 6 ]
        then
            echo "\caption{Percent Energy Efficiency: BH, Jet, Winds, and NT}" >> $fname
        fi
        if [ $numtbl -eq 7 ]
        then
            echo "\caption{Percent Energy Efficiency: Magnetized Wind and Entire Wind}" >> $fname
        fi
        if [ $numtbl -eq 8 ]
        then
            echo "\caption{Specific Angular Momentum: BH, Jet, Winds, and NT}" >> $fname
        fi
        if [ $numtbl -eq 9 ]
        then
            echo "\caption{Specific Angular Momentum: Magnetized Wind and Entire Wind}" >> $fname
        fi
        if [ $numtbl -eq 10 ]
        then
            echo "\caption{Spin-Up Parameter}" >> $fname
        fi
        if [ $numtbl -eq 15 ]
        then
            echo "\caption{Spin-Up Parameter: BH, Jet, Winds, and NT}" >> $fname
        fi
        if [ $numtbl -eq 11 ]
        then
            echo "\caption{Absolute Magnetic Flux per unit: Rest-Mass Fluxes, Initial Magnetic Fluxes, Available Magnetic Fluxes, and BH Magnetic Flux}" >> $fname
        fi
        if [ $numtbl -eq 12 ]
        then
            echo "\caption{Inner and Outer Radii for Least-Square Fits, Disk+Corona Stagnation Radius, and Fitted Power-Law Indices for Disk Flow}" >> $fname
        fi
        #
        if [ $numtbl -eq 13 ]
        then
            echo "\caption{Inner and Outer Radii for Least-Square Fits, and Fitted Power-Law Indices for Wind Flow}" >> $fname
        fi
        #
        if [ $numtbl -eq 14 ]
        then
            echo "\caption{Inner and Outer Radii for Least-Square Fits, Disk+Corona Stagnation Radius, and Fitted Power-Law Indices for Disk and Wind Flows}" >> $fname
        fi
        #
        echo "\begin{center}" >> $fname
        rawnumc=`grep "Latex$numtbl:" tables$moviedirname.tex | sed 's/[HV]Latex$numtbl: //g' | tail -1 | wc | awk '{print $2}'`
        numc=$(( ($rawnumc - 2)/2 ))
        str1="\begin{tabular}[h]{|"
        str2=""
        for striter in `seq 1 $numc`
        do
            if [ $striter -eq 1 ]
            then
                str2=$str2"l|"
            else
                str2=$str2"r|"
            fi
        done
        str3="}"
        strfinal=$str1$str2$str3
        echo $strfinal >> $fname
        echo "\hline" >> $fname
        # if change model names, probably have to change the below
        #
        if [ $numtbl -eq 14 ] # fits
        then
            # no 2D or MB09D models here
            egrep "Latex$numtbl:|Latex:" tables$moviedirname.tex | sed 's/\([0-9]\)%/\1\\%/g' | sed 's/[HV]Latex'$numtbl': //g' | sed 's/[HV]Latex: //g' | sed 's/\$\&/$ \&/g'   | sed 's/A0\.94BpN100 /\\\\\nA0\.94BpN100 /g' | sed 's/{\\bf A-0.94BfN40HR} /\\\\\n{\\bf A-0.94BfN40HR} /g' | sed 's/A-0\.94BtN10 /\\\\\nA-0\.94BtN10 /g'  | sed 's/MB09Q /\\\\\nMB09Q /g'| sed 's/A-0.9N100 /\\\\\nA-0.9N100 /g'  | sed 's/} \&/}$ \&/g' | sed 's/} \\/}$  \\/g' | sed 's/nan/0/g' | sed 's/e+0/e/g' | sed 's/e-0/e-/g'  | column  -t >> $fname
        else
            egrep "Latex$numtbl:|Latex:" tables$moviedirname.tex | sed 's/\([0-9]\)%/\1\\%/g' | sed 's/[HV]Latex'$numtbl': //g' | sed 's/[HV]Latex: //g' | sed 's/\$\&/$ \&/g'   | sed 's/A0\.94BpN100 /\\\\\nA0\.94BpN100 /g' | sed 's/{\\bf A-0.94BfN40HR} /\\\\\n{\\bf A-0.94BfN40HR} /g' | sed 's/A-0\.94BtN10 /\\\\\nA-0\.94BtN10 /g'  | sed 's/MB09D /\\\\\nMB09D /g'| sed 's/A-0.9N100 /\\\\\nA-0.9N100 /g'  | sed 's/} \&/}$ \&/g' | sed 's/} \\/}$  \\/g' | sed 's/nan/0/g' | sed 's/e+0/e/g' | sed 's/e-0/e-/g'  | column  -t >> $fname
        fi
        #
        echo "\hline" >> $fname
        echo "\hline" >> $fname
        echo "\end{tabular}" >> $fname
        echo "\end{center}" >> $fname
        echo "\label{tbl$numtbl}" >> $fname
        echo "\end{table*}" >> $fname
        ###############################

        # Copy over to final table file names

        cp $fname tbl$numtbl.tex

    done


    echo "For paper, now do:   scp tbl[0-9].tex jon@ki-rh42:/data/jon/thickdisk/harm_thickdisk/ ; scp tbl[0-9][0-9].tex jon@ki-rh42:/data/jon/thickdisk/harm_thickdisk/"
    echo "For paper, now do:   scp tbl[0-9].tex jon@physics-179.umd.edu:/data/jon/harm_harmrad/ ; scp tbl[0-9][0-9].tex jon@physics-179.umd.edu:/data/jon/harm_harmrad/"
    
    



    ########################
    # Aux tables:

	echo "Doing Aux Tables"

    grep "Latex42:" tables$moviedirname.tex | sed 's/[HV]Latex42: //g'  | column  -t > table42$moviedirname.tex
    grep "Latex93:" tables$moviedirname.tex | sed 's/[HV]Latex93: //g'  | column  -t > table93$moviedirname.tex
    grep "Latex94:" tables$moviedirname.tex | sed 's/[HV]Latex94: //g'  | column  -t > table94$moviedirname.tex
    grep "Latex95:" tables$moviedirname.tex | sed 's/[HV]Latex95: //g'  | column  -t > table95$moviedirname.tex
    grep "Latex96:" tables$moviedirname.tex | sed 's/[HV]Latex96: //g'  | column  -t > table96$moviedirname.tex
    grep "Latex97:" tables$moviedirname.tex | sed 's/[HV]Latex97: //g'  | column  -t > table97$moviedirname.tex
    grep "Latex99:" tables$moviedirname.tex | sed 's/[HV]Latex99: //g'  | column  -t > table99$moviedirname.tex

    echo "Done with collection"

fi




echo "Done with all stages"
