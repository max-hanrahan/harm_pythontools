#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things


EXPECTED_ARGS=9
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {modelname make1d makemerge makeplot makeframes makemovie makeavg makeavgmerge makeavgplot}"
    echo "e.g. sh makemovie.sh thickdisk7 1 1 1 1 0 0 0 0"
    exit $E_BADARGS
fi


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
#  In order to run mkstreamlinefigure(), you need to have generated 2D
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

export MYPYTHONPATH=$HOME/py/

export MREADPATH=$MYPYTHONPATH/mread/
export initfile=$MREADPATH/__init__.py
echo "initfile=$initfile"
export myrand=${RANDOM}
echo "RANDOM=$myrand"

export localpath=`pwd`


modelname=$1
make1d=$2
makemerge=$3
makeplot=$4
makeframes=$5
makemovie=$6
makeavg=$7
makeavgmerge=$8
makeavgplot=$9


############################
#
# Make time series (vs. t and vs. r)
#
############################
if [ $make1d -eq 1 ]
then

    export runn=7
    export numparts=1


    export myinitfile1=$localpath/__init__.py.1.$myrand
    echo "myinitfile1="${myinitfile1}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#NEW FORMAT[\n \t]*#Plot qtys vs. time[\n \t]*generate_time_series()/if True:\n\t#NEW FORMAT\n\t#Plot qtys vs. time\n\tgenerate_time_series()/g;p;}'  $initfile > $myinitfile1


    # 3) already be in <directory that contains "dumps" directory> or cd to it
    
    # can also create new directory and create reduced list of fieldline files.  See createlinksalt.sh
    
    # 4) Then generate the file
    
    # This is a poor-man's parallelization technique: first, thread #i
    # generates a file, qty_${runi}_${runn}.npy, which contains a fraction of
    # time slices.  Then, I merge all these partial files into one single
    # file.
    
    # Requirements to consider inside __init__.py:
    # A) Must have at least one fieldline????.bin dump beyond fti=8000 for averaging period or else script dies.
    # or create a file titf.txt that contains the following:
##comment
#1000 2000 8000 20000
#   last two numbers indicate range of averaging.  E.g., Set 8000->0 if only using fieldline0000.bin dump and no other dumps.

    # Options to consider inside __init__.py:
    # A) 
    
    export je=$(( $numparts - 1 ))
    # above two must be exactly divisible
    export itot=$(( $runn/$numparts ))
    export ie=$(( $itot -1 ))

    export resid=$(( $runn - $itot * $numparts ))
    
    echo "Running with $itot cores simultaneously"
    
    # just a test:
    # echo "nohup python $myinitfile1 $modelname $runi $runn &> python_${runi}_${runn}.out &"
    # exit 
    
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
	    ilist=`seq 0 $ie`
	    doilist=1
	fi

	if [ $doilist -eq 1 ]
	then
            echo "Data vs. Time: Starting simultaneous run of $itot jobs for group $j"
            for i in $ilist
            do
    		export runi=$(( $i + $itot * $j ))
    		textrun="Data vs. Time: Running i=$i j=$j giving runi=$runi with runn=$runn"
    	        #echo $textrun >> out
    		echo $textrun
                # sleep in order for all threads not to read in at once and overwhelm the drive
	        # No, fieldline file itself of up to order 400M should be cached in memory for most systems.
    		sleep 1
                # run job
		#nohup python $myinitfile1 $modelname $runi $runn &> python_${runi}_${runn}.out &
		((nohup python $myinitfile1 $modelname $runi $runn 2>&1 1>&3 | tee python_${runi}_${runn}.stderr.out) 3>&1 1>&2 | tee python_${runi}_${runn}.out) > python_${runi}_${runn}.full.out 2>&1 &

	    done
	    wait
	    echo "Data vs. Time: Ending simultaneous run of $itot jobs for group $j"
	fi
    done
    
    wait

    # remove created file
    rm -rf $myinitfile1

fi


###################################
#
# Merge npy files
#
####################################
if [ $makemerge -eq 1 ]
then

    # runn should be same as when creating files
    export runn=7


    export myinitfile2=$localpath/__init__.py.2.$myrand
    echo "myinitfile2="${myinitfile2}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#NEW FORMAT[\n \t]*#Plot qtys vs. time[\n \t]*generate_time_series()/if True:\n\t#NEW FORMAT\n\t#Plot qtys vs. time\n\tgenerate_time_series()/g;p;}'  $initfile > $myinitfile2


    echo "Merge to single file"
    #nohup python $myinitfile2 $modelname $runn $runn &> python_${runn}_${runn}.out
    ((nohup python $myinitfile2 $modelname $runn $runn 2>&1 1>&3 | tee python_${runn}_${runn}.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.out) > python_${runn}_${runn}.full.out 2>&1

    # remove created file
    rm -rf $myinitfile2

fi



###################################
#
# Maake plots and latex tables
#
####################################
if [ $makeplot -eq 1 ]
then

    export myinitfile3=$localpath/__init__.py.3.$myrand
    echo "myinitfile3="${myinitfile3}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#NEW FORMAT[\n \t]*#Plot qtys vs. time[\n \t]*generate_time_series()/if True:\n\t#NEW FORMAT\n\t#Plot qtys vs. time\n\tgenerate_time_series()/g;p;}'  $initfile > $myinitfile3


    echo "Generate the plots"
    # &> 
    ((nohup python $myinitfile3 $modelname 2>&1 1>&3 | tee python.plot.stderr.out) 3>&1 1>&2 | tee python.plot.out) > python.plot.full.out 2>&1

    makemontage=0
    if [ $makemontage -eq 1 ]
    then
        # create montage of t vs. r and t vs. h plots
        files=`ls -rt plot*.png`
        montage -geometry 300x600 $files montage.png
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
    fi


    # remove created file
    rm -rf $myinitfile3

fi



###################################
#
# MOVIE FRAMES
#
####################################
if [ $makeframes -eq 1 ]
then



    # Now you are ready to generate movie frames, you can do that in
    # parallel, too, in a very similar way.
    
    # 1) You disable the time series section of ~/py/mread/__init__.py and
    # instead enable the movie section
    
    export myinitfile4=$localpath/__init__.py.${myrand}.4
    echo "myinitfile4="${myinitfile4}
    
    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#make a movie[\n \t]*mkmovie()/if True:\n\t#make a movie\n\tmkmovie()/g;p;}'  $initfile > $myinitfile4
    
    
    # Options to consider inside __init__.py:
    #
    # A) can change showextra=False to True:
    # def plotqtyvstime(qtymem,ihor=11,whichplot=None,ax=None,findex=None,fti=None,ftf=None,showextra=True,prefactor=100)
    # 
    #
    # B) Can choose vmin and vmax for lrho range in movie:
    # mkframe("lrho%04d_Rz%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax1,cb=False,pt=False)
    # mkframexy("lrho%04d_xy%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax2,cb=True,pt=False,dostreamlines=True)
    #
    # C) Can choose frame size by changing plotlen=# where # is # of M that plot will go to in each direction.  Change plotlen when plotgen being setup in mkmovie().  Or directly change mkmovie(framesize=50) to another #.


    
    # 2) Now run job as before.  But makeing movie frames takes about 2X more memory, so increase parts by 2X
    
    export runn=6
    export numparts=1

    
    export je=$(( $numparts - 1 ))
    # above two must be exactly divisible
    export itot=$(( $runn/$numparts ))
    export ie=$(( $itot -1 ))
    
    echo "Running with $itot cores simultaneously"
    
    
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
	    ilist=`seq 0 $ie`
	    doilist=1
	fi

	if [ $doilist -eq 1 ]
	then
	    echo "Movie Frames: Starting simultaneous run of $itot jobs for group $j"
	    for i in $ilist
	    do
		export runi=$(( $i + $itot * $j ))
		textrun="Movie Frames: Running i=$i j=$j giving runi=$runi with runn=$runn"
   	        #echo $textrun >> out
		echo $textrun
                # sleep in order for all threads not to read in at once and overwhelm the drive
		sleep 1
	        # run job
		#nohup python $myinitfile4 $modelname $runi $runn &> &

		((nohup python $myinitfile4 $modelname $runi $runn  2>&1 1>&3 | tee python_${runi}_${runn}.movieframes.stderr.out) 3>&1 1>&2 | tee python_${runi}_${runn}.movieframes.out) > python_${runi}_${runn}.movieframes.full.out 2>&1 &


	    done
	    wait
	    echo "Movie Frames: Ending simultaneous run of $itot jobs for group $j"
	fi
    done
    
    
    # remove created file
    rm -rf $myinitfile4
    
fi


###################################
#
# MOVIE File
#
####################################
if [ $makemovie -eq 1 ]
then

    #  now can create an avi with:
    
    fps=25
    #
    #ffmpeg -i lrho%04d_Rzxym1.png -r $fps -sameq lrho.mp4
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
    else
        # Sasha's command:
	    ffmpeg -y -fflags +genpts -r $fps -i lrho%04d_Rzxym1.png -vcodec mpeg4 -sameq -qmax 5 lrho.$modelname.avi
	    ffmpeg -y -fflags +genpts -r $fps -i lrhosmall%04d_Rzxym1.png -vcodec mpeg4 -sameq -qmax 5 lrhosmall.$modelname.avi

        # for Roger (i.e. any MAC)
	    ffmpeg -y -fflags +genpts -r $fps -i lrho%04d_Rzxym1.png -sameq -qmax 5 lrho.$modelname.mov
	    ffmpeg -y -fflags +genpts -r $fps -i lrhosmall%04d_Rzxym1.png -sameq -qmax 5 lrhosmall.$modelname.mov
    fi

    echo "Now do: mplayer -loop 0 lrho.$modelname.avi OR lrhosmall.$modelname.avi"

fi




###################################
#
# avg File (takes average of 20 fieldline files per avg file created)
#
####################################
if [ $makeavg -eq 1 ]
then

    echo "Doing avg file"

    export runn=7
    export numparts=1


    export myinitfile5=$localpath/__init__.py.5.$myrand
    echo "myinitfile5="${myinitfile5}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#2DAVG[\n \t]*mk2davg()/if True:\n\t#2DAVG\n\tmk2davg()/g;p;}'  $initfile > $myinitfile5
    
    export je=$(( $numparts - 1 ))
    # above two must be exactly divisible
    export itot=$(( $runn/$numparts ))
    export ie=$(( $itot -1 ))

    export resid=$(( $runn - $itot * $numparts ))
    
    echo "Running with $itot cores simultaneously"
    
    # just a test:
    # echo "nohup python $myinitfile1 $modelname $runi $runn &> python_${runi}_${runn}.out &"
    # exit 
    
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
	    ilist=`seq 0 $ie`
	    doilist=1
	fi

	if [ $doilist -eq 1 ]
	then
            echo "Data vs. Time: Starting simultaneous run of $itot jobs for group $j"
            for i in $ilist
            do
    		export runi=$(( $i + $itot * $j ))
    		textrun="Data vs. Time: Running i=$i j=$j giving runi=$runi with runn=$runn"
    	        #echo $textrun >> out
    		echo $textrun
                # sleep in order for all threads not to read in at once and overwhelm the drive
	        # No, fieldline file itself of up to order 400M should be cached in memory for most systems.
    		sleep 1
            # run job
		    ((nohup python $myinitfile5 $modelname $runi $runn 2>&1 1>&3 | tee python_${runi}_${runn}.avg.stderr.out) 3>&1 1>&2 | tee python_${runi}_${runn}.avg.out) > python_${runi}_${runn}.avg.full.out 2>&1 &

	    done
	    wait
	    echo "Data vs. Time: Ending simultaneous run of $itot jobs for group $j"
	fi
    done
    
    wait

    # remove created file
    rm -rf $myinitfile5

fi


###################################
#
# Merge avg npy files
#
####################################
if [ $makeavgmerge -eq 1 ]
then

    # should be same as when creating avg files
    export runn=7


    export myinitfile6=$localpath/__init__.py.6.$myrand
    echo "myinitfile6="${myinitfile6}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#2DAVG[\n \t]*mk2davg()/if True:\n\t#2DAVG\n\tmk2davg()/g;p;}'  $initfile > $myinitfile6


    echo "Merge avg files to single avg file"
    # <index of first avg file to use> <index of last avg file to use> <step=1>
    # must be step=1, or no merge occurs
    step=1
    ((nohup python $myinitfile6 $modelname 0 $runn $step 2>&1 1>&3 | tee python_${runn}_${runn}.avg.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.avg.out) > python_${runn}_${runn}.avg.full.out 2>&1

    # remove created file
    rm -rf $myinitfile6


    # copy resulting avg file to avg2d.npy
    avg2dmerge=`ls -vrt avg2d*_*_*.npy | head -1`    
    cp $avg2dmerge avg2d.npy


fi



###################################
#
# Make avg plot
#
####################################
if [ $makeavgplot -eq 1 ]
then

    export myinitfile7=$localpath/__init__.py.7.$myrand
    echo "myinitfile7="${myinitfile7}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#fig2 with grayscalestreamlines and red field lines[\n \t]*mkstreamlinefigure()/if True:\n\t#fig2 with grayscalestreamlines and red field lines\n\tmkstreamlinefigure()/g;p;}'  $initfile > $myinitfile7


    echo "Generate the avg plots"
    # &> 
    ((nohup python $myinitfile7 $modelname 2>&1 1>&3 | tee python.plot.avg.stderr.out) 3>&1 1>&2 | tee python.plot.avg.out) > python.plot.avg.full.out 2>&1

    
    # remove created file
    rm -rf $myinitfile7

fi



