#!/bin/bash

EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {arg1 arg2 arg3}"
    exit $E_BADARGS
    # e.g. sh makemovie.sh 1 1 0
fi

# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things

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

# The streamline code does not need gdet B^i's.  It uses B^i's. 

# 2) Change "False" to "True" in the section of __main__ that runs
# generate_time_series()

export MYPYTHONPATH=$HOME/py/

export MREADPATH=$MYPYTHONPATH/mread/
export initfile=$MREADPATH/__init__.py
echo "initfile=$initfile"
export myrand=${RANDOM}
echo "RANDOM=$myrand"


export runn=10
export numparts=2


if [ $1 -eq 1 ]
then

    export myinitfile=$initfile.$myrand
    echo "myinitfile=$myinitfile"

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#NEW FORMAT[\n \t]*#Plot qtys vs. time[\n \t]*generate_time_series()/if True:\n\t#NEW FORMAT\n\t#Plot qtys vs. time\n\tgenerate_time_series()/g;p;}'  $initfile > $myinitfile


    # 3) already be in <directory that contains "dumps" directory> or cd to it
    
    # can also create new directory and create reduced list of fieldline files.  See createlinksalt.sh
    
    # 4) Then generate the file
    
    # This is a poor-man's parallelization technique: first, thread #i
    # generates a file, qty_${runi}_${runn}.npy, which contains a fraction of
    # time slices.  Then, I merge all these partial files into one single
    # file.
    
    
    export je=$(( $numparts - 1 ))
    # above two must be exactly divisible
    export itot=$(( $runn/$numparts ))
    export ie=$(( $itot -1 ))
    
    echo "Running with $itot cores simultaneously"
    
    # just a test:
    # echo "nohup python $myinitfile $runi $runn &> python_${runi}_${runn}.out &"
    # exit 
    
    # LOOP:
    
    for j in `seq 0 $je`
    do
        echo "Data vs. Time: Starting simultaneous run of $ie jobs for group $j"
        for i in `seq 0 $ie`
        do
    	    export runi=$(( $i + $itot * $j ))
    	    textrun="Data vs. Time: Running i=$i j=$j giving runi=$runi with runn=$runn"
    	    #echo $textrun >> out
    	    echo $textrun
            # sleep in order for all threads not to read in at once and overwhelm the drive
    	    sleep 5
            # run job
	    nohup python $myinitfile $runi $runn &> python_${runi}_${runn}.out &
	done
	wait
	echo "Data vs. Time: Ending simultaneous run of $ie jobs for group $j"
    done
    
    wait
    #merge to single file
    nohup python $myinitfile $runn $runn &> python_${runn}_${runn}.out
    #generate the plots
    nohup python $myinitfile &> python.plot.out
    
    # remove created file
    rm -rf $myinitfile

fi


if [ $2 -eq 1 ]
then

    # Now you are ready to generate movie frames, you can do that in
    # parallel, too, in a very similar way.
    
    # 1) You disable the time series section of ~/py/mread/__init__.py and
    # instead enable the movie section
    
    export myinitfile2=$initfile.$myrand.2
    echo "myinitfile2=$myinitfile.2"
    
    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#make a movie[\n \t]*mkmovie()/if True:\n\t#make a movie\n\tmkmovie()/g;p;}'  $initfile > $myinitfile2
    
    
    # 2) Now run job as before.  But makeing movie frames takes about 2X more memory, so increase parts by 2X
    
    export runn=12
    export numparts=4

    
    export je=$(( $numparts - 1 ))
    # above two must be exactly divisible
    export itot=$(( $runn/$numparts ))
    export ie=$(( $itot -1 ))
    
    echo "Running with $itot cores simultaneously"
    
    
    for j in `seq 0 $je`
    do
	echo "Movie Frames: Starting simultaneous run of $ie jobs for group $j"
	for i in `seq 0 $ie`
	do
	    export runi=$(( $i + $itot * $j ))
	    textrun="Movie Frames: Running i=$i j=$j giving runi=$runi with runn=$runn"
    	    #echo $textrun >> out
	    echo $textrun
            # sleep in order for all threads not to read in at once and overwhelm the drive
	    sleep 5
	    # run job
	    nohup python $myinitfile2 $runi $runn &> python_${runi}_${runn}.2.out &
	done
	wait
	echo "Movie Frames: Ending simultaneous run of $ie jobs for group $j"
    done
    
    
    # remove created file
    rm -rf $myinitfile2
    
fi


if [ $3 -eq 1 ]
then

    #  now can create an avi with:
    
    fps=4
    #
    ffmpeg -i lrho%04d_Rzxym1.png -r $fps lrho.mp4

fi


# mplayer -loop 0 lrho.avi

