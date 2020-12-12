#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things



EXPECTED_ARGS=1
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {dirname without trailing /}"
    echo "e.g. sh createlinkes.sh thickdiskrr2"
    exit $E_BADARGS
fi

thedir=$1


# On ki-jmck:

# 1) Ensure sizes all correctly large
#ls -alRS thickdisk11* | grep fieldline | grep -v "\->" | sort -nrk 5

# 2) If not all large, then order list of dirs in order of simulation so overwrites use newer file and repeat

# 3) setup directory list

rm -rf dirs${thedir}.txt
list=`ls  | grep ${thedir}`
for fil in $list ; do echo $fil >> dirs${thedir}.txt ; done

mkdir -p /data2/jmckinne/${thedir}/dumps/
cd /data2/jmckinne/${thedir}/
mv ../dirs${thedir}.txt .

# 4) Edit dir list and choose one's want

# 5) create new full-sim dir and change to dumps dir

cd /data2/jmckinne/${thedir}/dumps/

# 6) make links
for mydir in `cat ../dirs${thedir}.txt` ; do echo $mydir ; for fil in `ls ../../$mydir/dumps/fieldline*.bin` ; do echo $fil ; ln -sf $fil . ;  done ; done

# 7) Also make links to gdump.bin and dump0000.bin

firstdir=`head -1 ../dirs${thedir}.txt`

ln -s ../../$firstdir/dumps/gdump.bin .
ln -s ../../$firstdir/dumps/dump0000.bin .

cd ..

cp -a ../thickdisk9/movie1 .
cd movie1/
rm -rf *.png *.eps *.npy python*.out *.avi *.pdf dumps
alias cp='cp'
cp ~/py/scripts/createlinksalt.sh .
sh createlinksalt.sh 100 ../ ./

# now cd ${thedir}/movie1/ and edit makemovie.sh and __init__.local.py if necessary and run makemovie.sh 1 1 1




# Notes:
# thickdisk7:
# 0267 and 2292 are bad for thickdisk7, but newer one exists that overwrites it.


