#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things


origpwd=`pwd`

EXPECTED_ARGS=1
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {dirname without trailing /}"
    echo "e.g. sh createlinkes.sh thickdiskrr2"
    exit $E_BADARGS
fi

thedir=$1

mkdir -p $thedir

# On ki-jmck:

# 1) Ensure sizes all correctly large
#ls -alRS thickdisk11* | grep fieldline | grep -v "\->" | sort -nrk 5

# 2) If not all large, then order list of dirs in order of simulation so overwrites use newer file and repeat

# 3) setup directory list

echo "compute1"
rm -rf dirs${thedir}.txt
alias ls='ls'
alias lssdir='ls -ap | grep / | sed "s/\///"'
list0=`lssdir  | grep ${thedir} | grep -v nextnextnextnext | grep -v nextnextnext | grep -v nextnext | grep -v next`
list1=`lssdir  | grep ${thedir} | grep next | grep -v nextnextnextnext | grep -v nextnextnext | grep -v nextnext`
list2=`lssdir  | grep ${thedir} | grep nextnext | grep -v nextnextnextnext | grep -v nextnextnext`
list3=`lssdir  | grep ${thedir} | grep nextnextnext | grep -v nextnextnextnext`
list4=`lssdir  | grep ${thedir} | grep nextnextnextnext`
listfinal=`echo $list0 $list1 $list2 $list3 $list4`
list=`echo $listfinal | sed 's/'${thedir}'\///g' | sed 's/'${thedir}' //g'`

listfinalnum=`echo $listfinal | wc -w`
listnum=`echo $list | wc -w`

if [ $listfinalnum -eq $listnum ]
then
    echo "listfinalnum and listnum are same!  Didn't remove ${thedir}"
    echo "begin echo of listfinal"
    echo $listfinal
    echo "begin echo of list"
    echo $list
    echo "${thedir}" >> badguys1_${thedir}.txt
    exit
fi

#list=`echo $listfinal | sed 's/'${thedir}'\///g' | sed 's/'${thedir}' //g' | sed 's/' ${thedir}'//g'`
#
echo "begin echo of list"
echo $list
echo "end echo of list"
for fil in $list ; do echo $fil >> dirs${thedir}.txt ; done

#ls -alrt dirs${thedir}.txt

if [ -e "dirs${thedir}.txt" ]
then
    echo "Got some subparts in dirs${thedir}.txt"
else
    echo "Got no subparts in dirs${thedir}.txt"
    echo "${thedir}" >> badguys2_${thedir}.txt
    exit
fi

# check that not killing original directory with actual dump data
dirparts=`cat dirs${thedir}.txt`
echo $dirparts
numdirparts=`echo $dirparts | wc -w`
echo "Number of parts for ${thedir} is $numdirparts"

if [ $numdirparts -lt 1 ]
then
    echo "No sub parts, assume mistake and has original dumps"
    echo "${thedir}" >> badguys3_${thedir}.txt
    exit
fi


echo "compute2"
if [ "thickdisk7" == "${thedir}" ]
then
    manydir=`pwd`
    basedir=`pwd`
    dumpsdir=$basedir/dumps/
    mkdir -p dumps/
    #rm -rf $dumpsdir/fieldline*.bin
    #rm -rf $dumpsdir/gdump*.bin
    #rm -rf $dumpsdir/dump*.bin
else
    manydir=`pwd`
    cd ${thedir}/
    basedir=`pwd`
    dumpsdir=$basedir/dumps/
    mkdir -p dumps/
    #rm -rf $dumpsdir/fieldline*.bin
    #rm -rf $dumpsdir/gdump*.bin
    #rm -rf $dumpsdir/dump*.bin
    mv ../dirs${thedir}.txt .
fi

# 4) Edit dir list and choose one's want



# 5) create new full-sim dir and change to dumps dir

echo "compute3"
cd $dumpsdir
#rm -rf fieldline*.bin
#rm -rf dump0000.bin
#rm -rf gdump.bin

#exit

# 6) make links
echo "compute4"
sleep 1


for mydir in `cat ../dirs${thedir}.txt`
do

    # avoid using directory itself in case didn't remove base directory
    if [ "${mydir}" == "${thedir}" ]
    then
        continue
    fi

    echo $mydir
    for fil in `ls $manydir/$mydir/dumps/fieldline*.bin` 
    do
        #echo $fil 
        ln -sf $fil .
    done


    # 7) Also make links to gdump.bin and dump0000.bin
    #firstdir=`head -1 ../dirs${thedir}.txt`
    if [ -e $manydir/$mydir/dumps/gdump.bin ]
    then
        ln -s $manydir/$mydir/dumps/gdump.bin .
        ln -s $manydir/$mydir/dumps/dump0000.bin .
    fi
    if [ -e $manydir/$mydir/dumps/gdump.THETAROT0.bin ]
    then
        ln -s $manydir/$mydir/dumps/gdump.THETAROT0.bin .
    fi
    if [ -e $manydir/$mydir/coordparms.dat ]
    then
        cd ../
        ln -s $manydir/$mydir/coordparms.dat .
        cd $dumpsdir
    fi
    if [ -e $manydir/$mydir/dimensions.txt ]
    then
        cd ../
        ln -s $manydir/$mydir/dimensions.txt .
        cd $dumpsdir
    fi
    if [ -e $manydir/$mydir/nprlistinfo.dat ]
    then
        cd ../
        ln -s $manydir/$mydir/nprlistinfo.dat .
        cd $dumpsdir
    fi
done


cd ..

echo " "
echo $list
echo "Number of parts for ${thedir} is $numdirparts"



cd $origpwd


echo "Checking if file sizes indicate each file is valid"
allfiles=`ls -alRS ${thedir}* | grep fieldline | grep -v "\->" | sort -nrk 5`
topsize=`echo "$allfiles" | head -1 | awk '{print $5}'`

echo "topsize=$topsize"
# just look at top 20 files:
# first sed gets rid of deadfile and \\* required to grab *
filestocheck=`echo "$allfiles" | egrep 'fieldline[0-9]+\.bin' | tail -20 | awk '{print $8}' | sed -e 's/deadfieldline[0-9]\+\.bin\\*//g' | sed -e 's/\*//g' | sed -e 's/deadfieldline[0-9]\+\.bin//g'`

for fil in $filestocheck ; do echo $fil ; newfil=`ls -alrt $dumpsdir/$fil | awk '{print $10}' | sed -e 's/\.\.\/\.\.\///g'` ; ls -alrt $newfil ; done

echo "topsize=$topsize"
lastmydir=`cat $basedir/dirs${thedir}.txt | tail -1`
echo "lastmydir=$lastmydir"



#cp -a ../thickdisk9/movie1 .
#cd movie1/
#rm -rf *.png *.eps *.npy python*.out *.avi *.pdf
#alias cp='cp'
#cp ~/py/scripts/createlinksalt.sh .
#sh createlinksalt.sh 100 ../ ./

# now cd ${thedir}/movie1/ and edit makemovie.sh and __init__.local.py if necessary and run makemovie.sh 1 1 1




# Notes:
# thickdisk7:
# 0267 and 2292 are bad for thickdisk7, but newer one exists that overwrites it.


