#!/bin/bash

# create links in alternative location so can avoid doing all files -- just link some skipped sample of files so can only do that sample of files.

EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {skipfactor olddir newdir}"
    echo "e.g. sh createlinksalt.sh 100 /data2/jmckinne/thickdisk7/ ./"
    exit $E_BADARGS
fi

# get args
skipfactor=$1
oldpath=$2
newpath=$3


origfullpath=`pwd`

# get old directory
cd $oldpath
oldfullpath=`pwd`

# get new directory
cd $origfullpath
cd $newpath
newfullpath=`pwd`

mkdir -p $newfullpath
mkdir -p $newfullpath/dumps



# 2) make links

cd $newfullpath/dumps/


#allfiles=`ls -al ${oldfullpath}/dumps/fieldline*.bin | awk '{print $8}'`
#allfiles=`ls ${oldfullpath}/dumps/fieldline*.bin`
allfiles=`find ${oldfullpath}/dumps/ -name "fieldline*.bin"`
numfiles=`echo $allfiles | wc | awk '{print $2}'`

iii=0
jjj=0
for fil in $allfiles
do
    condition=$(( $iii % $skipfactor ))
    if [ $condition -eq 0 ]
	then
    #echo $fil
	ln -sf $fil .
	jjj=$(( $jjj + 1 ))
    fi
    iii=$(( $iii + 1 ))
done

echo "Skipped so that went from $numfiles files to $jjj files"

if [ -e $oldfullpath/dumps/gdump*bin ]
then
    ln -sf $oldfullpath/dumps/gdump*bin .
fi
if [ -e $oldfullpath/dumps/gdump ]
then
    ln -sf $oldfullpath/dumps/gdump .
fi
if [ -e $oldfullpath/dumps/dump0000 ]
then
    ln -sf $oldfullpath/dumps/dump0000 .
fi

cd $newfullpath/
ln -sf $oldfullpath/coordparms.dat .
ln -sf $oldfullpath/dimensions.txt .


