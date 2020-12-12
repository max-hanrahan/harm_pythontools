#!/bin/bash

#https://www.globusonline.org/globus_connect/gcforlinux/

doactivate=0

if [ $doactivate -eq 1 ]
then
    # first remove endpoint
    ssh pseudotensor@cli.globusonline.org endpoint-remove ki-jmck

    # if not already activated, get key:
    key=`ssh pseudotensor@cli.globusonline.org endpoint-add --gc ki-jmck | tail -1`

     # install certificate and key 
    sh globusconnect -setup $key

    # start server on local computer
    sh globusconnect -start &
fi

###################
# general copy

# get credential
ssh pseudotensor@cli.globusonline.org endpoint-activate xsede#nautilus

# copy from local computer to nautilus
#ssh pseudotensor@cli.globusonline.org scp pseudotensor#ki-jmck:/data2/jmckinne/makeallmovie.sh xsede#nautilus:/lustre/medusa/jmckinne/

dircollect='thickdisk7 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9full2pi sasham5 sasham2 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'

cd /data2/jmckinne/
mkdir -p /data2/jmckinne/alllinks/

cd /data2/jmckinne/alllinks/
for mydir in $dircollect
do
    mkdir $mydir
    cd $mydir
    ln -s /data2/jmckinne/$mydir/dumps/ .
    cd ..
done

# test:
# ssh pseudotensor@cli.globusonline.org mkdir xsede#nautilus:/lustre/medusa/jmckinne/thickdiskr15g/
# ssh pseudotensor@cli.globusonline.org scp -r pseudotensor#ki-jmck:/data2/jmckinne/thickdiskr15g/ xsede#nautilus:/lustre/medusa/jmckinne/thickdiskr15g/

for mydir in $dircollect
do
    echo "Doing $mydir"
    # create dirs
    ssh pseudotensor@cli.globusonline.org mkdir xsede#nautilus:/lustre/medusa/jmckinne/alllinks/
    ssh pseudotensor@cli.globusonline.org mkdir xsede#nautilus:/lustre/medusa/jmckinne/alllinks/$mydir/
    ssh pseudotensor@cli.globusonline.org mkdir xsede#nautilus:/lustre/medusa/jmckinne/alllinks/$mydir/dumps/
    ssh pseudotensor@cli.globusonline.org mkdir xsede#nautilus:/lustre/medusa/jmckinne/alllinks/$mydir/images/
    #
    #
    echo "Doing transfer for $mydir"
    # below works but stupid
    if [ 1 -eq 0 ]
    then
        cd /data2/jmckinne/alllinks/$mydir/dumps/
        alias ls='ls'
        myfiles=`ls fieldline*.bin gdump*.bin dump0000*.bin`
        #
        for myfile in `echo $myfiles`
        do
            echo "Doing transfer for dir=$mydir and file=$myfile"
            ssh pseudotensor@cli.globusonline.org scp pseudotensor#ki-jmck:/data2/jmckinne/alllinks/$mydir/dumps/$myfile xsede#nautilus:/lustre/medusa/jmckinne/alllinks/$mydir/dumps/$myfile
        done
    fi
    if [ 1 -eq 1 ]
    then
        echo "Doing transfer for dir=$mydir"
        ssh pseudotensor@cli.globusonline.org scp -r pseudotensor#ki-jmck:/data2/jmckinne/alllinks/$mydir/dumps/ xsede#nautilus:/lustre/medusa/jmckinne/alllinks/$mydir/dumps/
    fi

    echo "Done transfer for $mydir"
done


# just copy everything:
ssh pseudotensor@cli.globusonline.org scp -r pseudotensor#ki-jmck:/data2/jmckinne/ xsede#nautilus:/lustre/medusa/jmckinne/data2/jmckinne/
ssh pseudotensor@cli.globusonline.org scp -r pseudotensor#ki-jmck:/data1/jmckinne/ xsede#nautilus:/lustre/medusa/jmckinne/data1/jmckinne/


#ssh pseudotensor@cli.globusonline.org scp -r pseudotensor#ki-jmck:/data2/jmckinne/alllinks/ xsede#nautilus:/lustre/medusa/jmckinne/alllinks/


#ssh pseudotensor@cli.globusonline.org transfer pseudotensor#ki-jmck/data2/jmckinne/alllinks/ xsede#nautilus/lustre/medusa/jmckinne/alllinks/ -r



