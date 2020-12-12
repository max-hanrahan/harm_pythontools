#!/bin/bash

#dircollecttilt="thickdiskfull3d7tilt0.7 thickdiskfull3d7tilt1.5708 sashaam9full2pit0.15 sashaa9b100t0.15 sashaa99t0.15 sashaam9full2pit0.3 sashaa9b100t0.3 sashaa99t0.3 sashaam9full2pit0.6 sashaa9b100t0.6 sashaa99t0.6 sashaam9full2pit1.5708 sashaa9b100t1.5708 sashaa99t1.5708"

dircollecttilt="thickdiskfull3d7tilt0.35"





for fil in $dircollecttilt
do

    ssh jmckinne@ki-jmck.slac.stanford.edu "mkdir -p /home/jmckinne/testall/$fil/only7files/"
    
    # for the ~500 small files:
    ssh pseudotensor@cli.globusonline.org scp -D -r -s 2 xsede#nautilus:/lustre/medusa/jmckinne/data3/jmckinne/jmckinne/$fil/only7files pseudotensor#ki-jmck:/home/jmckinne/testall/$fil/

    # for linked big files:
    scp -rp /lustre/medusa/jmckinne/data3/jmckinne/jmckinne/$fil/only7files/dumps/ ki-jmck.slac.stanford.edu:/home/jmckinne/testall/$fil/only7files/

done