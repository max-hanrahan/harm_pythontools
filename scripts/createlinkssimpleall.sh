dirrunsall='thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3  thickdisk17 thickdisk10 thickdisk15 thickdisk15r thickdisk2 thickdisk3 thickdisk33 runlocaldipole3dfiducial blandford3d_new sasham9 sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

# careful, dumps directory of the below directory will be wiped clean!
# removed 2D runs, blandford3d_new run since no new parts
# also removed thickdisk8 with no parts
dirrunswithparts='thickdisk11 thickdisk12 thickdisk13 thickdiskrr2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 thickdisk9 thickdiskr3  thickdisk17 thickdisk10 thickdisk15 thickdisk15r thickdisk2 thickdisk3 thickdisk33 runlocaldipole3dfiducial sasham9 sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

#dirrunswithparts='thickdisk8'

rm -rf badguys1.txt
rm -rf badguys2.txt
rm -rf badguys3.txt

for mydir in ${dirrunswithparts}
do

    sh createlinkssimple.sh $mydir

done