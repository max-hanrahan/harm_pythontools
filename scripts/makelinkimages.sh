list=`ls lrhosmall*.png`

ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagesmall$ii.png
done

rm -rf imagesmall.mp4
ffmpeg -i imagesmall%d.png -r 25 -sameq imagesmall.mp4

#exit

list=`ls lrho*.png | grep -v lrhosmall`

ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagenormal$ii.png
done

rm -rf imagenormal.mp4
ffmpeg -i imagenormal%d.png -r 25 -sameq imagenormal.mp4


ffmpeg -framerate 10 -i image%d.png -sameq seedevo_v_hc.mp4
#-framerate works better than -r; has to come first
