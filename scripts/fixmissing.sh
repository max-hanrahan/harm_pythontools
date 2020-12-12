#header="stream"
#header="initfinal"
header=$1


linklist=`find -L ./ -type l -lname '$header*' -exec ls  {} \;`
# delete links
find -L ./ -type l -lname $header'*' -exec rm {} \;


sortedlist=`ls ${header}*.png | sed 's/@//g' | sed 's/'$header'//g' | sed 's/\.png//g' | sort -n`

initialnum=`echo "$sortedlist" | head -1`
finalnum=`echo "$sortedlist" | tail -1`

echo "seq $initialnum $finalnum"
list=`seq $initialnum $finalnum`

#exit

for fnum in $list
do

snum=$(printf "%04d" "$fnum")
fil=${header}${snum}.png
echo "Trying $fil"

  if [ -e $fil ]
      then
      echo "Exists: $fil"
      # first file should always exist
      lastexistfnum=$fnum
  else
      lastsnum=$(printf "%04d" "$lastexistfnum")
      fillast=${header}${lastsnum}.png
      ln -s $fillast $fil
      echo "Made link: $fillast $fil"
  fi


done
