function sigint() {
   echo -n "SIGINT: "
   pkill -P $$
   echo "killed all child processes"
   exit 1
}

trap sigint SIGINT

#workdir=$TMPDIR/harmonic-oscillator
workdir=/data/work/harmonic-oscillator
#workdir=harmonic-oscillator
mkdir -p $workdir/frames

QR=$workdir/qr
CR=$workdir/cr
QN=$workdir/qn
CN=$workdir/cn

FPS=20

MOVIE_FILE=harmonic-oscillator-solanim.mp4

nproc=$(($(nproc)/3)) # divided by 3 because we create one process for each of the three stages of processing

IFS=$'\n'
declare -a arr=($(cat ${QR}-Nt.txt 2> /dev/null ; echo ; cat ${QN}-Nt.txt 2> /dev/null ; echo ; cat ${CR}-Nt.txt 2> /dev/null ; echo ; cat ${CN}-Nt.txt 2> /dev/null))

for ((i=1; i <= $nproc; i++));
do
    ./solanim.py -P $nproc -p $i -d $workdir/frames -s $QR -s $CR -s $QN -s $CN -ff 0 &
    ff=$(echo "${arr[*]}" | sort -nr | head -n1)
    ./solanim.py -P $nproc -p $i -d $workdir/frames -s ${QR}2 -s ${CR}2 -s ${QN}2 -s ${CN}2 -ff $ff &
    arr=($(cat ${QR}2-Nt.txt 2> /dev/null ; echo ; cat ${QN}2-Nt.txt 2> /dev/null ; echo ; cat ${CR}2-Nt.txt 2> /dev/null ; echo ; cat ${CN}2-Nt.txt 2> /dev/null ))
    ff=$((ff + $(echo "${arr[*]}" | sort -nr | head -n1)))
    ./solanim.py -P $nproc -p $i -d $workdir/frames -s ${QR}3 -s ${CR}3 -s ${QN}3 -s ${CN}3 -ff $ff &
done
wait

ffmpeg -loglevel quiet -y -r $FPS -f image2 -i $workdir/frames/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r $FPS $MOVIE_FILE
