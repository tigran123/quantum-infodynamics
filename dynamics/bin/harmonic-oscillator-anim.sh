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
mkdir -p $workdir/frames{1,2,3}

QR=$workdir/qr
CR=$workdir/cr
QN=$workdir/qn
CN=$workdir/cn

FPS=20

MOVIE_FILE=harmonic-oscillator-solanim.mp4

nproc=$(($(nproc)/3)) # divided by 3 because we create one process for each of the three stages of processing

for ((i=1; i <= $nproc; i++));
do
    ./solanim.py -P $nproc -p $i -d $workdir/frames1 -s $QR -s $CR -s $QN -s $CN &
    ./solanim.py -P $nproc -p $i -d $workdir/frames2 -s ${QR}2 -s ${CR}2 -s ${QN}2 -s ${CN}2 &
    ./solanim.py -P $nproc -p $i -d $workdir/frames3 -s ${QR}3 -s ${CR}3 -s ${QN}3 -s ${CN}3 &
done
wait

ffmpeg -loglevel quiet -y -r $FPS -f image2 -pattern_type glob -i $workdir'/frames?/*.png' -f mp4 -q:v 0 -vcodec libx264 -r $FPS $MOVIE_FILE
