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
    for ((stage=1; stage <= 3; stage++));
    do
        ./solanim.py -P $nproc -p $i -d $workdir/frames$stage -s ${QR}$stage -s ${CR}$stage -s ${QN}$stage -s ${CN}$stage &
    done
done
wait

ffmpeg -loglevel quiet -y -r $FPS -f image2 -pattern_type glob -i $workdir'/frames*/*.png' -f mp4 -q:v 0 -vcodec libx264 -r $FPS $MOVIE_FILE
