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

SOLQR=$workdir/solqr
SOLCR=$workdir/solcr
SOLQN=$workdir/solqn
SOLCN=$workdir/solcn
FPS=20

MOVIE_FILE=harmonic-oscillator-solanim.mp4

nproc=$(nproc)
for ((i=1; i <= $nproc; i++));
do
    ./solanim.py -P $nproc -p $i -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
done
wait
ffmpeg -loglevel quiet -y -r $FPS -f image2 -i $workdir/frames/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r $FPS $MOVIE_FILE
