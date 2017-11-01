function sigint() {
   echo -n "SIGINT: "
   pkill -P $$
   echo "killed all child processes"
   exit
}

trap sigint SIGINT

workdir=$TMPDIR/fast-pendulum
mkdir -p $workdir/frames

#SOLQR=$workdir/solqr
#SOLCR=$workdir/solcr
#SOLQN=$workdir/solqn
SOLCN=$workdir/solcn

MOVIE_FILE=fast-pendulum-cn.mp4
FPS=20
nproc=$(nproc)
for ((i=1; i <= $nproc; i++));
do
    python3 solanim.py -W -P $nproc -p $i -d $workdir/frames -s $SOLCN &
done
wait

ffmpeg -loglevel quiet -y -r $FPS -f image2 -i $workdir/frames/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r $FPS $MOVIE_FILE

mpv -fs -loop -geometry +1200+0 $MOVIE_FILE
