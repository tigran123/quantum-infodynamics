workdir=/tmp/free-particles
rm -rf $workdir ; mkdir -p $workdir/frames{1,2,3}

LUXON=$workdir/luxon
TARDYON=$workdir/tardyon
TACHYON=$workdir/tachyon
nproc=$(nproc)
FPS=20

PARAMS="-r -x0 0.0 -p0 4.0 -sigmax 0.2 -sigmap 0.2 -x1 -20.0 -x2 40.0 -Nx 256 -p1 3.0 -p2 5.0 -Np 256 -t1 0.0 -t2 25.0 -u U_free"

python3 solve.py -d "Free Luxon (m=0, v=c)"    -m 0.0  $PARAMS -tol 0.00001 -s $LUXON  &
python3 solve.py -d "Free Tardyon (m=3, v=0.8c)"  -m 3.0  $PARAMS -tol 0.00001 -s $TARDYON &
python3 solve.py -d "Free Tachyon (m=3i, v=1.5c)" -m 3.0j $PARAMS -tol 0.1  -s $TACHYON &
wait

for ((i=1; i <= $nproc; i++));
do
    python3 solanim.py -P $nproc -p $i -np -d $workdir/frames1 -s $TARDYON &
done
wait
ffmpeg -loglevel quiet -y -r $FPS -f image2 -i $workdir/frames1/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r $FPS tardyon.mp4 &

for ((i=1; i <= $nproc; i++));
do
    python3 solanim.py -P $nproc -p $i -np -d $workdir/frames2 -s $LUXON &
done
wait
ffmpeg -loglevel quiet -y -r $FPS -f image2 -i $workdir/frames2/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r $FPS luxon.mp4 &

for ((i=1; i <= $nproc; i++));
do
    python3 solanim.py -P $nproc -p $i -np -d $workdir/frames3 -s $TACHYON &
done
wait
ffmpeg -loglevel quiet -y -r $FPS -f image2 -i $workdir/frames3/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r $FPS tachyon.mp4

#python3 solplay.py -o $MOVIE_FILE -np -s $LUXON -s $TARDYON -s $TACHYON

#mpv -fs -loop -geometry +1200+0 $MOVIE_FILE
