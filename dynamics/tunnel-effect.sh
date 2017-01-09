#workdir=$(mktemp -d ${TMPDIR:-/tmp}/solanim.XXXX)
workdir=/tmp/tunnel
mkdir -p $workdir/frames
echo "workdir=$workdir"
INIT_FILE=$workdir/init.npz
SOLQ=$workdir/solq.npz
SOLC=$workdir/solc.npz
WQ=$workdir/Wq.npz
WC=$workdir/Wc.npz
MOVIE_FILE=tunnel-effect-qc-nonrel.mp4

python3 mkinit.py -x1 -18.0 -x2 22.0 -Nx 1024 \
                  -p1 -5.0 -p2 5.0 -Np 1024 \
                  -t1 0.0 -t2 30 \
                  -f0 f0-gauss-x-4.py -u U_tun_nonrel.py -o $INIT_FILE

python3 prinit.py $INIT_FILE

exit

python3 solve.py -tol 0.01 -i $INIT_FILE -o $SOLQ -W $WQ &
python3 solve.py -c -tol 0.01 -i $INIT_FILE -o $SOLC -W $WC &
wait

python3 solanim.py -P 4 -p 1 -d $workdir/frames -i $INIT_FILE -s $SOLQ -s $SOLC &
python3 solanim.py -P 4 -p 2 -d $workdir/frames -i $INIT_FILE -s $SOLQ -s $SOLC &
python3 solanim.py -P 4 -p 3 -d $workdir/frames -i $INIT_FILE -s $SOLQ -s $SOLC &
python3 solanim.py -P 4 -p 4 -d $workdir/frames -i $INIT_FILE -s $SOLQ -s $SOLC &
wait

ffmpeg -loglevel quiet -y -r 25 -f image2 -i $workdir/frames/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r 25 $MOVIE_FILE
mpv --really-quiet -fs -loop -geometry +1200+0 $MOVIE_FILE
