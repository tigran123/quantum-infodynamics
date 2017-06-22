workdir=$(mktemp -d ${TMPDIR:-/tmp}/solanim.XXXX)
workdir=/tmp/harmonic-oscillator
rm -rf $workdir ; mkdir -p $workdir/frames

SOLQR=$workdir/solqr
SOLCR=$workdir/solcr
SOLQN=$workdir/solqn
SOLCN=$workdir/solcn
nproc=$(nproc)
FPS=20

MOVIE_FILE=harmonic-oscillator.mp4
PARAMS="-x0 0.0 -p0 1.0 -sigmax 0.2 -sigmap 0.1 -x1 -5.0 -x2 5.0 -Nx 512 -p1 -4.0 -p2 4.0 -Np 512 -t1 0.0 -t2 6.283185307179586 -u U_harmonic"

python3 solve.py -r -d "Quantum Relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLQR &
python3 solve.py -r -c -d "Classical Relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLCR &
python3 solve.py -d "Quantum Non-relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLQN &
python3 solve.py -c -d "Classical Non-relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLCN &
wait

for ((i=1; i <= $nproc; i++));
do
    python3 solanim.py -P $nproc -p $i -np -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
done
wait
ffmpeg -loglevel quiet -y -r $FPS -f image2 -i $workdir/frames/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r $FPS $MOVIE_FILE

#python3 solplay.py -o $MOVIE_FILE -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN
