workdir=$(mktemp -d ${TMPDIR:-/tmp}/solanim.XXXX)
mkdir -p $workdir/frames

SOLQR=$workdir/solqr
SOLCR=$workdir/solcr
SOLQN=$workdir/solqn
SOLCN=$workdir/solcn

MOVIE_FILE=harmonic-oscillator.mp4
PARAMS="-x1 -5.0 -x2 5.0 -Nx 512 -p1 -4.0 -p2 4.0 -Np 512 -t1 0.0 -t2 6.283185307179586 -f0 f0-gauss -u U_harmonic"

python3 solve.py -r -d "Quantum Relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLQR &
python3 solve.py -r -c -d "Classical Relativistic Oscillator" $PARAMS -tol 0.0009 -s $SOLCR &
python3 solve.py -d "Quantum Non-relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLQN &
python3 solve.py -c -d "Classical Non-relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLCN &
wait

python3 solanim.py -P 4 -p 1 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
python3 solanim.py -P 4 -p 2 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
python3 solanim.py -P 4 -p 3 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
python3 solanim.py -P 4 -p 4 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
wait

ffmpeg -loglevel quiet -y -r 25 -f image2 -i $workdir/frames/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r 25 $MOVIE_FILE
#mpv --really-quiet -fs -loop -geometry +1200+0 $MOVIE_FILE
