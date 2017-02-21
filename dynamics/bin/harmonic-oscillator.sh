workdir=$(mktemp -d ${TMPDIR:-/tmp}/solanim.XXXX)

SOLQR=$workdir/solqr
SOLCR=$workdir/solcr
SOLQN=$workdir/solqn
SOLCN=$workdir/solcn

MOVIE_FILE=harmonic-oscillator.mp4
PARAMS="-x0 0.0 -p0 1.0 -sigmax 0.2 -sigmap 0.1 -x1 -5.0 -x2 5.0 -Nx 512 -p1 -4.0 -p2 4.0 -Np 512 -t1 0.0 -t2 6.283185307179586 -u U_harmonic"

python3 solve.py -r -d "Quantum Relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLQR &
python3 solve.py -r -c -d "Classical Relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLCR &
python3 solve.py -d "Quantum Non-relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLQN &
python3 solve.py -c -d "Classical Non-relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLCN &
wait

#mkdir -p $workdir/frames
#python3 solanim.py -P 4 -p 1 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
#python3 solanim.py -P 4 -p 2 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
#python3 solanim.py -P 4 -p 3 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
#python3 solanim.py -P 4 -p 4 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
#wait
#ffmpeg -loglevel quiet -y -r 25 -f image2 -i $workdir/frames/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r 25 $MOVIE_FILE

python3 solplay.py -o $MOVIE_FILE -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN
