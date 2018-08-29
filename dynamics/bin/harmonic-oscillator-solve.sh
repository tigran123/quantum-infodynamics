function sigint() {
   echo -n "SIGINT: "
   pkill -P $$
   echo "killed all child processes"
   exit
}

trap sigint SIGINT

workdir=$TMPDIR/harmonic-oscillator
rm -rf $workdir ; mkdir -p $workdir

SOLQR=$workdir/solqr
SOLCR=$workdir/solcr
SOLQN=$workdir/solqn
SOLCN=$workdir/solcn
FPS=20

PARAMS="-x0 0.0 -p0 1.0 -sigmax 0.2 -sigmap 0.1 -x1 -5.0 -x2 5.0 -Nx 512 -p1 -4.0 -p2 4.0 -Np 512 -t1 0.0 -t2 6.283185307179586 -u U_harmonic"

./solve.py -r -d "Quantum Relativistic Oscillator" $PARAMS -tol 0.007 -s $SOLQR &
./solve.py -r -c -d "Classical Relativistic Oscillator" $PARAMS -tol 0.0005 -s $SOLCR &
./solve.py -d "Quantum Non-relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLQN &
./solve.py -c -d "Classical Non-relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLCN &
wait
