function sigint() {
   echo -n "SIGINT: "
   pkill -P $$
   echo "killed all child processes"
   exit
}

trap sigint SIGINT

#workdir=$TMPDIR/harmonic-oscillator
workdir=harmonic-oscillator-out
rm -rf $workdir ; mkdir -p $workdir

SOLQR=$workdir/solqr
SOLCR=$workdir/solcr
SOLQN=$workdir/solqn
SOLCN=$workdir/solcn

PARAMS="-x0 0.0 -p0 3.0 -sigmax 0.7 -sigmap 0.8 -x1 -6.0 -x2 6.0 -Nx 512 -p1 -7.0 -p2 7.0 -Np 512 -t1 0.0 -t2 12 -N 200 -u U_harmonic"

./solve.py -r -d "Quantum Relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLQR &
./solve.py -d "Quantum Non-relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLQN &
./solve.py -r -c -d "Classical Relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLCR &
./solve.py -c -d "Classical Non-relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLCN &
wait
#./solve.py -r -c -d "Classical Relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLCR
