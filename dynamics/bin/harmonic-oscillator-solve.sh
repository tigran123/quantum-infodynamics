function sigint() {
   echo -n "SIGINT: "
   pkill -P $$
   echo "killed all child processes"
   exit
}

trap sigint SIGINT

#workdir=$TMPDIR/harmonic-oscillator
workdir=/data/work/harmonic-oscillator
#workdir=harmonic-oscillator
rm -rf $workdir ; mkdir -p $workdir

SOLQR=$workdir/solqr
SOLCR=$workdir/solcr
SOLQN=$workdir/solqn
SOLCN=$workdir/solcn

PARAMS="-x0 0.0 -x1 -6.0 -x2 6.0 \
        -p0 3.0 -p1 -7.0 -p2 7.0 \
        -sigmax 0.3 -sigmap 0.6 \
        -Nx 512 -Np 512 \
        -t1 0.0 -t2 24 -N 200 \
        -u U_harmonic"

./solve.py -r -d "Quantum Relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLQR &
./solve.py -d "Quantum Non-relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLQN &
./solve.py -r -c -d "Classical Relativistic Oscillator" $PARAMS -tol 0.005 -s $SOLCR &
./solve.py -c -d "Classical Non-relativistic Oscillator" $PARAMS -tol 0.001 -s $SOLCN &
wait
