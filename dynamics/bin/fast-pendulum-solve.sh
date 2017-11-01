function sigint() {
   echo -n "SIGINT: "
   pkill -P $$
   echo "killed all child processes"
   exit
}

trap sigint SIGINT

workdir=$TMPDIR/fast-pendulum
rm -rf $workdir ; mkdir -p $workdir

#SOLQR=$workdir/solqr
#SOLCR=$workdir/solcr
#SOLQN=$workdir/solqn
SOLCN=$workdir/solcn

PARAMS="-N 2000 -x0 0 -p0 3 -sigmax 0.2 -sigmap 0.2 -x1 -2 -x2 40 -Nx 1024 -p1 -2 -p2 5 -Np 1024 -t1 0 -t2 7 -u U_pendulum"
#python3 solve.py -r -d "Quantum Relativistic Pendulum" $PARAMS -tol 0.001 -s $SOLQR &
#python3 solve.py -r -c -d "Classical Relativistic Pendulum" $PARAMS -tol 0.001 -s $SOLCR &
#python3 solve.py -d "Quantum Non-relativistic Pendulum" $PARAMS -tol 0.001 -s $SOLQN &
python3 solve.py -c -d "Mathematical Pendulum" $PARAMS -tol 0.0001 -s $SOLCN &
wait
