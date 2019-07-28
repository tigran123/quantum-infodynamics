function sigint() {
   echo -n "SIGINT: "
   pkill -P $$
   echo "killed all child processes"
   exit 1
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

PARAMS="-adaptive 1 -tol 0.01 \
        -mm 1 -mmsize 32 \
        -x0 0.0 -x1 -6.0 -x2 6.0 \
        -p0 2.0 -p1 -7.0 -p2 7.0 \
        -sigmax 0.3 -sigmap 0.6 \
        -Nx 512 -Np 512 \
        -t1 0.0 -t2 24 -N 100 \
        -u U_harmonic"

./solve.py -r -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (Q)" $PARAMS -s $SOLQR &
./solve.py -d "\$H(x,p)=p^2/2+x^2\$ (Q)" $PARAMS -s $SOLQN &
./solve.py -r -c -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (C)" $PARAMS -s $SOLCR &
./solve.py -c -d "\$H(x,p)=p^2/2+x^2\$ (C)" $PARAMS -s $SOLCN &
wait
