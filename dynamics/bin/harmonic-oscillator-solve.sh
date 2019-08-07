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

INIT=$workdir/init

QR=$workdir/qr
CR=$workdir/cr
QN=$workdir/qn
CN=$workdir/cn

INIT_PARAMS="-x0 0.0 -x1 -6.0 -x2 6.0 \
             -p0 2.0 -p1 -7.0 -p2 7.0 \
             -sigmax 0.3 -sigmap 0.6 \
             -Nx 512 -Np 512"

SOLVE_PARAMS="-adaptive Yes -tol 0.1 \
              -mm Yes -mmsize 32 \
              -N 100 -u U_harmonic"

./initgauss.py $INIT_PARAMS -o $INIT

( ./solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (Q)" -r $SOLVE_PARAMS -i $INIT  -o ${QR}1 -t1 0  -t2 6 && \
  ./solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (Q)" -r $SOLVE_PARAMS -i ${QR}1 -o ${QR}2 -t1 6  -t2 12 && \
  ./solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (Q)" -r $SOLVE_PARAMS -i ${QR}2 -o ${QR}3 -t1 12 -t2 18 ) &

( ./solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (C)" -r -c $SOLVE_PARAMS -i $INIT  -o ${CR}1 -t1 0  -t2 6 && \
  ./solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (C)" -r -c $SOLVE_PARAMS -i ${CR}1 -o ${CR}2 -t1 6  -t2 12 && \
  ./solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (C)" -r -c $SOLVE_PARAMS -i ${CR}2 -o ${CR}3 -t1 12 -t2 18 ) &

( ./solve.py -d "\$H(x,p)=p^2/2+x^2\$ (Q)" $SOLVE_PARAMS -i $INIT  -o ${QN}1 -t1 0  -t2 6 && \
  ./solve.py -d "\$H(x,p)=p^2/2+x^2\$ (Q)" $SOLVE_PARAMS -i ${QN}1 -o ${QN}2 -t1 6  -t2 12 && \
  ./solve.py -d "\$H(x,p)=p^2/2+x^2\$ (Q)" $SOLVE_PARAMS -i ${QN}2 -o ${QN}3 -t1 12 -t2 18 ) &

( ./solve.py -d "\$H(x,p)=p^2/2+x^2\$ (C)" -c $SOLVE_PARAMS -i $INIT  -o ${CN}1 -t1 0  -t2 6 && \
  ./solve.py -d "\$H(x,p)=p^2/2+x^2\$ (C)" -c $SOLVE_PARAMS -i ${CN}1 -o ${CN}2 -t1 6  -t2 12 && \
  ./solve.py -d "\$H(x,p)=p^2/2+x^2\$ (C)" -c $SOLVE_PARAMS -i ${CN}2 -o ${CN}3 -t1 12 -t2 18 ) &

wait
