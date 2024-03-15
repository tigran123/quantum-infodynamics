function sigint() {
   echo -n "SIGINT: "
   if pkill -P $$ ; then
       echo "$0: killed all child processes"
   else
       echo "$0: Failed to kill child processes"
   fi
   exit 1
}

trap sigint SIGINT

#workdir=$TMPDIR/harmonic-oscillator
#workdir=/data/work/harmonic-oscillator
workdir=harmonic-oscillator
rm -rf $workdir ; mkdir -p $workdir

INIT=$workdir/init
PYTHON=python3.12

QR=$workdir/qr
CR=$workdir/cr
QN=$workdir/qn
CN=$workdir/cn

INIT_PARAMS="-x0 0.0 -x1 -6.0 -x2 6.0 \
             -p0 2.0 -p1 -7.0 -p2 7.0 \
             -sigmax 0.3 -sigmap 0.6 \
             -Nx 512 -Np 512"

SOLVE_PARAMS="-adaptive Yes -tol 0.01 \
              -mm Yes -mmsize 32 \
              -N 60 -u U_harmonic"

$PYTHON initgauss.py $INIT_PARAMS -o $INIT

( $PYTHON solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (Q)" -r $SOLVE_PARAMS -i $INIT  -o ${QR}1 -t1 0  -t2 6 && \
  $PYTHON solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (Q)" -r $SOLVE_PARAMS -i ${QR}1 -o ${QR}2 -t1 6  -t2 12 && \
  $PYTHON solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (Q)" -r $SOLVE_PARAMS -i ${QR}2 -o ${QR}3 -t1 12 -t2 18 ) &

( $PYTHON solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (C)" -r -c $SOLVE_PARAMS -i $INIT  -o ${CR}1 -t1 0  -t2 6 && \
  $PYTHON solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (C)" -r -c $SOLVE_PARAMS -i ${CR}1 -o ${CR}2 -t1 6  -t2 12 && \
  $PYTHON solve.py -d "\$H(x,p)=\sqrt{1+p^2}+x^2\$ (C)" -r -c $SOLVE_PARAMS -i ${CR}2 -o ${CR}3 -t1 12 -t2 18 ) &

( $PYTHON solve.py -d "\$H(x,p)=p^2/2+x^2\$ (Q)" $SOLVE_PARAMS -i $INIT  -o ${QN}1 -t1 0  -t2 6 && \
  $PYTHON solve.py -d "\$H(x,p)=p^2/2+x^2\$ (Q)" $SOLVE_PARAMS -i ${QN}1 -o ${QN}2 -t1 6  -t2 12 && \
  $PYTHON solve.py -d "\$H(x,p)=p^2/2+x^2\$ (Q)" $SOLVE_PARAMS -i ${QN}2 -o ${QN}3 -t1 12 -t2 18 ) &

( $PYTHON solve.py -d "\$H(x,p)=p^2/2+x^2\$ (C)" -c $SOLVE_PARAMS -i $INIT  -o ${CN}1 -t1 0  -t2 6 && \
  $PYTHON solve.py -d "\$H(x,p)=p^2/2+x^2\$ (C)" -c $SOLVE_PARAMS -i ${CN}1 -o ${CN}2 -t1 6  -t2 12 && \
  $PYTHON solve.py -d "\$H(x,p)=p^2/2+x^2\$ (C)" -c $SOLVE_PARAMS -i ${CN}2 -o ${CN}3 -t1 12 -t2 18 ) &

wait
