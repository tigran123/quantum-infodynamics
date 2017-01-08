workdir=$(mktemp -d ${TMPDIR:-/tmp}/solanim.XXXX)
mkdir -p $workdir/frames
echo "workdir=$workdir"
INIT_FILE=$workdir/init.npz
SOL_FILE_QUANTUM=$workdir/solq.npz
SOL_FILE_CLASSICAL=$workdir/solc.npz

python3 mkinit.py -x1 -5.0 -x2 5.0 -Nx 1024 \
                  -p1 -4.0 -p2 4.0 -Np 1024 \
                  -t1 0.0 -t2 6.283185307179586 \
                  -f0 f0-gauss.py -u U_osc_rel.py -o $INIT_FILE

python3 prinit.py -i $INIT_FILE

python3 solve.py -tol 0.01 -i $INIT_FILE -o $SOL_FILE_QUANTUM &
python3 solve.py -tol 0.003 -i $INIT_FILE -o $SOL_FILE_CLASSICAL -c &
wait

#python3 solanim.py -d $workdir/frames -i $INIT_FILE -s $SOL_FILE_QUANTUM -s $SOL_FILE_CLASSICAL
python3 solanim.py -P 4 -p 1 -d $workdir/frames -i $INIT_FILE -s $SOL_FILE_QUANTUM -s $SOL_FILE_CLASSICAL &
python3 solanim.py -P 4 -p 2 -d $workdir/frames -i $INIT_FILE -s $SOL_FILE_QUANTUM -s $SOL_FILE_CLASSICAL &
python3 solanim.py -P 4 -p 3 -d $workdir/frames -i $INIT_FILE -s $SOL_FILE_QUANTUM -s $SOL_FILE_CLASSICAL &
python3 solanim.py -P 4 -p 4 -d $workdir/frames -i $INIT_FILE -s $SOL_FILE_QUANTUM -s $SOL_FILE_CLASSICAL &
wait

ffmpeg -loglevel quiet -y -r 25 -f image2 -i $workdir/frames/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r 25 osc_rel_quantum_classical.mp4 
