INIT="init.npz"
OUTQ="solq"
#OUTC="solc"

python3 mkinit.py -x1 -5.0 -x2 5.0 -Nx 256 \
                  -p1 -4.0 -p2 4.0 -Np 512 \
                  -t1 0.0 -t2 6.283185307179586 -tol 0.01 \
                  -f0 f0-gauss.py -u U_osc_rel.py -o $INIT

python3 prinit.py -i $INIT

python3 solve.py -i $INIT -o ${OUTQ}.npz
#python3 solve.py -i $INIT -o ${OUTC}.npz -c

rm -rf frames; mkdir frames
python3 solanim.py -i $INIT -s ${OUTQ}.npz

ffmpeg -loglevel quiet -y -r 25 -f image2 -i frames/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r 25 ${OUTQ}.mp4 
