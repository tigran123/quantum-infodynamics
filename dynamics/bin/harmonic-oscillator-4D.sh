workdir=harmonic-4D
rm -rf $workdir ; mkdir -p $workdir/frames

SOLQR=$workdir/solqr
SOLCR=$workdir/solcr
SOLQN=$workdir/solqn
SOLCN=$workdir/solcn

MOVIE_FILE=harmonic-oscillator-4D.mp4
PARAMS="-x0  0.0 -y0  0.0 \
        -px0 1.0 -py0 0.0 \
        -sigmax 0.2 -sigmay 0.2 -sigmapx 0.1 -sigmapy 0.1 \
        -x1 -5.0 -x2 5.0 \
        -y1 -5.0 -y2 5.0 \
        -px1 -4.0 -px2 4.0 \
        -py1 -4.0 -py2 4.0 \
        -Nx 32 -Ny 32 -Npx 128 -Npy 128 \
        -t1 0.0 -t2 3.1415926 \
        -u U_harmonic_4D"

python3 solve4D.py -c -d "Classical Non-relativistic Oscillator" $PARAMS -tol 0.1 -s $SOLCN
exit

python3 solve4D.py -r -d "Quantum Relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLQR &
python3 solve4D.py -r -c -d "Classical Relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLCR &
python3 solve4D.py -d "Quantum Non-relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLQN &
python3 solve4D.py -c -d "Classical Non-relativistic Oscillator" $PARAMS -tol 0.01 -s $SOLCN &
wait

#python3 solanim.py -P 4 -p 1 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
#python3 solanim.py -P 4 -p 2 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
#python3 solanim.py -P 4 -p 3 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
#python3 solanim.py -P 4 -p 4 -d $workdir/frames -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN &
#wait
#ffmpeg -loglevel quiet -y -r 25 -f image2 -i $workdir/frames/%05d.png -f mp4 -q:v 0 -vcodec libx264 -r 25 $MOVIE_FILE
python3 solplay.py -o $MOVIE_FILE -s $SOLQR -s $SOLCR -s $SOLQN -s $SOLCN
