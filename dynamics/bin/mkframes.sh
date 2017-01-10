workdir=/tmp/tunnel
INIT=$workdir/init.npz
#SOLQ=$workdir/solq.npz
SOLC=$workdir/solc.npz

python3 solanim.py -P 6 -p 1 -d $workdir/frames -i $INIT -s $SOLC &
python3 solanim.py -P 6 -p 2 -d $workdir/frames -i $INIT -s $SOLC &
python3 solanim.py -P 6 -p 3 -d $workdir/frames -i $INIT -s $SOLC &
python3 solanim.py -P 6 -p 4 -d $workdir/frames -i $INIT -s $SOLC &
python3 solanim.py -P 6 -p 5 -d $workdir/frames -i $INIT -s $SOLC &
python3 solanim.py -P 6 -p 6 -d $workdir/frames -i $INIT -s $SOLC &
wait
