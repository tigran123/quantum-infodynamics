"""
  prinit.py --- Print initial data (init.npz)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  License: GPL
"""
from numpy import load, sum, set_printoptions, nan
from argparse import ArgumentParser as argp

p = argp(description="Print initial data")
p.add_argument("filename", action="store", help="Initial data file name")
p.add_argument("-d", action="store_true", help="Dump arrays (brief)", dest="dump_arrays")
p.add_argument("-df", action="store_true", help="Dump arrays (full)", dest="dump_arrays_full")
args = p.parse_args()

dump_arrays = args.dump_arrays
dump_arrays_full = args.dump_arrays_full
if dump_arrays_full: dump_arrays=True

with load(args.filename) as data:
    params = data['params'][()] # very mysterious indexing! ;)
    x1 = params['x1']; x2 = params['x2']; Nx = params['Nx']
    p1 = params['p1']; p2 = params['p2']; Np = params['Np']
    t1 = params['t1']; t2 = params['t2']
    Hmin = params['Hmin']; Hmax = params['Hmax']
    f0 = data['f0']; H = data['H']
    if dump_arrays:
        U = data['U']; qdU = data['qdU']; qdT = data['qdT']
        cdU = data['cdU']; cdT = data['cdT']

print("x1=% .3f, x2=% .3f, Nx=%d" % (x1,x2,Nx))
print("p1=% .3f, p2=% .3f, Np=%d" % (p1,p2,Np))
print("t1=% .3f, t2=% .3f" % (t1,t2))
print("Hmin=% .3f, Hmax=% .3f" % (Hmin,Hmax))

if dump_arrays_full: set_printoptions(threshold=nan)
if dump_arrays:
    print("f0=", f0)
    print("U=", U)
    print("H=", H)
    print("qdU=", qdU)
    print("qdT=", qdT)
    print("cdU=", cdU)
    print("cdT=", cdT)

dx = (x2-x1)/Nx
dp = (p2-p1)/Np
dmu = dx*dp
norm = sum(f0)*dmu
E = sum(f0*H)*dmu
print("Calculated norm of f0(x,p)=", norm)
print("Calculated energy E=", E)
