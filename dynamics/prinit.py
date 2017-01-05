"""
  prinit.py --- Print initial data (init.npz)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  License: GPL
"""
from numpy import load, sum, set_printoptions, nan
import argparse as arg

p = arg.ArgumentParser(description="Print initial data")
p.add_argument("-i", action="store", help="Initial data file name", dest="filename", required=True)
p.add_argument("-d", action="store_true", help="Dump arrays (brief)", dest="dump_arrays")
p.add_argument("-df", action="store_true", help="Dump arrays (full)", dest="dump_arrays_full")
args = p.parse_args()

filename = args.filename
dump_arrays = args.dump_arrays
dump_arrays_full = args.dump_arrays_full
if dump_arrays_full: dump_arrays=True

with load(filename) as data:
    (x1,x2,Nx,p1,p2,Np,t1,t2,tol,Hmin,Hmax) = data['params']
    f0 = data['f0']; H = data['H']
    if dump_arrays:
        U = data['U']; qdU = data['qdU'];   qdT = data['qdT']
        cdU = data['cdU'];   cdT = data['cdT']

print("x1=% .3f, x2=% .3f, Nx=%d" % (x1,x2,Nx))
print("p1=% .3f, p2=% .3f, Np=%d" % (p1,p2,Np))
print("t1=% .3f, t2=% .3f" % (t1,t2))
print("tol=%.4f" % tol)
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
