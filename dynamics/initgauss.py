#!/usr/bin/env python3.8

"""
   initgauss.py --- Quantum Infodynamics Tools (Cauchy Data Generator)
   Author: Tigran Aivazian <aivazian.tigran@gmail.com>
   License: GPL
"""

from argparse import ArgumentParser as argp
from numpy import array, zeros, mgrid, pi, exp, memmap, savez

p = argp(description="Quantum Infodynamics Tools - Gaussian Cauchy Data Generator")
p.add_argument("-x0", action="append", help="Initial packet's x-coordinate (multiple OK)", dest="x0", type=float, required=True, default=[])
p.add_argument("-p0", action="append", help="Initial packet's p-coordinate (multiple OK)", dest="p0", type=float, required=True, default=[])
p.add_argument("-sigmax", action="append", help="Initial packet's sigmax (multiple OK)", dest="sigmax", type=float, required=True, default=[])
p.add_argument("-sigmap", action="append", help="Initial packet's sigma (multiple OK)", dest="sigmap", type=float, required=True, default=[])
p.add_argument("-x1", action="store", help="Starting coordinate", dest="x1", type=float, required=True)
p.add_argument("-x2", action="store", help="Final coordinate", dest="x2", type=float, required=True)
p.add_argument("-Nx", action="store", help="Number of points in x direction", dest="Nx", type=int, required=True)
p.add_argument("-p1", action="store", help="Starting momentum", dest="p1", type=float, required=True)
p.add_argument("-p2", action="store", help="Final momentum", dest="p2", type=float, required=True)
p.add_argument("-Np", action="store", help="Number of points in p direction", dest="Np", type=int, required=True)
p.add_argument("-o",  action="store", help="Cauchy data file name", dest="ofilename", required=True)
args = p.parse_args()

ofilename = args.ofilename
Wfilename = ofilename + '_W.npy'

(x1,x2,Nx,p1,p2,Np) = (args.x1,args.x2,args.Nx,args.p1,args.p2,args.Np)
(x0,p0,sigmax,sigmap) = map(array, (args.x0, args.p0, args.sigmax, args.sigmap))

if Nx & (Nx-1): print("WARNING: Nx=%d is not a power 2, FFT may be slowed down" % Nx)
if Np & (Np-1): print("WARNING: Np=%d is not a power 2, FFT may be slowed down" % Np)

assert x2 > x1 and p2 > p1 and Nx > 0 and Np > 0
npoints = len(x0)
assert p0.shape == (npoints,) and sigmax.shape == (npoints,) and sigmap.shape == (npoints,)

def gauss(x, p, x0, p0, sigmax, sigmap):
    Z = 1./(2.*pi*sigmax*sigmap)
    return Z*exp(-((x-x0)**2/(2.*sigmax**2)+(p-p0)**2/(2.*sigmap**2)))

Winit = zeros((Nx,Np), dtype='float64')
dx = (x2-x1)/Nx
dp = (p2-p1)/Np
xx,pp = mgrid[x1:x2-dx:Nx*1j,p1:p2-dp:Np*1j]
for (ax0,ap0,asigmax,asigmap) in zip(x0, p0, sigmax, sigmap):
    Winit += gauss(xx,pp,ax0,ap0,asigmax,asigmap)
Winit /= npoints
W = memmap(Wfilename, dtype='float64', mode='w+', shape=(1, Nx, Np))
W[0] = Winit
W.flush()
del W
params = {'Wfilename': Wfilename, 'x0': x0, 'p0': p0, 'x1': x1, 'x2': x2, 'Nx': Nx, 'p1': p1, 'p2': p2, 'Np': Np, 'Nt': 1}
savez(ofilename, params=params)
