"""
  mkinit.py --- Generate compressed initial data (init.npz)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  License: GPL
"""
import argparse as arg
from numpy import savez_compressed, mgrid, linspace, pi, newaxis, amin, amax
from os.path import isfile, splitext
from scipy.fftpack import fftshift, ifftshift, fft, ifft

p = arg.ArgumentParser(description="Generate compressed initial data")
p.add_argument("-o",  action="store", help="Output file name", dest="ofilename", required=True)
p.add_argument("-x1", action="store", help="Starting x-coordinate", dest="x1", type=float, required=True)
p.add_argument("-x2", action="store", help="Final x-coordinate", dest="x2", type=float, required=True)
p.add_argument("-Nx", action="store", help="Number of points in x direction", dest="Nx", type=int, required=True)
p.add_argument("-p1", action="store", help="Starting p-coordinate", dest="p1", type=float, required=True)
p.add_argument("-p2", action="store", help="Final p-coordinate", dest="p2", type=float, required=True)
p.add_argument("-Np", action="store", help="Number of points in p direction", dest="Np", type=int, required=True)
p.add_argument("-t1", action="store", help="Starting time", dest="t1", type=float, required=True)
p.add_argument("-t2", action="store", help="Final time", dest="t2", type=float, required=True)
p.add_argument("-tol", action="store", help="Absolute error tolerance", dest="tol", type=float, required=True)
p.add_argument("-f0", action="store", help="Python source of f0(x,p)", dest="srcf0", required=True)
p.add_argument("-u",  action="store", help="Python source of U(x), T(p), U'(x) and T'(p)", dest="srcU", required=True)
args = p.parse_args() # parse command-line arguments into args

# initialise our variables with the values passed via command-line
ofilename = args.ofilename
srcf0 = args.srcf0
srcU = args.srcU
(x1,x2,Nx,p1,p2,Np) = (args.x1,args.x2,args.Nx,args.p1,args.p2,args.Np)
(t1,t2,tol) = (args.t1,args.t2,args.tol)

def pr_exit(str):
    print("ERROR:" + str)
    exit()

# perform validation of all the passed values
if not isfile(srcf0): pr_exit("No such file '%s'" %(srcf0))
if not isfile(srcU): pr_exit("No such file '%s'" %(srcU))
if x2 <= x1: pr_exit("x2 must be greater than x1, but %f <= %f" %(x2,x1))
if p2 <= p1: pr_exit("p2 must be greater than p1, but %f <= %f" %(p2,p1))
if t2 <= t1: pr_exit("t2 must be greater than t1, but %f <= %f" %(t2,t1))
if Nx <= 0: pr_exit("Nx must be positive, but %d <= 0" %(Nx))
if Np <= 0: pr_exit("Np must be positive, but %d <= 0" %(Np))
if tol <= 0: pr_exit("Tolerance must be positive, but %d <= 0" %(tol))

# construct the mesh grid for evaluating f0(x,p), U(x), dUdx(x), T(p), dTdp(p)
xv,dx = linspace(x1, x2, Nx, endpoint=False, retstep=True)
pv,dp = linspace(p1, p2, Np, endpoint=False, retstep=True)
xx,pp = mgrid[x1:x2-dx:Nx*1j,p1:p2-dp:Np*1j]

# ranges in Fourier image spaces (theta is conjugated to p)
dtheta = 2.*pi/(p2-p1)
theta_amp = dtheta*Np/2.
thetav = linspace(-theta_amp, theta_amp - dtheta, Np)

# lam is conjugated to x
dlam = 2.*pi/(x2-x1)
lam_amp = dlam*Nx/2.
lamv = linspace(-lam_amp, lam_amp - dlam, Nx)

# now shift them all to center zero frequency
X = fftshift(xv)[:,newaxis]
P = fftshift(pv)[newaxis,:]
Theta = fftshift(thetav)[newaxis,:]
Lam = fftshift(lamv)[:,newaxis]

def qd(f, x, dx):
    """Quantum differential of function f(x) at a point x on the increment dx"""
    hbar = 1.0
    return (f(x+1j*hbar*dx/2.) - f(x-1j*hbar*dx/2.))/(1j*hbar)

# load the python modules with initial distribution and the physical model (U(x) and T(p))
f0mod = __import__(splitext(srcf0)[0])
Umod = __import__(splitext(srcU)[0])

qdU = qd(Umod.U, X, 1j*Theta)
qdT = qd(Umod.T, P, -1j*Lam)/2.
cdU = Umod.dUdx(X)*1j*Theta
cdT = -Umod.dTdp(P)*1j*Lam/2.

Hm = Umod.T(pp)+Umod.U(xx)
(Hmin,Hmax) = (amin(Hm),amax(Hm))

if isfile(ofilename): print("WARNING: Overwriting file '%s'..." % (ofilename))

# save data in the compressed .npz file which can be dumped by prinit.py and used by the solvers
savez_compressed(ofilename, x1=x1, x2=x2, Nx=Nx, p1=p1, p2=p2, Np=Np, t1=t1, t2=t2, tol=tol, f0=f0mod.f0(xx,pp),
                 U=Umod.U(xv), H=Hm, Hmin=Hmin, Hmax=Hmax, qdU=qdU, qdT=qdT, cdU=cdU, cdT=cdT)
