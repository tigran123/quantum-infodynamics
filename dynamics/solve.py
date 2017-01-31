"""
   solve.py --- Quantum Infodynamics Solver (Spectral Split Propagator of Second Order with Adaptive Timestep Control)
   Author: Tigran Aivazian <aivazian.tigran@gmail.com>
   License: GPL
"""

from numpy import linspace, mgrid, pi, newaxis, exp, real, savez, amin, amax, sum, abs, memmap, sqrt, sign
import argparse as arg
from time import time

# select FFT implementation: pyfftw is the fastest and numpy is the slowest, scipy is in between these two
#from numpy.fft import ifftshift, fftshift, fft, ifft
#from scipy.fftpack import fftshift, ifftshift, fft, ifft
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifftshift, ifft
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(10)

p = arg.ArgumentParser(description="Quantum Infodynamics Tools - Equations Solver")
p.add_argument("-d",  action="store", help="Description text", dest="descr", required=True)
p.add_argument("-x0", action="store", help="Initial wave packet's x-coordinate", dest="x0", type=float, required=True)
p.add_argument("-p0", action="store", help="Initial wave packet's p-coordinate", dest="p0", type=float, required=True)
p.add_argument("-sigmax", action="store", help="Initial wave packet's x-deviation", dest="sigmax", type=float, required=True)
p.add_argument("-sigmap", action="store", help="Initial wave packet's p-deviation ", dest="sigmap", type=float, required=True)
p.add_argument("-x1", action="store", help="Starting coordinate", dest="x1", type=float, required=True)
p.add_argument("-x2", action="store", help="Final coordinate", dest="x2", type=float, required=True)
p.add_argument("-Nx", action="store", help="Number of points in x direction", dest="Nx", type=int, required=True)
p.add_argument("-p1", action="store", help="Starting momentum", dest="p1", type=float, required=True)
p.add_argument("-p2", action="store", help="Final momentum", dest="p2", type=float, required=True)
p.add_argument("-Np", action="store", help="Number of points in p direction", dest="Np", type=int, required=True)
p.add_argument("-t1", action="store", help="Starting time", dest="t1", type=float, required=True)
p.add_argument("-t2", action="store", help="Final time", dest="t2", type=float, required=True)
p.add_argument("-u",  action="store", help="Python source of U(x) and U'(x)", dest="srcU", required=True)
p.add_argument("-s",  action="store", help="Solution file name", dest="sfilename", required=True)
p.add_argument("-c",  action="store_true", help="Use classical (non-quantum) propagator", dest="classical")
p.add_argument("-r",  action="store_true", help="Use relativistic dynamics", dest="relat")
p.add_argument("-m",  action="store", help="Rest mass in a.u. (default=1.0)", type=float, dest="mass", default=1.0)
p.add_argument("-tol", action="store", help="Absolute error tolerance", dest="tol", type=float, required=True)
args = p.parse_args()

sfilename = args.sfilename
Wfilename = sfilename + '_W.npz'

def pr_exit(str):
    print("ERROR:" + str)
    exit()

# load the python module with the physical model (U(x) and dUdx(x))
Umod = __import__(args.srcU)

(descr,x0,p0,sigmax,sigmap,x1,x2,Nx,p1,p2,Np,t1,t2,tol,mass) = (args.descr,args.x0,args.p0,args.sigmax,args.sigmap,
                                    args.x1,args.x2,args.Nx,args.p1,args.p2,args.Np,args.t1,args.t2, args.tol,args.mass)
if tol <= 0: pr_exit("Tolerance must be positive, but %f <=0" % tol)
if mass < 0: pr_exit("Mass cannot be negative, but %f < 0" % mass)
if x2 <= x1: pr_exit("x2 must be greater than x1, but %f <= %f" %(x2,x1))
if p2 <= p1: pr_exit("p2 must be greater than p1, but %f <= %f" %(p2,p1))
if t2 <= t1: pr_exit("t2 must be greater than t1, but %f <= %f" %(t2,t1))
if Nx <= 0: pr_exit("Nx must be positive, but %d <= 0" % Nx)
if Np <= 0: pr_exit("Np must be positive, but %d <= 0" % Np)
if sigmax <= 0: pr_exit("sigmax must be positive, but %f <= 0" % sigmax)
if sigmap <= 0: pr_exit("sigmap must be positive, but %f <= 0" % sigmap)
if Nx & (Nx-1): print("WARNING: Nx=%d is not a power 2, FFT may be slowed down" % Nx)
if Np & (Np-1): print("WARNING: Np=%d is not a power 2, FFT may be slowed down" % Np)

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

def gauss(x, p, x0, p0, sigmax, sigmap):
    Z = 1./(2.*pi*sigmax*sigmap)
    return Z*exp(-((x-x0)**2/(2.*sigmax**2)+(p-p0)**2/(2.*sigmap**2)))

def qd(f, x, dx):
    hbar = 1.0 # Planck's constant in a.u.
    #hbar = 1.0545718e-34 # Planck's constant in J*s (SI)
    return (f(x+1j*hbar*dx/2.) - f(x-1j*hbar*dx/2.))/(1j*hbar)

c = 137.03604 # speed of light in a.u.

def dTdp_rel(p):
    if mass == 0.0:
        return c*sign(p)
    else:
        return c*p/sqrt(p**2 + mass**2*c**2)

(T,dTdp) = (lambda p: c*sqrt(p**2 + mass**2*c**2),dTdp_rel) if args.relat else (lambda p: p**2/(2.*mass),lambda p: p/mass)

(dU,dT) = (Umod.dUdx(X)*1j*Theta,-dTdp(P)*1j*Lam/2.) if args.classical else (qd(Umod.U,X,1j*Theta),qd(T,P,-1j*Lam)/2.)

Uv = Umod.U(xv)
Tv = T(pv)
H = T(pp)+Umod.U(xx)

def solve_spectral(Winit, expU, expT):
    B = fft(Winit, axis=0) # (x,p) -> (λ,p)
    B *= expT
    B = ifft(B, axis=0) # (λ,p) -> (x,p)
    B = fft(B, axis=1) # (x,p) -> (x,θ)
    B *= expU
    B = ifft(B, axis=1) # (x,θ) -> (x,p)
    B = fft(B, axis=0) # (x,p) -> (λ,p)
    B *= expT
    B = ifft(B, axis=0) # (λ,p) -> (x,p)
    return real(B) # to avoid python warning

def adjust_step(cur_dt, Winit, maxtries=15):
    tries = 0
    dt = cur_dt
    while True:
        tries += 1
        expU = exp(dt*dU)
        expT = exp(dt*dT)
        W1 = solve_spectral(Winit, expU, expT)
        expUn = exp(0.5*dt*dU)
        expTn = exp(0.5*dt*dT)
        W2 = solve_spectral(solve_spectral(Winit, expUn, expTn), expUn, expTn)
        if amax(abs(W2 - W1)) <= tol or tries > maxtries: break
        dt *= 0.7
    return (W1, dt, expU, expT)

t_start = time()
dt = (t2-t1)/20. # the first very rough guess of time step
W = [fftshift(gauss(xx,pp,x0,p0,sigmax,sigmap))]
tv = [t1]
t = t1
Nt = 1
while t <= t2:
    if Nt%300 == 299: print("%s: step %d"%(descr,Nt))
    if Nt%20 == 1:
        (Wnext, new_dt, expU, expT) = adjust_step(dt, W[-1])
        W.append(Wnext)
        if new_dt != dt:
            est_steps = (t2-t)//new_dt
            print("%s: step %d, adjusted dt %.4f -> %.4f, estimated %d steps left" %(descr,Nt,dt,new_dt,est_steps))
            dt = new_dt
    else:
        W.append(solve_spectral(W[-1], expU, expT))
    t += dt
    Nt += 1
    tv.append(t)

print("%s: solved in %8.2f seconds, %d steps" % (descr, time() - t_start, Nt))

W = ifftshift(W, axes=(1,2))
rho = sum(W, axis=2)*dp
phi = sum(W, axis=1)*dx
E = sum(H*W,axis=(1,2))*dx*dp
if args.relat: # so we can compare it with the non-relativistic kinetic energy
    Erest = mass*c**2
    E -= Erest
    Tv -= Erest

params = {'Wmin': amin(W), 'Wmax': amax(W), 'rho_min': amin(rho), 'rho_max': amax(rho),
          'Hmin': amin(H), 'Hmax': amax(H), 'Emin': amin(E), 'Emax': amax(E), 'H0': T(p0) + Umod.U(x0),
          'phi_min': amin(phi), 'phi_max': amax(phi), 'tol': tol, 'Wfilename': Wfilename, 'Nt': Nt,
          'x1': x1, 'x2': x2, 'Nx': Nx, 'p1': p1, 'p2': p2, 'Np': Np, 'descr': descr}

t_start = time()
savez(sfilename, t=tv, rho=rho, phi=phi, H=H, U=Uv, T=Tv, E=E, params=params)
fp = memmap(Wfilename, dtype='float64', mode='w+', shape=(Nt, Nx, Np))
fp[:] = W[:]
del fp # causes the flush of memmap
print("%s: solution saved in %8.2f seconds" % (descr, time() - t_start))
