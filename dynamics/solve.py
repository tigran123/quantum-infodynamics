"""
   solve.py --- Quantum Infodynamics Solver
   Author: Tigran Aivazian <aivazian.tigran@gmail.com>
   License: GPL
"""

from numpy import linspace, mgrid, pi, newaxis, exp, real, savez, amin, amax, sum, abs, memmap
import argparse as arg
from time import time
from os.path import isfile, splitext

# select FFT implementation
#from numpy.fft import ifftshift, fftshift, fft, ifft
#from scipy.fftpack import fftshift, ifftshift, fft, ifft
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifftshift, ifft
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(10)

p = arg.ArgumentParser(description="Quantum Infodynamics Solver")
p.add_argument("-x1", action="store", help="Starting coordinate", dest="x1", type=float, required=True)
p.add_argument("-x2", action="store", help="Final coordinate", dest="x2", type=float, required=True)
p.add_argument("-Nx", action="store", help="Number of points in x direction", dest="Nx", type=int, required=True)
p.add_argument("-p1", action="store", help="Starting momentum", dest="p1", type=float, required=True)
p.add_argument("-p2", action="store", help="Final momentum", dest="p2", type=float, required=True)
p.add_argument("-Np", action="store", help="Number of points in p direction", dest="Np", type=int, required=True)
p.add_argument("-t1", action="store", help="Starting time", dest="t1", type=float, required=True)
p.add_argument("-t2", action="store", help="Final time", dest="t2", type=float, required=True)
p.add_argument("-f0", action="store", help="Python source of f0(x,p)", dest="srcf0", required=True)
p.add_argument("-u",  action="store", help="Python source of U(x), T(p), U'(x) and T'(p)", dest="srcU", required=True)
p.add_argument("-o", action="store", help="Solution file name", dest="ofilename", required=True)
p.add_argument("-W", action="store", help="Solution W(x,p,t) file name", dest="Wfilename", required=True)
p.add_argument("-c", action="store_true", help="Use classical (non-quantum) propagator", dest="classical")
p.add_argument("-tol", action="store", help="Absolute error tolerance", dest="tol", type=float, required=True)
args = p.parse_args()

def pr_exit(str):
    print("ERROR:" + str)
    exit()

srcf0 = args.srcf0; srcU = args.srcU
(x1,x2,Nx,p1,p2,Np,t1,t2,tol) = (args.x1,args.x2,args.Nx,args.p1,args.p2,args.Np,args.t1,args.t2,args.tol)
if tol <= 0: pr_exit("Tolerance value must be positive, but %f <=0" % tol)
if not isfile(srcf0): pr_exit("No such file '%s'" %(srcf0))
if not isfile(srcU): pr_exit("No such file '%s'" %(srcU))
if x2 <= x1: pr_exit("x2 must be greater than x1, but %f <= %f" %(x2,x1))
if p2 <= p1: pr_exit("p2 must be greater than p1, but %f <= %f" %(p2,p1))
if t2 <= t1: pr_exit("t2 must be greater than t1, but %f <= %f" %(t2,t1))
if Nx <= 0: pr_exit("Nx must be positive, but %d <= 0" % Nx)
if Np <= 0: pr_exit("Np must be positive, but %d <= 0" % Np)
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

def qd(f, x, dx):
    """Quantum differential of function f(x) at a point x on the increment dx"""
    hbar = 1.0
    return (f(x+1j*hbar*dx/2.) - f(x-1j*hbar*dx/2.))/(1j*hbar)

# load the python modules with initial distribution and the physical model (U(x) and T(p))
f0mod = __import__(splitext(srcf0)[0])
Umod = __import__(splitext(srcU)[0])

if args.classical:
    method = "CLASS"
    dU = Umod.dUdx(X)*1j*Theta
    dT = -Umod.dTdp(P)*1j*Lam/2.
else:
    method = "QUANT"
    dU = qd(Umod.U, X, 1j*Theta)
    dT = qd(Umod.T, P, -1j*Lam)/2.

H = Umod.T(pp)+Umod.U(xx)
x0 = f0mod.x0
p0 = f0mod.p0

t_start = time()
W0 = fftshift(f0mod.f0(xx,pp))

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
    global tol, dU, dT
    
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
        dt *= 0.6
    return (W1, dt, expU, expT)

t_start = time()
dp = (p2-p1)/Np
dx = (x2-x1)/Nx
dt = (t2-t1)/20. # the first very rough guess of time step
W = [W0]
tv = [t1]
t = t1
i = 0
while t <= t2:
    if i%300 == 299: print("%s: step %d"%(method,i))
    if i%20 == 0:
        (Wnext, new_dt, expU, expT) = adjust_step(dt, W[i])
        W.append(Wnext)
        if new_dt != dt:
            est_steps = (t2-t)//new_dt
            print("%s: step %d, adjusted dt %.3f -> %.3f, estimated %d steps left" %(method,i,dt,new_dt,est_steps))
            dt = new_dt
    else:
        W.append(solve_spectral(W[i], expU, expT))
    t += dt
    i += 1
    tv.append(t)

W = ifftshift(W, axes=(1,2))
rho = sum(W, axis=2)*dp
phi = sum(W, axis=1)*dx
Nt = len(tv)
params = {'Wmin': amin(W), 'Wmax': amax(W), 'rho_min': amin(rho), 'rho_max': amax(rho), 'Hmin': amin(H), 'Hmax': amax(H),
          'phi_min': amin(phi), 'phi_max': amax(phi), 'tol': tol, 'Wfilename': args.Wfilename, 'Nt': Nt,
          'x0': x0, 'p0': p0, 'x1': x1, 'x2': x2, 'Nx': Nx, 'p1': p1, 'p2': p2, 'Np': Np}

print("%s: solved in %8.2f seconds, %d steps" % (method, time() - t_start, Nt))

t_start = time()
savez(args.ofilename, t=tv, rho=rho, phi=phi, H=H, params=params)
fp = memmap(args.Wfilename, dtype='float64', mode='w+', shape=(Nt, Nx, Np))
fp[:] = W[:]
del fp
print("%s: solution saved in %8.2f seconds" % (method, time() - t_start))
