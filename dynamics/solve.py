"""
   solve.py --- Quantum Infodynamics Solver
   Author: Tigran Aivazian <aivazian.tigran@gmail.com>
   License: GPL
"""

from numpy import load, exp, zeros, real, savez, amin, amax, sum, abs
import argparse as arg
from os.path import isfile
from time import time

# select FFT implementation
#from numpy.fft import ifftshift, fftshift, fft, ifft
#from scipy.fftpack import fftshift, ifftshift, fft, ifft
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifftshift, ifft
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(10)

p = arg.ArgumentParser(description="Quantum Infodynamics Solver")
p.add_argument("-i", action="store", help="Initial data file name", dest="ifilename", required=True)
p.add_argument("-o", action="store", help="Solution file name", dest="ofilename", required=True)
p.add_argument("-c", action="store_true", help="Use classical (non-quantum) propagator", dest="classical")
args = p.parse_args()

with load(args.ifilename) as data:
    (x1,x2,Nx,p1,p2,Np,t1,t2,tol) = data['params'][:9]
    W0 = fftshift(data['f0'])
    if args.classical:
        dU = data['cdU']; dT = data['cdT']
    else:
        dU = data['qdU']; dT = data['qdT']

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
        delta = amax(abs(W2-W1))
        if delta <= tol or tries > maxtries: break
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
    if i%20 == 0:
        (Wnext, new_dt, expU, expT) = adjust_step(dt, W[i])
        W.append(Wnext)
        if new_dt != dt:
            print("Step %d, adjusted dt from"%i, dt, "to ", new_dt)
            dt = new_dt
    else:
        W.append(solve_spectral(W[i], expU, expT))
    t += dt
    i += 1
    tv.append(t)

W = ifftshift(W, axes=(1,2))
(Wmin,Wmax) = (amin(W),amax(W))
rho = sum(W, axis=2)*dp
(rho_min,rho_max) = (amin(rho),amax(rho))
phi = sum(W, axis=1)*dx
(phi_min,phi_max) = (amin(phi), amax(phi))
params = (Wmin,Wmax,rho_min,rho_max,phi_min,phi_max)

print("Solved in %8.3f seconds, %d steps" % (time() - t_start, len(tv)))

ofilename = args.ofilename
if isfile(ofilename): print("WARNING: Overwriting file '%s'..." % (ofilename))
savez(ofilename, t=tv, W=W, rho=rho, phi=phi, params=params)
