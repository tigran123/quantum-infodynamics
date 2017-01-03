"""
   solve.py --- Quantum Infodynamics Solver
   Author: Tigran Aivazian <aivazian.tigran@gmail.com>
   License: GPL
"""

from numpy import load, exp, zeros, real, savez, amin, amax, sum, abs
import argparse as arg
from os.path import isfile

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
    x1 = data['x1']; x2 = data['x2']; Nx = data['Nx']
    p1 = data['p1']; p2 = data['p2']; Np = data['Np']
    t1 = data['t1']; t2 = data['t2']; tol = data['tol']
    W0 = fftshift(data['f0'])
    if args.classical:
        dU = data['cdU']; dT = data['cdT']
    else:
        dU = data['qdU']; dT = data['qdT']

def solve_spectral(Winit):
    global expU, expT

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

def find_optimal_time_step(dt_guess, maxtries=15):
    global expU, expT, W0, tol
    
    tries = 0
    dt = dt_guess
    while True:
        tries += 1
        expU = exp(dt*dU)
        expT = exp(dt*dT)
        W1 = solve_spectral(W0)
        dt *= 0.5
        expU = exp(dt*dU)
        expT = exp(dt*dT)
        W2 = solve_spectral(solve_spectral(W0))
        diff = amax(abs(W2-W1))
        if diff <= tol: return (2.*dt, tries)
        if tries > maxtries: break
    return dt

dp = (p2-p1)/Np
dx = (x2-x1)/Nx
dt = (t2-t1)/20. # the first very rough guess of time step
(dt,tries) = find_optimal_time_step(dt)
Nt = int((t2-t1)/dt)
dt =  (t2-t1)/(Nt-1) # adjust dt to fit whole number of times into t1...t2 interval
print("After %d attempts found optimal dt=" % tries, dt, "Nt=", Nt)

expU = exp(dt*dU)
expT = exp(dt*dT)

W = zeros((Nx, Np, Nt))
W[...,0] = W0

for k in range(Nt-1):
    W[...,k+1] = solve_spectral(W[...,k])
 
W = ifftshift(W, axes=(0,1))
(Wmin,Wmax) = (amin(W),amax(W))
rho = sum(W, axis=1)*dp
(rho_min,rho_max) = (amin(rho),amax(rho))
phi = sum(W, axis=0)*dp
(phi_min,phi_max) = (amin(phi), amax(phi))

ofilename = args.ofilename
if isfile(ofilename): print("WARNING: Overwriting file '%s'..." % (ofilename))

savez(ofilename, x1=x1, x2=x2, Nx=Nx, p1=p1, p2=p2, Np=Np, t1=t1, t2=t2, Nt=Nt,
                 W=W, Wmin=Wmin, Wmax=Wmax, rho=rho, rho_min=rho_min, rho_max=rho_max,
                 phi=phi, phi_min=phi_min, phi_max=phi_max)
