"""
   solve.py --- Quantum Infodynamics Solver
   Author: Tigran Aivazian <aivazian.tigran@gmail.com>
   License: GPL
"""

from numpy import load, exp, zeros, real, savez, amin, amax, sum, abs, memmap
import argparse as arg
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
p.add_argument("-W", action="store", help="Solution W(x,p,t) file name", dest="Wfilename", required=True)
p.add_argument("-c", action="store_true", help="Use classical (non-quantum) propagator", dest="classical")
p.add_argument("-tol", action="store", help="Absolute error tolerance", dest="tol", type=float, required=True)
args = p.parse_args()

def pr_exit(str):
    print("ERROR:" + str)
    exit()

tol = args.tol
if tol <= 0: pr_exit("Tolerance value must be positive, but %f <=0" % tol)

t_start = time()
with load(args.ifilename) as data:
    params = data['params'][()] # very mysterious indexing! ;)
    x1 = params['x1']; x2 = params['x2']; Nx = params['Nx']
    p1 = params['p1']; p2 = params['p2']; Np = params['Np']
    t1 = params['t1']; t2 = params['t2']
    W0 = fftshift(data['f0'])
    if args.classical:
        method = "Classical"
        dU = data['cdU']; dT = data['cdT']
    else:
        method = "Quantum"
        dU = data['qdU']; dT = data['qdT']
print("%s: initial data loaded in %8.2f seconds" % (method, time() - t_start))

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
params = {'Wmin': amin(W), 'Wmax': amax(W), 'rho_min': amin(rho), 'rho_max': amax(rho),
          'phi_min': amin(phi), 'phi_max': amax(phi), 'tol': tol, 'Wfilename': args.Wfilename, 'Nt': Nt}

print("%s: solved in %8.2f seconds, %d steps" % (method, time() - t_start, Nt))

t_start = time()
savez(args.ofilename, t=tv, rho=rho, phi=phi, params=params)
fp = memmap(args.Wfilename, dtype='float64', mode='w+', shape=(Nt, Nx, Np))
fp[:] = W[:]
del fp
print("%s: solution saved in %8.2f seconds" % (method, time() - t_start))
