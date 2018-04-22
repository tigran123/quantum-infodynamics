"""
   solve.py --- Quantum Infodynamics Solver (Spectral Split Propagator of Second Order with Adaptive Timestep Control)
   Author: Tigran Aivazian <aivazian.tigran@gmail.com>
   License: GPL
"""

from numpy import linspace, mgrid, pi, newaxis, exp, real, savez, amin, amax, sum, abs, memmap, sqrt, sign, zeros, array
import argparse as arg
from time import time

p = arg.ArgumentParser(description="Quantum Infodynamics Tools - Equations Solver")
p.add_argument("-d",  action="store", help="Description text", dest="descr", required=True)
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
p.add_argument("-t1", action="store", help="Starting time", dest="t1", type=float, required=True)
p.add_argument("-t2", action="store", help="Final time", dest="t2", type=float, required=True)
p.add_argument("-u",  action="store", help="Python source of U(x) and U'(x)", dest="srcU", required=True)
p.add_argument("-s",  action="store", help="Solution file name", dest="sfilename", required=True)
p.add_argument("-c",  action="store_true", help="Use classical (non-quantum) propagator", dest="classical")
p.add_argument("-r",  action="store_true", help="Use relativistic dynamics", dest="relat")
p.add_argument("-m",  action="store", help="Rest mass in a.u. (default=1.0)", type=complex, dest="mass", default=1.0)
p.add_argument("-N",  action="store", help="Initial number of time steps (default=50)", dest="N", type=int, default=50)
p.add_argument("-tol", action="store", help="Absolute error tolerance", dest="tol", type=float, required=True)
args = p.parse_args()

sfilename = args.sfilename
Wfilename = sfilename + '_W.npz'

(descr,x1,x2,Nx,p1,p2,Np,t1,t2,tol,mass,N) = (args.descr,args.x1,args.x2,args.Nx,args.p1,args.p2,args.Np,args.t1,args.t2,args.tol,args.mass,args.N)

(x0,p0,sigmax,sigmap) = map(array, (args.x0, args.p0, args.sigmax, args.sigmap))

def pr_msg(str):
    print(descr + ": " + str)

def pr_exit(str):
    pr_msg("ERROR: " + str)
    exit()

if Nx & (Nx-1): pr_msg("WARNING: Nx=%d is not a power 2, FFT may be slowed down" % Nx)
if Np & (Np-1): pr_msg("WARNING: Np=%d is not a power 2, FFT may be slowed down" % Np)

assert tol > 0 and x2 > x1 and p2 > p1 and Nx > 0 and Np > 0
npoints = len(x0)
assert p0.shape == (npoints,) and sigmax.shape == (npoints,) and sigmap.shape == (npoints,)

Umod = __import__(args.srcU) # load the physical model (U(x) and dUdx(x) definitions)

try: # auto-select FFT implementation: pyfftw is the fastest and numpy is the slowest
    import pyfftw
except ImportError:
    pr_msg("WARNING: pyfftw not available, trying scipy")
    try:
        from scipy.fftpack import fftshift, ifftshift, fft, ifft
    except ImportError:
        pr_msg("WARNING: scipy.fftpack not available, trying numpy")
        try:
            from numpy.fft import ifftshift, fftshift, fft, ifft
        except ImportError:
            pr_exit("No FFT implementation available, tried: pyfftw, scipy, numpy")
else:
    from pyfftw.interfaces.numpy_fft import fft, fftshift, ifftshift, ifft
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)

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
    """qdf(x,dx) --- quantum differential of function f(x) at a point x on the increment dx"""
    hbar = 1.0 # Planck's constant enters the theory _only_ via quantum differential
    return (f(x+1j*hbar*dx/2.) - f(x-1j*hbar*dx/2.))/(1j*hbar)

#c = 137.03604 # speed of light in a.u.
c = 1.0 # speed of light in 'natural units'

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
    B = fft(Winit, axis=0, threads=4) # (x,p) -> (lambda,p)
    B *= expT
    B = ifft(B, axis=0, threads=4) # (lambda,p) -> (x,p)
    B = fft(B, axis=1, threads=4) # (x,p) -> (x,theta)
    B *= expU
    B = ifft(B, axis=1, threads=4) # (x,theta) -> (x,p)
    B = fft(B, axis=0, threads=4) # (x,p) -> (lambda,p)
    B *= expT
    B = ifft(B, axis=0, threads=4) # (lambda,p) -> (x,p)
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
dt = (t2-t1)/N # the first very rough guess of time step
Winit = zeros((Nx,Np))
for (ax0,ap0,asigmax,asigmap) in zip(x0, p0, sigmax, sigmap):
    Winit += gauss(xx,pp,ax0,ap0,asigmax,asigmap)
Winit /= npoints

W = [fftshift(Winit)]
tv = [t1]
t = t1
Nt = 1
while t <= t2:
    if Nt%100 == 0: pr_msg("%5d steps" % Nt)
    if Nt%20 == 1:
        (Wnext, new_dt, expU, expT) = adjust_step(dt, W[-1])
        W.append(Wnext)
        if new_dt != dt:
            est_steps = (t2-t)//new_dt + 1
            pr_msg("step %d, dt=%.4f -> %.4f, ~%d steps left" %(Nt,dt,new_dt,est_steps))
            dt = new_dt
    else:
        W.append(solve_spectral(W[-1], expU, expT))
    t += dt
    Nt += 1
    tv.append(t)

pr_msg("solved in %.1fs, %d steps" % (time() - t_start, Nt))

W = ifftshift(W, axes=(1,2))
rho = sum(W, axis=2)*dp
phi = sum(W, axis=1)*dx
E = sum(H*W,axis=(1,2))*dx*dp

params = {'Wmin': amin(W), 'Wmax': amax(W), 'rho_min': amin(rho), 'rho_max': amax(rho),
          'Hmin': amin(H), 'Hmax': amax(H), 'Emin': amin(E), 'Emax': amax(E),
          'phi_min': amin(phi), 'phi_max': amax(phi), 'tol': tol, 'Wfilename': Wfilename, 'Nt': Nt,
          'x1': x1, 'x2': x2, 'Nx': Nx, 'p1': p1, 'p2': p2, 'Np': Np, 'descr': descr}

savez(sfilename, t=tv, rho=rho, phi=phi, H=H, U=Uv, T=Tv, E=E, H0=T(p0)+Umod.U(x0), params=params)
fp = memmap(Wfilename, dtype='float64', mode='w+', shape=(Nt, Nx, Np))
fp[:] = W[:]
del fp # causes the flush of memmap
