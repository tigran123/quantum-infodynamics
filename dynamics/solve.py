"""
   solve.py --- Quantum Infodynamics Tools (Spectral Split Propagator of Second Order with Adaptive Timestep Control)
   Author: Tigran Aivazian <aivazian.tigran@gmail.com>
   License: GPL
"""

from numpy import linspace, mgrid, pi, newaxis, exp, real, load, savez, amin, amax, sum, abs, memmap, sqrt, sign
from argparse import ArgumentParser as argp
from timeit import default_timer as timer
import sys

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

p = argp(description="Quantum Infodynamics Tools - Equations Solver")
p.add_argument("-d",  action="store", help="Description text", dest="descr", required=True)
p.add_argument("-t1", action="store", help="Starting time", dest="t1", type=float, required=True)
p.add_argument("-t2", action="store", help="Final time", dest="t2", type=float, required=True)
p.add_argument("-u",  action="store", help="Python source of U(x) and U'(x)", dest="srcU", required=True)
p.add_argument("-o",  action="store", help="Solution file name", dest="ofilename", required=True)
p.add_argument("-i",  action="store", help="Cauchy data file name", dest="ifilename", required=True)
p.add_argument("-c",  action="store_true", help="Use classical (non-quantum) propagator", dest="classical")
p.add_argument("-r",  action="store_true", help="Use relativistic dynamics", dest="relat")
p.add_argument("-m",  action="store", help="Rest mass in a.u. (default=1.0)", type=complex, dest="mass", default=1.0)
p.add_argument("-N",  action="store", help="Initial number of time steps (default=100)", dest="N", type=int, default=100)
p.add_argument("-adaptive", help="Use adaptive timestep control) (default=Yes)", dest="adaptive", const=True, type=str2bool, nargs='?', default=True)
p.add_argument("-mm", help="Use memory-mapped array for W(x,p,t) (default=Yes)", dest="mm", const=True, type=str2bool, nargs='?', default=True)
p.add_argument("-mmsize", help="Initial size (in GB) of the memory-mapped array for W(x,p,t) (default=32)", dest="mmsize", type=int, default=32)
p.add_argument("-tol", action="store", help="Relative error tolerance (0 < tol < 1)", dest="tol", type=float)
args = p.parse_args()

oWfilename = args.ofilename + '_W.npy'

(descr,t1,t2,tol,mass,N,mm,adaptive) = (args.descr,args.t1,args.t2,args.tol,args.mass,args.N,args.mm,args.adaptive)

def pr_msg(str):
    print(descr + ": " + str)

def pr_exit(str):
    pr_msg("ERROR: " + str)
    sys.exit()

with load(args.ifilename + '.npz', allow_pickle=True) as data:
    params = data['params'][()]
    iWfilename = params['Wfilename']
    x0 = params['x0']; x1 = params['x1']; x2 = params['x2']; Nx = params['Nx']
    p0 = params['p0']; p1 = params['p1']; p2 = params['p2']; Np = params['Np']
    Nt = params['Nt']

if Nx & (Nx-1): pr_msg("WARNING: Nx=%d is not a power 2, FFT may be slowed down" % Nx)
if Np & (Np-1): pr_msg("WARNING: Np=%d is not a power 2, FFT may be slowed down" % Np)

assert not adaptive or 0 < tol < 1, "Tolerance value %.2f outside (0,1) range" % tol

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

def qd(f, x, dx):
    """qdf(x,dx) --- quantum differential of function f(x) at a point x on the increment dx"""
    #hbar = 1.0545718e-34 # Planck's constant enters the theory _only_ via quantum differential
    hbar = 1.0 # in 'natural units'
    return (f(x+1j*hbar*dx/2.) - f(x-1j*hbar*dx/2.))/(1j*hbar)

#c = 137.03604 # speed of light in a.u.
c =  1.0 # speed of light in 'natural units'
#c = 299792458.0 # speed of light in SI

def dTdp_rel(p):
    if mass == 0.0:
        return c*sign(p)
    else:
        return c*p/sqrt(p**2 + mass**2*c**2)

def solve_spectral_mt(Winit, expU, expT):
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

def solve_spectral(Winit, expU, expT):
    B = fft(Winit, axis=0) # (x,p) -> (lambda,p)
    B *= expT
    B = ifft(B, axis=0) # (lambda,p) -> (x,p)
    B = fft(B, axis=1) # (x,p) -> (x,theta)
    B *= expU
    B = ifft(B, axis=1) # (x,theta) -> (x,p)
    B = fft(B, axis=0) # (x,p) -> (lambda,p)
    B *= expT
    B = ifft(B, axis=0) # (lambda,p) -> (x,p)
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
        if sum(abs(W1-W2))/sum(abs(W1)) < tol: break
        if tries > maxtries:
            pr_msg("WARNING: adjust_step: giving up after %d attempts" % maxtries)
            break
        dt *= 0.7
    return (W1, dt, expU, expT)

(T,dTdp) = (lambda p: c*sqrt(p**2 + mass**2*c**2),dTdp_rel) if args.relat else (lambda p: p**2/(2.*mass),lambda p: p/mass)
(dU,dT) = (Umod.dUdx(X)*1j*Theta,-dTdp(P)*1j*Lam/2.) if args.classical else (qd(Umod.U,X,1j*Theta),qd(T,P,-1j*Lam)/2.)

dt = (t2-t1)/N # the first very rough guess of time step
expU,expT = exp(dt*dU),exp(dt*dT)
Winit = memmap(iWfilename, dtype='float64', mode='r', shape=(Nt, Nx, Np))

if mm:
    Ntmax = (1024)**3*args.mmsize//(Winit.itemsize*Nx*Np)
    assert Ntmax > N, "Pre-allocation out of mm-mapped array size, increase -mmsize"
    W = memmap(oWfilename, dtype='float64', mode='w+', shape=(Ntmax, Nx, Np))
    W[0] = fftshift(Winit[Nt-1])
else:
    W = [fftshift(Winit[Nt-1])]

tv = [t1]
t = t1
Nt = 1
t_calc = 0.0
t_start = timer()
t2 -= dt
while (dt > 0 and t < t2) or (dt < 0 and t > t2):
    if Nt%100 == 1:
        Ntleft = (t2-t)//dt
        assert not mm or Ntmax > Nt + Ntleft, "Calculation out of mm-mapped array size, increase -mmsize"
        if Nt > 1:
            pr_msg("%d steps (%.2f steps/second), ~%d steps left" % (Nt, Nt/(timer()-t_start), Ntleft))
        else:
            pr_msg("%d step, ~%d steps left" % (Nt, Ntleft))
    if adaptive and Nt%20 == 1:
        (Wnext, new_dt, expU, expT) = adjust_step(dt, W[Nt-1])
        if mm:
            W[Nt] = Wnext
        else:
            W.append(Wnext)
        if new_dt != dt:
            Ntleft = (t2-t)//new_dt
            assert not mm or Ntmax > Nt + Ntleft, "Adaptive calculation out of mm-mapped array size, increase -mmsize"
            pr_msg("step %d (%.2f steps/second), dt=%.4f -> %.4f, ~%d steps left" % (Nt, Nt/(timer()-t_start), dt, new_dt, Ntleft))
            dt = new_dt
    else:
        if mm:
            W[Nt] = solve_spectral(W[Nt-1], expU, expT)
        else:
            W.append(solve_spectral(W[Nt-1], expU, expT))
    t += dt
    Nt += 1
    tv.append(t)

t_end = timer()
pr_msg("solved in %.2fs, %d steps (%.2f steps/second)" % (t_end - t_start, Nt, Nt/(t_end - t_start)))

if mm:
    #t_start = timer()
    nbytes = W.itemsize*Nt*Nx*Np
    W.base.resize(nbytes)
    W.flush()
    del W
    W = memmap(oWfilename, dtype='float64', mode='r+', shape=(Nt, Nx, Np))
    #pr_msg("Wigner function resized to shape (%d,%d,%d), %d bytes in %.2fs" % (Nt, Nx, Np, nbytes, timer() - t_start))

#t_start = timer()
if mm:
    W[:] = ifftshift(W, axes=(1,2))
else:
    W = ifftshift(W, axes=(1,2))
#pr_msg("Wigner function shifted in %.2fs" % (timer() - t_start))

#t_start = timer()
rho = sum(W, axis=2)*dp
phi = sum(W, axis=1)*dx
H = real(T(pp)+Umod.U(xx))
H0 = real(T(p0) + Umod.U(x0))
E = sum(H*W,axis=(1,2))*dx*dp

X = sum(xx * W,axis=(1,2))*dx*dp
X2 = sum(xx**2 * W,axis=(1,2))*dx*dp
deltaX = sqrt(abs(X2-X*X))

P = sum(pp * W,axis=(1,2))*dx*dp
P2 = sum(pp**2 * W,axis=(1,2))*dx*dp
deltaP = sqrt(abs(P2-P*P))

params = {'Wmin': amin(W), 'Wmax': amax(W), 'rho_min': amin(rho), 'rho_max': amax(rho),
          'Hmin': amin(H), 'Hmax': amax(H), 'Emin': amin(E), 'Emax': amax(E),
          'phi_min': amin(phi), 'phi_max': amax(phi), 'Wfilename': oWfilename, 'Nt': Nt,
          'x0': x0, 'x1': x1, 'x2': x2, 'Nx': Nx, 'p0': p0, 'p1': p1, 'p2': p2, 'Np': Np, 'descr': descr}

#pr_msg("parameters calculated in %.2fs" % (timer() - t_start))

#t_start = timer()
savez(args.ofilename, t=tv, rho=rho, phi=phi, H=H, E=E, deltaX=deltaX, deltaP=deltaP, H0=H0, params=params)

if not mm:
    fp = memmap(oWfilename, dtype='float64', mode='w+', shape=(Nt, Nx, Np))
    fp[:] = W[:]
    del fp # causes the flush of memmap

#pr_msg("solution saved in %.2fs" % (timer() - t_start))
