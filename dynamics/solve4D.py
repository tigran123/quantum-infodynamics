"""
   solve4D.py --- Quantum Infodynamics Solver in Four-Dimensional Phase Space
   Author: Tigran Aivazian <aivazian.tigran@gmail.com>
   License: GPL
"""

from numpy import linspace, mgrid, pi, newaxis, exp, real, savez, amin, amax, sum, abs, memmap, sqrt, sign, zeros, array
import argparse as arg
from time import time

p = arg.ArgumentParser(description="Quantum Infodynamics Tools - 4D Equations Solver")
p.add_argument("-d",  action="store", help="Description text", dest="descr", required=True)
p.add_argument("-x0", action="append", help="Initial packet's x-coordinate (multiple OK)", dest="x0", type=float, required=True, default=[])
p.add_argument("-y0", action="append", help="Initial packet's y-coordinate (multiple OK)", dest="y0", type=float, required=True, default=[])
p.add_argument("-px0", action="append", help="Initial packet's px-coordinate (multiple OK)", dest="px0", type=float, required=True, default=[])
p.add_argument("-py0", action="append", help="Initial packet's py-coordinate (multiple OK)", dest="py0", type=float, required=True, default=[])
p.add_argument("-sigmax", action="append", help="Initial packet's σx (multiple OK)", dest="sigmax", type=float, required=True, default=[])
p.add_argument("-sigmay", action="append", help="Initial packet's σy (multiple OK)", dest="sigmay", type=float, required=True, default=[])
p.add_argument("-sigmapx", action="append", help="Initial packet's σpx (multiple OK)", dest="sigmapx", type=float, required=True, default=[])
p.add_argument("-sigmapy", action="append", help="Initial packet's σpy (multiple OK)", dest="sigmapy", type=float, required=True, default=[])
p.add_argument("-x1", action="store", help="Starting x-coordinate", dest="x1", type=float, required=True)
p.add_argument("-x2", action="store", help="Final x-coordinate", dest="x2", type=float, required=True)
p.add_argument("-y1", action="store", help="Starting y-coordinate", dest="y1", type=float, required=True)
p.add_argument("-y2", action="store", help="Final y-coordinate", dest="y2", type=float, required=True)
p.add_argument("-Nx", action="store", help="Number of points in x direction", dest="Nx", type=int, required=True)
p.add_argument("-Ny", action="store", help="Number of points in y direction", dest="Ny", type=int, required=True)
p.add_argument("-px1", action="store", help="Starting px-momentum", dest="px1", type=float, required=True)
p.add_argument("-px2", action="store", help="Final px-momentum", dest="px2", type=float, required=True)
p.add_argument("-Npx", action="store", help="Number of points in px direction", dest="Npx", type=int, required=True)
p.add_argument("-py1", action="store", help="Starting py-momentum", dest="py1", type=float, required=True)
p.add_argument("-py2", action="store", help="Final py-momentum", dest="py2", type=float, required=True)
p.add_argument("-Npy", action="store", help="Number of points in py direction", dest="Npy", type=int, required=True)
p.add_argument("-t1", action="store", help="Starting time", dest="t1", type=float, required=True)
p.add_argument("-t2", action="store", help="Final time", dest="t2", type=float, required=True)
p.add_argument("-u",  action="store", help="Python source of U(x,y), dUdx(x,y) and dUdy(x,y)", dest="srcU", required=True)
p.add_argument("-s",  action="store", help="Solution file name", dest="sfilename", required=True)
p.add_argument("-c",  action="store_true", help="Use classical (non-quantum) propagator", dest="classical")
p.add_argument("-r",  action="store_true", help="Use relativistic dynamics", dest="relat")
p.add_argument("-m",  action="store", help="Rest mass in a.u. (default=1.0)", type=float, dest="mass", default=1.0)
p.add_argument("-tol", action="store", help="Absolute error tolerance", dest="tol", type=float, required=True)
args = p.parse_args()

sfilename = args.sfilename
Wfilename = sfilename + '_W.npz'

(descr,x1, x2, Nx, y1, y2, Ny,
       px1,px2,Npx,py1,py2,Npy,
       t1,t2,tol,mass) = (args.descr, args.x1, args.x2, args.Nx, args.y1, args.y2, args.Ny,
                                      args.px1,args.px2,args.Npx,args.py1,args.py2,args.Npy,
                                      args.t1,args.t2,args.tol,args.mass)

(x0,y0,px0,py0,sigmax,sigmay,sigmapx,sigmapy) = map(array, (args.x0, args.y0, args.px0, args.py0,
                                                    args.sigmax, args.sigmay, args.sigmapx, args.sigmapy))

def pr_msg(str):
    print(descr + ": " + str)

def pr_exit(str):
    pr_msg("ERROR: " + str)
    exit()

if Nx & (Nx-1): pr_msg("WARNING: Nx=%d is not a power 2, FFT may be slowed down" % Nx)
if Ny & (Ny-1): pr_msg("WARNING: Ny=%d is not a power 2, FFT may be slowed down" % Ny)
if Npx & (Npx-1): pr_msg("WARNING: Npx=%d is not a power 2, FFT may be slowed down" % Npx)
if Npy & (Npy-1): pr_msg("WARNING: Npy=%d is not a power 2, FFT may be slowed down" % Npy)

assert tol > 0 and mass >= 0 and x2 > x1 and y2 > y1 and px2 > px1 and py2 > py1 and Nx > 0 and Ny > 0 and Npx > 0 and Npy > 0
npoints = len(x0)
assert y0.shape == (npoints,) and px0.shape == (npoints,) and py0.shape == (npoints,) and sigmax.shape == (npoints,) and sigmay.shape == (npoints,) and sigmapx.shape == (npoints,) and sigmapy.shape == (npoints,)

Umod = __import__(args.srcU) # load the physical model

try: # auto-select FFT implementation: pyfftw is the fastest and numpy is the slowest
    import pyfftw
except ImportError:
    pr_msg("WARNING: pyfftw not available, trying scipy")
    try:
        from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
    except ImportError:
        pr_msg("WARNING: scipy.fftpack not available, trying numpy")
        try:
            from numpy.fft import ifftshift, fftshift, fft2, ifft2
        except ImportError:
            pr_exit("No FFT implementation available, tried: pyfftw, scipy, numpy")
else:
    from pyfftw.interfaces.numpy_fft import fft2, fftshift, ifftshift, ifft2
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)

# construct the mesh grid for evaluating f0(x,y,px,py), U(x,y), dUdx(x,y), dUdy(x,y), T(px,py), dTdpx(px,py), dTdpy(px,py)
xv,dx = linspace(x1, x2, Nx, endpoint=False, retstep=True)
yv,dy = linspace(y1, y2, Ny, endpoint=False, retstep=True)
pxv,dpx = linspace(px1, px2, Npx, endpoint=False, retstep=True)
pyv,dpy = linspace(py1, py2, Npy, endpoint=False, retstep=True)
xx,yy,ppx,ppy = mgrid[x1:x2-dx:Nx*1j,y1:y2-dy:Ny*1j,px1:px2-dpx:Npx*1j,py1:py2-dpy:Npy*1j]

# ranges in Fourier image spaces (thetax is conjugated to px)
dthetax = 2.*pi/(px2-px1)
thetax_amp = dthetax*Npx/2.
thetaxv = linspace(-thetax_amp, thetax_amp - dthetax, Npx)

dthetay = 2.*pi/(py2-py1)
thetay_amp = dthetay*Npy/2.
thetayv = linspace(-thetay_amp, thetay_amp - dthetay, Npy)

# lamx is conjugated to x
dlamx = 2.*pi/(x2-x1)
lamx_amp = dlamx*Nx/2.
lamxv = linspace(-lamx_amp, lamx_amp - dlamx, Nx)

dlamy = 2.*pi/(y2-y1)
lamy_amp = dlamy*Ny/2.
lamyv = linspace(-lamy_amp, lamy_amp - dlamy, Ny)

# now shift them all to center zero frequency
X = fftshift(xv)[:,newaxis,newaxis,newaxis]
Y = fftshift(yv)[newaxis,:,newaxis,newaxis]
Px = fftshift(pxv)[newaxis,newaxis,:,newaxis]
Py = fftshift(pyv)[newaxis,newaxis,newaxis,:]

LamX = fftshift(lamxv)[:,newaxis,newaxis,newaxis]
LamY = fftshift(lamyv)[newaxis,:,newaxis,newaxis]
ThetaX = fftshift(thetaxv)[newaxis,newaxis,:,newaxis]
ThetaY = fftshift(thetayv)[newaxis,newaxis,newaxis,:]

def gauss(x, y, px, py, x0, y0, px0, py0, sigmax, sigmay, sigmapx, sigmapy):
    Z = 1./(4.*pi**2*sigmax*sigmay*sigmapx*sigmapy)
    return Z*exp(-(x-x0)**2/(2.*sigmax**2) - (y-y0)**2/(2.*sigmay**2) - (px-px0)**2/(2.*sigmapx**2) - (py-py0)**2/(2.*sigmapy**2))

def qd(f, x, dx, y, dy):
    hbar = 1.0 # Planck's constant in a.u.
    #hbar = 1.0545718e-34 # Planck's constant in J*s (SI)
    return (f(x+1j*hbar*dx/2., y+1j*hbar*dy/2.) - f(x-1j*hbar*dx/2., y-1j*hbar*dy/2))/(1j*hbar)

#c = 137.03604 # speed of light in a.u.
c = 1.0 # speed of light in 'natural units'

if args.relat:
    T = lambda px, py: c*sqrt(px**2 + py**2 + mass**2*c**2)
    dTdpx = lambda px, py: c*px/sqrt(px**2 + py**2 + mass**2*c**2)
    dTdpy = lambda px, py: c*py/sqrt(px**2 + py**2 + mass**2*c**2)
else:
    T = lambda px, py: (px**2 + py**2)/(2.*mass)
    dTdpx = lambda px, py: px/mass
    dTdpy = lambda px, py: py/mass

if args.classical:
    dU = Umod.dUdx(X,Y)*1j*ThetaX + Umod.dUdy(X,Y)*1j*ThetaY
    dT = -1j*(dTdpx(Px,Py)*LamX + dTdpy(Px,Py)*LamY)/2.
else:
    dU = qd(Umod.U, X,  1j*ThetaX, Y,  1j*ThetaY)
    dT = qd(T,     Px, -1j*LamX,  Py, -1j*LamY)/2.

H = T(ppx,ppy) + Umod.U(xx,yy)

def solve_spectral(Winit, expU, expT):
    B = fft2(Winit, axes=(0,1)) # (x,y,px,py) -> (λx,λy,px,py)
    B *= expT
    B = ifft2(B, axes=(0,1)) # (λx,λy,px,py) -> (x,y,px,py)
    B = fft2(B, axes=(2,3)) # (x,y,px,py) -> (x,y,θx,θy)
    B *= expU
    B = ifft2(B, axes=(2,3)) # (x,y,θx,θy) -> (x,y,px,py)
    B = fft2(B, axes=(0,1)) # (x,y,px,py) -> (λx,λy,px,py)
    B *= expT
    B = ifft2(B, axes=(0,1)) # (λx,λy,px,py) -> (x,y,px,py)
    return real(B) # to avoid python warning

def adjust_step(cur_dt, Winit):
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
        if amax(abs(W2 - W1)) <= tol or tries > 15: break
        dt *= 0.7
    return (W1, dt, expU, expT)

t_start = time()
dt = (t2-t1)/20. # the first very rough guess of time step
Winit = zeros((Nx,Ny,Npx,Npy))
for (ax0,ay0,apx0,apy0,asigmax,asigmay,asigmapx,asigmapy) in zip(x0, y0, px0, py0, sigmax, sigmay, sigmapx, sigmapy):
    Winit += gauss(xx, yy, ppx, ppy, ax0, ay0, apx0, apy0, asigmax, asigmay, asigmapx, asigmapy)
Winit /= npoints

dmux = dx*dy
dmup = dpx*dpy
dmu = dmux*dmup
W = fftshift(Winit)
rho = [sum(Winit, axis=(2,3))*dmup]
phi = [sum(Winit, axis=(0,1))*dmux]

Energy = sum(H*Winit)*dmu
if args.relat: # so we can compare it with the non-relativistic kinetic energy
    Energy -= mass*c**2
E = [Energy]

tv = [t1]
t = t1
Nt = 1
while t <= t2:
    if Nt%300 == 299: pr_msg("step %d"%Nt)
    if Nt%20 == 1:
        (W, new_dt, expU, expT) = adjust_step(dt, W)
        if new_dt != dt:
            pr_msg("step %d, dt %.4f -> %.4f, ~%d steps left" % (Nt, dt, new_dt, (t2-t)//new_dt + 1))
            dt = new_dt
    else:
        W = solve_spectral(W, expU, expT)
    Wp = ifftshift(W)
    rho.append(sum(Wp, axis=(2,3))*dmup)
    phi.append(sum(Wp, axis=(0,1))*dmux)
    Energy = sum(H*Wp)*dmu
    if args.relat: # so we can compare it with the non-relativistic kinetic energy
        Energy -= mass*c**2
    E.append(Energy)
    t += dt
    Nt += 1
    tv.append(t)

pr_msg("solved in %.1fs, %d steps" % (time() - t_start, Nt))

params = {'rho_min': amin(rho), 'rho_max': amax(rho), 'Hmin': amin(H), 'Hmax': amax(H), 'Emin': amin(E), 'Emax': amax(E),
          'phi_min': amin(phi), 'phi_max': amax(phi), 'tol': tol, 'Nt': Nt,
          'x1': x1, 'x2': x2, 'Nx': Nx, 'y1': y1, 'y2': y2, 'Ny': Ny, 
          'px1': px1, 'px2': px2, 'Npx': Npx, 'py1': py1, 'py2': py2, 'Npy': Npy,
          'descr': descr}

t_start = time()
savez(sfilename, t=tv, rho=rho, phi=phi, H=H, E=E, H0=T(px0,py0)+Umod.U(x0,y0), params=params)
pr_msg("solution saved in %.1fs" % (time() - t_start))
