"""
   solve6D.py --- Quantum Infodynamics Solver in Six-Dimensional Phase Space
   Author: Tigran Aivazian <aivazian.tigran@gmail.com>
   License: GPL
"""

from numpy import linspace, mgrid, pi, newaxis, exp, real, savez, amin, amax, sum, abs, memmap, sqrt, sign, zeros, array
import argparse as arg
from time import time

p = arg.ArgumentParser(description="Quantum Infodynamics Tools - Equations Solver")
p.add_argument("-d",  action="store", help="Description text", dest="descr", required=True)
p.add_argument("-x0", action="append", help="Initial packet's x-coordinate (multiple OK)", dest="x0", type=float, required=True, default=[])
p.add_argument("-y0", action="append", help="Initial packet's y-coordinate (multiple OK)", dest="y0", type=float, required=True, default=[])
p.add_argument("-z0", action="append", help="Initial packet's z-coordinate (multiple OK)", dest="z0", type=float, required=True, default=[])
p.add_argument("-px0", action="append", help="Initial packet's px-coordinate (multiple OK)", dest="px0", type=float, required=True, default=[])
p.add_argument("-py0", action="append", help="Initial packet's py-coordinate (multiple OK)", dest="py0", type=float, required=True, default=[])
p.add_argument("-pz0", action="append", help="Initial packet's pz-coordinate (multiple OK)", dest="pz0", type=float, required=True, default=[])
p.add_argument("-sigmax", action="append", help="Initial packet's σx (multiple OK)", dest="sigmax", type=float, required=True, default=[])
p.add_argument("-sigmay", action="append", help="Initial packet's σy (multiple OK)", dest="sigmay", type=float, required=True, default=[])
p.add_argument("-sigmaz", action="append", help="Initial packet's σz (multiple OK)", dest="sigmaz", type=float, required=True, default=[])
p.add_argument("-sigmapx", action="append", help="Initial packet's σpx (multiple OK)", dest="sigmapx", type=float, required=True, default=[])
p.add_argument("-sigmapy", action="append", help="Initial packet's σpy (multiple OK)", dest="sigmapy", type=float, required=True, default=[])
p.add_argument("-sigmapz", action="append", help="Initial packet's σpz (multiple OK)", dest="sigmapz", type=float, required=True, default=[])
p.add_argument("-x1", action="store", help="Starting x-coordinate", dest="x1", type=float, required=True)
p.add_argument("-x2", action="store", help="Final x-coordinate", dest="x2", type=float, required=True)
p.add_argument("-y1", action="store", help="Starting y-coordinate", dest="y1", type=float, required=True)
p.add_argument("-y2", action="store", help="Final y-coordinate", dest="y2", type=float, required=True)
p.add_argument("-z1", action="store", help="Starting z-coordinate", dest="z1", type=float, required=True)
p.add_argument("-z2", action="store", help="Final z-coordinate", dest="z2", type=float, required=True)
p.add_argument("-Nx", action="store", help="Number of points in x direction", dest="Nx", type=int, required=True)
p.add_argument("-Ny", action="store", help="Number of points in y direction", dest="Ny", type=int, required=True)
p.add_argument("-Nz", action="store", help="Number of points in z direction", dest="Nz", type=int, required=True)
p.add_argument("-px1", action="store", help="Starting px-momentum", dest="px1", type=float, required=True)
p.add_argument("-px2", action="store", help="Final px-momentum", dest="px2", type=float, required=True)
p.add_argument("-Npx", action="store", help="Number of points in px direction", dest="Npx", type=int, required=True)
p.add_argument("-py1", action="store", help="Starting py-momentum", dest="py1", type=float, required=True)
p.add_argument("-py2", action="store", help="Final py-momentum", dest="py2", type=float, required=True)
p.add_argument("-Npy", action="store", help="Number of points in py direction", dest="Npy", type=int, required=True)
p.add_argument("-pz1", action="store", help="Starting pz-momentum", dest="pz1", type=float, required=True)
p.add_argument("-pz2", action="store", help="Final pz-momentum", dest="pz2", type=float, required=True)
p.add_argument("-Npz", action="store", help="Number of points in pz direction", dest="Npz", type=int, required=True)
p.add_argument("-t1", action="store", help="Starting time", dest="t1", type=float, required=True)
p.add_argument("-t2", action="store", help="Final time", dest="t2", type=float, required=True)
p.add_argument("-u",  action="store", help="Python source of U(x,y,z), dUdx(x,y,z), dUdy(x,y,z), dUdz(x,y,z)", dest="srcU", required=True)
p.add_argument("-s",  action="store", help="Solution file name", dest="sfilename", required=True)
p.add_argument("-c",  action="store_true", help="Use classical (non-quantum) propagator", dest="classical")
p.add_argument("-r",  action="store_true", help="Use relativistic dynamics", dest="relat")
p.add_argument("-m",  action="store", help="Rest mass in a.u. (default=1.0)", type=float, dest="mass", default=1.0)
p.add_argument("-tol", action="store", help="Absolute error tolerance", dest="tol", type=float, required=True)
args = p.parse_args()

sfilename = args.sfilename
Wfilename = sfilename + '_W.npz'

(descr,
    x1, x2, Nx,
    y1, y2, Ny,
    z1, z2, Nz,
    px1,px2,Npx,
    py1,py2,Npy,
    pz1,pz2,Npz,
    t1, t2,
    tol,mass) = (args.descr,
                            args.x1, args.x2, args.Nx,
                            args.y1, args.y2, args.Ny,
                            args.z1, args.z2, args.Nz,
                            args.px1,args.px2,args.Npx,
                            args.py1,args.py2,args.Npy,
                            args.pz1,args.pz2,args.Npz,
                            args.t1, args.t2,
                            args.tol,args.mass)

(x0,y0,z0,px0,py0,pz0,sigmax,sigmay,sigmaz,sigmapx,sigmapy,sigmapz) = map(array, (args.x0, args.y0, args.z0,
                                                                                  args.px0,args.py0,args.pz0,
                                                    args.sigmax, args.sigmay, args.sigmaz,
                                                    args.sigmapx,args.sigmapy,args.sigmapz))

def pr_msg(str):
    print(descr + ": " + str)

def pr_exit(str):
    pr_msg("ERROR: " + str)
    exit()

if Nx & (Nx-1): pr_msg("WARNING: Nx=%d is not a power 2, FFT may be slowed down" % Nx)
if Ny & (Ny-1): pr_msg("WARNING: Ny=%d is not a power 2, FFT may be slowed down" % Ny)
if Nz & (Nz-1): pr_msg("WARNING: Nz=%d is not a power 2, FFT may be slowed down" % Nz)
if Npx & (Npx-1): pr_msg("WARNING: Npx=%d is not a power 2, FFT may be slowed down" % Npx)
if Npy & (Npy-1): pr_msg("WARNING: Npy=%d is not a power 2, FFT may be slowed down" % Npy)
if Npz & (Npz-1): pr_msg("WARNING: Npz=%d is not a power 2, FFT may be slowed down" % Npz)

assert tol > 0 and mass >= 0 and x2 > x1 and y2 > y1 and z2 > z1 and px2 > px1 and py2 > py1 and pz2 > pz1 and Nx > 0 and Ny > 0 and Nz > 0 and Npx > 0 and Npy > 0 and Npz > 0
npoints = len(x0)
assert y0.shape == (npoints,) and z0.shape == (npoints,) and px0.shape == (npoints,) and py0.shape == (npoints,) and pz0.shape == (npoints,) and sigmax.shape == (npoints,) and sigmay.shape == (npoints,) and sigmaz.shape == (npoints,) and sigmapx.shape == (npoints,) and sigmapy.shape == (npoints,) and sigmapz.shape == (npoints,)

Umod = __import__(args.srcU) # load the physical model

try: # auto-select FFT implementation: pyfftw is the fastest and numpy is the slowest
    import pyfftw
except ImportError:
    pr_msg("WARNING: pyfftw not available, trying scipy")
    try:
        from scipy.fftpack import fftshift, ifftshift, fftn, ifftn
    except ImportError:
        pr_msg("WARNING: scipy.fftpack not available, trying numpy")
        try:
            from numpy.fft import ifftshift, fftshift, fftn, ifftn
        except ImportError:
            pr_exit("No FFT implementation available, tried: pyfftw, scipy, numpy")
else:
    from pyfftw.interfaces.numpy_fft import fftn, fftshift, ifftshift, ifftn
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)

# construct the mesh grid for evaluating all fields on it
xv,dx = linspace(x1, x2, Nx, endpoint=False, retstep=True)
yv,dy = linspace(y1, y2, Ny, endpoint=False, retstep=True)
zv,dz = linspace(z1, z2, Nz, endpoint=False, retstep=True)

pxv,dpx = linspace(px1, px2, Npx, endpoint=False, retstep=True)
pyv,dpy = linspace(py1, py2, Npy, endpoint=False, retstep=True)
pzv,dpz = linspace(pz1, pz2, Npz, endpoint=False, retstep=True)

xgrid,ygrid,zgrid,pxgrid,pygrid,pzgrid = mgrid[x1:x2-dx:Nx*1j,     y1:y2-dy:Ny*1j,     z1:z2-dz:Nz*1j,
                                               px1:px2-dpx:Npx*1j, py1:py2-dpy:Npy*1j, pz1:pz2-dpz:Npz*1j]

# ranges in Fourier image spaces (theta is conjugated to p)
dthetax = 2.*pi/(px2-px1)
thetax_amp = dthetax*Npx/2.
thetaxv = linspace(-thetax_amp, thetax_amp - dthetax, Npx)

dthetay = 2.*pi/(py2-py1)
thetay_amp = dthetay*Npy/2.
thetayv = linspace(-thetay_amp, thetay_amp - dthetay, Npy)

dthetaz = 2.*pi/(pz2-pz1)
thetaz_amp = dthetay*Npz/2.
thetayv = linspace(-thetaz_amp, thetaz_amp - dthetaz, Npz)

# lam is conjugated to x
dlamx = 2.*pi/(x2-x1)
lamx_amp = dlamx*Nx/2.
lamxv = linspace(-lamx_amp, lamx_amp - dlamx, Nx)

dlamy = 2.*pi/(y2-y1)
lamy_amp = dlamy*Ny/2.
lamyv = linspace(-lamy_amp, lamy_amp - dlamy, Ny)

dlamz = 2.*pi/(z2-z1)
lamz_amp = dlamy*Nz/2.
lamzv = linspace(-lamz_amp, lamz_amp - dlamz, Nz)

# now shift them all to center zero frequency
X = fftshift(xv)[:,newaxis,newaxis,newaxis,newaxis,newaxis]
Y = fftshift(yv)[newaxis,:,newaxis,newaxis,newaxis,newaxis]
Z = fftshift(zv)[newaxis,newaxis,:,newaxis,newaxis,newaxis]
Px = fftshift(pxv)[newaxis,newaxis,newaxis,:,newaxis,newaxis]
Py = fftshift(pyv)[newaxis,newaxis,newaxis,newaxis,:,newaxis]
Pz = fftshift(pzv)[newaxis,newaxis,newaxis,newaxis,newaxis,:]

LamX = fftshift(lamxv)[:,newaxis,newaxis,newaxis,newaxis,newaxis]
LamY = fftshift(lamyv)[newaxis,:,newaxis,newaxis,newaxis,newaxis]
LamZ = fftshift(lamzv)[newaxis,newaxis,:,newaxis,newaxis,newaxis]
ThetaX = fftshift(thetaxv)[newaxis,newaxis,newaxis,:,newaxis,newaxis]
ThetaY = fftshift(thetayv)[newaxis,newaxis,newaxis,newaxis,:,newaxis]
ThetaZ = fftshift(thetayv)[newaxis,newaxis,newaxis,newaxis,newaxis,:]

def gauss(x, y, z, px, py, pz, x0, y0, z0, px0, py0, pz0, sigmax, sigmay, sigmaz, sigmapx, sigmapy, sigmapz):
    Z = 1./(8.*pi**3*sigmax*sigmay*sigmaz*sigmapx*sigmapy*sigmapz)
    return Z*exp(-(x-x0)**2/(2.*sigmax**2) - (y-y0)**2/(2.*sigmay**2) - (z-z0)**2/(2.*sigmaz**2) - (px-px0)**2/(2.*sigmapx**2) - (py-py0)**2/(2.*sigmapy**2) - (pz-pz0)**2/(2.*sigmapz**2))

# quantum differential
def qd(f, x, dx, y, dy, z, dz):
    hbar = 1.0 # Planck's constant in a.u.
    #hbar = 1.0545718e-34 # Planck's constant in J*s (SI)
    return (f(x+1j*hbar*dx/2., y+1j*hbar*dy/2., z+1j*hbar*dz/2) - f(x-1j*hbar*dx/2., y-1j*hbar*dy/2., z-1j*hbar*dz/2.))/(1j*hbar)

#c = 137.03604 # speed of light in a.u.
c = 1.0 # speed of light in 'natural units'

def dTdpx_rel(px, py, pz):
    return c*px/sqrt(px**2 + py**2 + pz**2 + mass**2*c**2)

def dTdpy_rel(px, py, pz):
    return c*py/sqrt(px**2 + py**2 + pz**2 + mass**2*c**2)

def dTdpz_rel(px, py, pz):
    return c*pz/sqrt(px**2 + py**2 + pz**2 + mass**2*c**2)

if args.relat:
    T = lambda px, py, pz: c*sqrt(px**2 + py**2 + pz**2 + mass**2*c**2)
    dTdpx = dTdpx_rel
    dTdpy = dTdpy_rel
    dTdpz = dTdpz_rel
else:
    T = lambda px, py, pz: (px**2 + py**2 + pz**2)/(2.*mass)
    dTdpx = lambda px, py, pz: px/mass
    dTdpy = lambda px, py, pz: py/mass
    dTdpz = lambda px, py, pz: pz/mass

if args.classical:
    dU = Umod.dUdx(X,Y,Z)*1j*ThetaX + Umod.dUdy(X,Y,Z)*1j*ThetaY + Umod.dUdz(X,Y,Z)*1j*ThetaZ
    dT = -1j*(dTdpx(Px,Py,Pz)*LamX + dTdpy(Px,Py,Pz)*LamY + dTdpz(Px,Py,Pz)*LamZ)/2.
else:
    dU = qd(Umod.U, X, 1j*ThetaX, Y, 1j*ThetaY, Z, 1j*ThetaZ)
    dT = qd(T, Px, -1j*LamX, Py, -1j*LamY, Pz, -1j*LamZ)/2.

H = T(pxgrid,pygrid,pzgrid)+Umod.U(xgrid,ygrid,zgrid)

def solve_spectral(Winit, expU, expT):
    B = fftn(Winit, axes=(0,1,2)) # (x,y,z,px,py,pz) -> (λx,λy,λz,px,py,pz)
    B *= expT
    B = ifftn(B, axes=(0,1,2)) # (λx,λy,px,py) -> (x,y,px,py)
    B = fftn(B, axes=(3,4,5)) # (x,y,px,py) -> (x,y,θx,θy)
    B *= expU
    B = ifftn(B, axes=(3,4,5)) # (x,y,z,θx,θy,θz) -> (x,y,z,px,py,pz)
    B = fftn(B, axes=(0,1,2)) # (x,y,z,px,py,pz) -> (λx,λy,λz,px,py,pz)
    B *= expT
    B = ifftn(B, axes=(0,1,2)) # (λx,λy,λz,px,py,pz) -> (x,y,z,px,py,pz)
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
Winit = zeros((Nx,Ny,Nz,Npx,Npy,Npz))
for (ax0,ay0,az0,apx0,apy0,apz0,asigmax,asigmay,asigmaz,asigmapx,asigmapy,asigmapz) in zip(x0, y0, z0, px0, py0, pz0, sigmax, sigmay, sigmaz, sigmapx, sigmapy, sigmapz):
    Winit += gauss(xgrid, ygrid, zgrid, pxgrid, pygrid, pzgrid, ax0, ay0, az0, apx0, apy0, apz0, asigmax, asigmay, asigmaz, asigmapx, asigmapy, asigmapz)
Winit /= npoints

rho = []
phi = []
W = fftshift(Winit)
tv = [t1]
t = t1
Nt = 1
while t <= t2:
    if Nt%300 == 299: pr_msg("step %d"%Nt)
    if Nt%20 == 1:
        (W, new_dt, expU, expT) = adjust_step(dt, W)
        if new_dt != dt:
            est_steps = (t2-t)//new_dt + 1
            pr_msg("step %d, dt %.4f -> %.4f, ~%d steps left" %(Nt,dt,new_dt,est_steps))
            dt = new_dt
    else:
        W = solve_spectral(W, expU, expT)
    Ws = ifftshift(W)
    rho.append(sum(Ws, axis=(3,4,5))*dpx*dpy*dpz)
    phi.append(sum(Ws, axis=(0,1,2))*dx*dy*dz)
    t += dt
    Nt += 1
    tv.append(t)

pr_msg("solved in %.1fs, %d steps" % (time() - t_start, Nt))

params = {'Wmin': amin(W), 'Wmax': amax(W), 'rho_min': amin(rho), 'rho_max': amax(rho),
          'Hmin': amin(H), 'Hmax': amax(H), 
          'phi_min': amin(phi), 'phi_max': amax(phi), 'tol': tol, 'Wfilename': Wfilename, 'Nt': Nt,
          'x1': x1, 'x2': x2, 'Nx': Nx, 'y1': y1, 'y2': y2, 'Ny': Ny, 'z1': z1, 'z2': z2, 'Nz': Nz,
          'px1': px1, 'px2': px2, 'Npx': Npx, 'py1': py1, 'py2': py2, 'Npy': Npy, 'pz1': pz1, 'pz2': pz2, 'Npz': Npz,
          'descr': descr}

t_start = time()
savez(sfilename, t=tv, rho=rho, phi=phi, H=H, H0=T(px0,py0,pz0)+Umod.U(x0,y0,z0), params=params)
#fp = memmap(Wfilename, dtype='float64', mode='w+', shape=(Nt, Nx, Ny, Nz, Npx, Npy, Npz))
#fp[:] = W[:]
#del fp # causes the flush of memmap
pr_msg("solution saved in %.1fs" % (time() - t_start))
