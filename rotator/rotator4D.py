from numpy import linspace, mgrid, pi, sqrt, exp, newaxis, sum, abs, amin, amax, real, cos, sin
import matplotlib.pyplot as plt

import pyfftw
from pyfftw.interfaces.numpy_fft import fft2, fftshift, ifftshift, ifft2
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(10)

(Nx,Ny,Npx,Npy) = (64, 64, 128, 128)

(x1,x2,y1,y2) = (-10.0, 10.0, -10.0, 10.0)
(px1, px2, py1, py2) = (-10.0, 10.0, -10.0, 10.0)
(t1,t2) = (0.0, 10.0)

(x0,y0) = (0.0, 0.0)
(sigma_x, sigma_y, sigma_px, sigma_py) = (1.0, 3.0, 1.0, 1.0)
Z = (1./(4.*pi**2*sqrt(sigma_x*sigma_y*sigma_px*sigma_py)))

mass = 1.0 # mass

# construct the mesh grid for evaluating f0(x,y,px,py), U(x,y), dUdx(x,y), dUdy(x,y), T(px,py), dTdpx(px,py), dTdpy(px,py)
xv,dx = linspace(x1, x2, Nx, endpoint=False, retstep=True)
yv,dy = linspace(y1, y2, Ny, endpoint=False, retstep=True)
pxv,dpx = linspace(px1, px2, Npx, endpoint=False, retstep=True)
pyv,dpy = linspace(py1, py2, Npy, endpoint=False, retstep=True)
xgrid,ygrid,pxgrid,pygrid = mgrid[x1:x2-dx:Nx*1j,y1:y2-dy:Ny*1j,px1:px2-dpx:Npx*1j,py1:py2-dpy:Npy*1j]
xx,yy = mgrid[x1:x2-dx:Nx*1j,y1:y2-dy:Ny*1j]

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

ThetaX = fftshift(thetaxv)[newaxis,newaxis,:,newaxis]
ThetaY = fftshift(thetayv)[newaxis,newaxis,newaxis,:]
LamX = fftshift(lamxv)[:,newaxis,newaxis,newaxis]
LamY = fftshift(lamyv)[newaxis,:,newaxis,newaxis]

Z = 1./(4.*pi**2*sigma_x*sigma_y*sigma_px*sigma_py)
def f0(x, y, px, py):
    return Z*exp(-(x-x0)**2/(2.*sigma_x**2) - (y-y0)**2/(2.*sigma_y**2) - (px+mass*y)**2/(2.*sigma_px**2) - (py-mass*x)**2/(2.*sigma_py**2))

def qd(f, x, dx, y, dy):
    hbar = 1.0 # Planck's constant in a.u.
    #hbar = 1.0545718e-34 # Planck's constant in J*s (SI)
    return (f(x+1j*hbar*dx/2., y+1j*hbar*dy/2.) - f(x-1j*hbar*dx/2., y-1j*hbar*dy/2))/(1j*hbar)

#c = 137.03604 # speed of light in a.u.
c = 1.0 # speed of light in 'natural units'

def dTdpx_rel(px, py):
    if mass == 0.0:
        return c*abs(px)/sqrt(px**2 + py**2)
    else:
        return c*px/sqrt(px**2 + py**2 + mass**2*c**2)

def dTdpy_rel(px, py):
    if mass == 0.0:
        return c*abs(py)/sqrt(px**2 + py**2)
    else:
        return c*py/sqrt(px**2 + py**2 + mass**2*c**2)

# relativistic
#T = lambda px, py: c*sqrt(px**2 + py**2 + mass**2*c**2)
#dTdpx = dTdpx_rel
#dTdpy = dTdpy_rel

# non-relativistic
T = lambda px, py: (px**2 + py**2)/(2.*mass)
dTdpx = lambda px, py: px/mass
dTdpy = lambda px, py: py/mass

Umod = __import__('U_free_4D') # load the physical model

# classical:
#dU = Umod.dUdx(X,Y)*1j*ThetaX + Umod.dUdy(X,Y)*1j*ThetaY
#dT = -1j*(dTdpx(Px,Py)*LamX + dTdpy(Px,Py)*LamY)/2.
# quantum
dU = qd(Umod.U, X, 1j*ThetaX, Y, 1j*ThetaY)
dT = qd(T, Px, -1j*LamX, Py, -1j*LamY)/2.

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

def W_analytic(x, y, px, py, t):
    return f0(x - px*t/mass, y - py*t/mass, px, py)

"""
def draw_single(W, k):
    rho = sum(W, axis=(2,3))*dpx*dpy
    phi = sum(W, axis=(0,1))*dx*dy
    Vx = sum(pxgrid*W, axis=(2,3))*dpx*dpy
    Vy = sum(pygrid*W, axis=(2,3))*dpx*dpy

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(19.2,10.8), dpi=100)

    ax1.imshow(rho.T, origin='lower', interpolation='none', extent=[x1,x2-dx,y1,y2-dy],vmin=amin(rho), vmax=amax(rho))
    ax1.quiver(xx, yy, Vx, Vy, color='y')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim([x1,x2-dx])
    ax1.set_ylim([y1,y2-dy])
    ax1.set_title("Spatial density, frame=%d" %k)

    ax2.imshow(phi.T, origin='lower', interpolation='none', extent=[px1,px2-dpx,py1,py2-dpy],vmin=amin(phi), vmax=amax(phi))
    ax2.set_xlabel('Px')
    ax2.set_ylabel('Py')
    ax2.set_xlim([px1,px2-dpx])
    ax2.set_ylim([py1,py2-dpy])
    ax2.set_title("Momentum density")

    plt.tight_layout()
    fig.savefig('frames' + '/%04d.png' % k)
    fig.clf()
    plt.close('all')
"""

def draw_frame(W, Wexact, k):
    rho = sum(W, axis=(2,3))*dpx*dpy
    phi = sum(W, axis=(0,1))*dx*dy
    Vx = sum(pxgrid*W, axis=(2,3))*dpx*dpy
    Vy = sum(pygrid*W, axis=(2,3))*dpx*dpy
    rho_exact = sum(Wexact, axis=(2,3))*dpx*dpy
    phi_exact = sum(Wexact, axis=(0,1))*dx*dy
    Vx_exact = sum(pxgrid*Wexact, axis=(2,3))*dpx*dpy
    Vy_exact = sum(pygrid*Wexact, axis=(2,3))*dpx*dpy

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(19.2,10.8), dpi=100)

    ax1.imshow(rho_exact.T, origin='lower', interpolation='none', extent=[x1,x2-dx,y1,y2-dy],vmin=amin(rho_exact), vmax=amax(rho_exact))
    ax1.quiver(xx, yy, Vx_exact, Vy_exact, color='y')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim([x1,x2-dx])
    ax1.set_ylim([y1,y2-dy])
    ax1.set_title("Spatial density (EXACT)")

    ax2.imshow(phi_exact.T, origin='lower', interpolation='none', extent=[px1,px2-dpx,py1,py2-dpy],vmin=amin(phi_exact), vmax=amax(phi_exact))
    ax2.set_xlabel('Px')
    ax2.set_ylabel('Py')
    ax2.set_xlim([px1,px2-dpx])
    ax2.set_ylim([py1,py2-dpy])
    ax2.set_title("Momentum density (EXACT)")

    ax3.imshow(rho.T, origin='lower', interpolation='none', extent=[x1,x2-dx,y1,y2-dy],vmin=amin(rho), vmax=amax(rho))
    ax3.quiver(xx, yy, Vx, Vy, color='y')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_xlim([x1,x2-dx])
    ax3.set_ylim([y1,y2-dy])
    ax3.set_title("Spatial density (NUMERIC)")

    ax4.imshow(phi.T, origin='lower', interpolation='none', extent=[px1,px2-dpx,py1,py2-dpy],vmin=amin(phi), vmax=amax(phi))
    ax4.set_xlabel('Px')
    ax4.set_ylabel('Py')
    ax4.set_xlim([px1,px2-dpx])
    ax4.set_ylim([py1,py2-dpy])
    ax4.set_title("Momentum density (NUMERIC)")

    plt.tight_layout()
    fig.savefig('frames' + '/%04d.png' % k)
    fig.clf()
    plt.close('all')

dt = (t2-t1)/1000. # the first very rough guess of time step
expU = exp(dt*dU)
expT = exp(dt*dT)
W = fftshift(f0(xgrid, ygrid, pxgrid, pygrid))
t = t1
Nt = 1

while t <= t2:
    Wexact = W_analytic(xgrid, ygrid, pxgrid, pygrid, t)
    #draw_single(Wexact, Nt)
    draw_frame(ifftshift(W), Wexact, Nt)
    W = solve_spectral(W, expU, expT)
    t += dt
    Nt += 1
