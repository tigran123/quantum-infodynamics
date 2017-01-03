"""
 Author: Tigran Aivazian <aivazian.tigran@gmail.com>
 License: GPL
"""

from numpy import pi, exp, cos, sin, linspace, mgrid, amax, amin, newaxis, sum, abs, zeros, sqrt, real, ma, interp
import matplotlib as mplt
mplt.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import dblquad, simps
import time

mplt.rc('font', family='serif', size=12)

# select FFT implementation
#from numpy.fft import ifftshift, fftshift, fft, ifft
#from scipy.fftpack import fftshift, ifftshift, fft, ifft
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifftshift, ifft
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(10)

omega  = 1.0 # cyclic frequency (radian/s)

(Nx,Np,Nt) = (64,64,10)

(x1,x2,p1,p2) = (-3.0, 3.0, -3.0, 3.0)
(t1,t2) = (0.0, 2.0*pi/omega) # simulate one period

from matplotlib.colors import Normalize

# shift the midpoint of a colormap to a specified location (usually 0.0)
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return ma.masked_array(interp(value, x, y))

norm = MidpointNormalize(midpoint=0.0)
norm1 = MidpointNormalize(midpoint=0.0)
norm2 = MidpointNormalize(midpoint=0.0)
norm3 = MidpointNormalize(midpoint=0.0)
norm4 = MidpointNormalize(midpoint=0.0)
norm5 = MidpointNormalize(midpoint=0.0)
norm6 = MidpointNormalize(midpoint=0.0)
norm7 = MidpointNormalize(midpoint=0.0)
norm8 = MidpointNormalize(midpoint=0.0)

def qd(f, x, dx):
    hbar = 1.0 # Planck's constant  1.0545718e-34 J*s in 'natural units'
    return (f(x+1j*hbar*dx/2.) - f(x-1j*hbar*dx/2.))/(1j*hbar)

(x0,p0,sigmax,sigmap) = (0.0, 1.0, 0.2, 0.1)
Z = 1./(2.*pi*sigmax*sigmap)

def f0(x,p):
    global x0,p0,sigmax,sigmap,Z
    return Z*exp(-((x-x0)**2/(2.*sigmax**2)+(p-p0)**2/(2.*sigmap**2)))

# parameters for the potential energy
m  = 1.0 # mass (kg)

# this analytical solution is obtainable by the method of characteristics or by noticing the
# Lie group symmetry of the phase flow:
# x d/dy - y d/dx = d/dφ , where x = r cos φ, y = r sin φ
def f(x,p,t):
    global m, omega
    return f0(x*cos(omega*t) - (p/(m*omega))*sin(omega*t), p*cos(omega*t) + m*omega*x*sin(omega*t))

def U(x):
    """Potential energy U(x)"""
    global m, omega
    return m*omega**2*x**2/2.

def dUdx(x):
    """Derivative of potential energy dU(x)/dx"""
    global m, omega
    return m*omega**2*x

def H(x,p):
    """Non-relativistic Hamiltonian"""
    global m
    return p**2/(2.*m) + U(x)

xv,dx = linspace(x1, x2, Nx, endpoint=False, retstep=True)
pv,dp = linspace(p1, p2, Np, endpoint=False, retstep=True)
#dmu = dx*dp
t,dt  = linspace(t1, t2, Nt, endpoint=False, retstep=True)
xx,pp = mgrid[x1:x2-dx:Nx*1j, p1:p2-dp:Np*1j]
xxx,ppp,ttt = mgrid[x1:x2-dx:Nx*1j, p1:p2-dp:Np*1j, t1:t2-dt:Nt*1j]

# ranges in Fourier image spaces (θ is conjugated to p)
dθ = 2.*pi/(p2-p1)
θ_amp = dθ*Np/2.
θv = linspace(-θ_amp, θ_amp - dθ, Np)

# λ is conjugated to x
dλ = 2.*pi/(x2-x1)
λ_amp = dλ*Nx/2.
λv = linspace(-λ_amp, λ_amp - dλ, Nx)

# now shift them all to center zero frequency
X = fftshift(xv)[:,newaxis]
P = fftshift(pv)[newaxis,:]
θ = fftshift(θv)[newaxis,:]
λ = fftshift(λv)[:,newaxis]

Hmatrix = H(xx,pp)
E_max = amax(Hmatrix)
E_min = amin(Hmatrix)
h_levels = linspace(E_min, E_max, 10) # the number of contour levels of H(x,p) to plot

def fmt(x, pos):
    return "%3.2f" % x

from mpl_toolkits.axes_grid1 import make_axes_locatable

def draw_frame(i, fname):
    global f_numeric_spectral, f_numeric_spectral_v1, f_numeric_spectral_v2, f_analytic, Wmin, Wmax, Wmin_v1, Wmax_v1, Wmin_v2, Wmax_v2
    global Wlevels1, Wticks1, Wlevels2, Wticks2, Wlevels3, Wticks3

    fig,((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(2, 4, figsize=(19.2,10.8), dpi=100)

    rho_a = f_analytic[...,i]

    rho1 = f_numeric_spectral[...,i]
    #xav = sum(xx*rho1)*dmu
    #pav = sum(pp*rho1)*dmu
    #r_limit = (xx-xav)**2 + (pp-pav)**2 > 1.0
    #rho1[r_limit] = 0.0

    rho2 = f_numeric_spectral_v1[...,i]
    #xav = sum(xx*rho2)*dmu
    #pav = sum(pp*rho2)*dmu
    #r_limit = (xx-xav)**2 + (pp-pav)**2 > 1.0
    #rho2[r_limit] = 0.0

    rho3 = f_numeric_spectral_v2[...,i]
    #xav = sum(xx*rho3)*dmu
    #pav = sum(pp*rho3)*dmu
    #r_limit = (xx-xav)**2 + (pp-pav)**2 > 1.0
    #rho3[r_limit] = 0.0

    ax1.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='k')
    im1 = ax1.contourf(xx, pp, rho_a, levels=Wlevels1, norm=norm1, cmap=cm.bwr)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im1, cax = cax, ticks = Wticks1, format=mplt.ticker.FuncFormatter(fmt))
    E_sum = sum(H(xx,pp)*rho_a)*dx*dp
    E_dblq = dblquad(lambda x, p: H(x,p)*f(x,p,t[i]), x1, x2-dx, lambda p: p1, lambda p: p2-dp)[0]
    title = 'AN, E_sum=% 6.4f, E_dblq= %6.4f' % (E_sum, E_dblq)
    ax1.set_title(title)
    ax1.grid(True)

    ax2.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='k')
    err = amax(abs(rho1 - rho_a))
    im2 = ax2.contourf(xx, pp, rho1, levels=Wlevels2, norm=norm2, cmap=cm.bwr)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im2, cax = cax, ticks = Wticks2, format=mplt.ticker.FuncFormatter(fmt))
    E_num = sum(H(xx,pp)*rho1)*dx*dp
    E_num_simps = simps(simps(H(xx,pp)*rho1,pv),xv)
    title = 'SS1, E_num=% 6.4f, E_num_simps= %6.4f\nError=% 6.4f' % (E_num, E_num_simps, err)
    ax2.set_title(title)
    ax2.grid(True)

    ax3.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='k')
    err = amax(abs(rho2 - rho_a))
    im3 = ax3.contourf(xx, pp, rho2, levels=Wlevels3, norm=norm3, cmap=cm.bwr)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im3, cax = cax, ticks = Wticks3, format=mplt.ticker.FuncFormatter(fmt))
    E_num = sum(H(xx,pp)*rho2)*dx*dp
    E_num_simps = simps(simps(H(xx,pp)*rho2,pv),xv)
    title = 'SS2v1, E_num=% 6.4f, E_num_simps= %6.4f\nError=% 6.4f' % (E_num, E_num_simps, err)
    ax3.set_title(title)
    ax3.grid(True)

    ax4.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='k')
    err = amax(abs(rho3 - rho_a))
    im4 = ax4.contourf(xx, pp, rho3, levels = Wlevels4, norm=norm4, cmap=cm.bwr)
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im4, cax = cax, ticks = Wticks4, format=mplt.ticker.FuncFormatter(fmt))
    E_num = sum(H(xx,pp)*rho3)*dx*dp
    E_num_simps = simps(simps(H(xx,pp)*rho3,pv),xv)
    title = 'SS2v2, E_num=% 6.4f, E_num_simps= %6.4f\nError=% 6.4f' % (E_num, E_num_simps, err)
    ax4.set_title(title)
    ax4.grid(True)

    ax5.set_title('Wmin=%8.6f, Wmax=%8.6f' % (Wmin_a, Wmax_a))
    ax5.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='k')
    im5 = ax5.imshow(rho_a.T, origin='lower', interpolation='none',
              extent=[x1,x2-dx,p1,p2-dp], vmin=Wmin_a, vmax=Wmax_a, norm=norm5, cmap=cm.bwr)
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im5, cax = cax, ticks = Wticks1, format=mplt.ticker.FuncFormatter(fmt))
    ax5.grid(True)

    ax6.set_title('Wmin=%8.6f, Wmax=%8.6f' % (Wmin, Wmax))
    ax6.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='k')
    im6 = ax6.imshow(rho1.T, origin='lower', interpolation='none',
              extent=[x1,x2-dx,p1,p2-dp], vmin=Wmin, vmax=Wmax, norm=norm6, cmap=cm.bwr)
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im6, cax = cax, ticks = Wticks2, format=mplt.ticker.FuncFormatter(fmt))
    plt.colorbar(im6, cax = cax, format=mplt.ticker.FuncFormatter(fmt))
    ax6.grid(True)

    ax7.set_title('Wmin=%8.6f, Wmax=%8.6f' % (Wmin_v1, Wmax_v1))
    ax7.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='k')
    im7 = ax7.imshow(rho2.T, origin='lower', interpolation='none',
              extent=[x1,x2-dx,p1,p2-dp], vmin=Wmin_v1, vmax=Wmax_v1, norm=norm7, cmap=cm.bwr)
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im7, cax = cax, ticks = Wticks3, format=mplt.ticker.FuncFormatter(fmt))
    ax7.grid(True)

    ax8.set_title('Wmin=%8.6f, Wmax=%8.6f' % (Wmin_v2, Wmax_v2))
    ax8.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='k')
    im8 = ax8.imshow(rho3.T, origin='lower', interpolation='none',
              extent=[x1,x2-dx,p1,p2-dp], vmin=Wmin_v2, vmax=Wmax_v2, norm=norm8, cmap=cm.bwr)
    divider = make_axes_locatable(ax8)
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im8, cax = cax, ticks = Wticks4, format=mplt.ticker.FuncFormatter(fmt))
    ax8.grid(True)

    plt.tight_layout()
    fig.savefig(fname)
    fig.clf()
    plt.close('all')

# solve using Spectral Split Propagator (Exponential Integrator)
f_sol_spectral = zeros((Nx, Np, Nt))
f_sol_spectral_v1 = zeros((Nx, Np, Nt))
f_sol_spectral_v2 = zeros((Nx, Np, Nt))

def F_ptoθ(W):
    return fft(W, axis=1)

def F_θtop(W):
    return ifft(W, axis=1)

def F_xtoλ(W):
    return fft(W, axis=0)

def F_λtox(W):
    return ifft(W, axis=0)

f_init = f0(xx,pp)
#r_limit = (xx-x0)**2 + (pp-p0)**2 > 1.0
#f_init[r_limit] = 0.0
f_init = fftshift(f_init)

# first order spectral split operator
def solve_spectral():
    global X, P, θ, λ, xx, pp, f_sol_spectral, f_init
    t_start,c_start = time.time(),time.clock()
    expU = exp(dt*qd(U,X,1j*θ))
    expλ = exp(-1j*dt*P*λ/m)
    f_sol_spectral[...,0] = f_init
    for k in range(Nt-1):
        B = F_ptoθ(f_sol_spectral[...,k]) # (x,p) -> (x,θ)
        B *= expU
        B = F_θtop(B) # (x,θ) -> (x,p)
        B = F_xtoλ(B) # (x,p) -> (λ,p)
        B *= expλ
        B = F_λtox(B) # (λ,p) -> (x,p)
        f_sol_spectral[...,k+1] = real(B) # to avoid python warning
    return (time.time() - t_start, time.clock() - c_start)

# second order spectral split operator
def solve_spectral_second_order():
    global X, P, θ, λ, xx, pp, f_sol_spectral_v1, f_init
    t_start,c_start = time.time(),time.clock()
    expU = exp(dt*qd(U,X,1j*θ))
    expλ = exp(-1j*dt*P*λ/(2.*m))
    f_sol_spectral_v1[...,0] = f_init
    for k in range(Nt-1):
        B = F_xtoλ(f_sol_spectral_v1[...,k]) # (x,p) -> (λ,p)
        B *= expλ
        B = F_λtox(B) # (λ,p) -> (x,p)
        B = F_ptoθ(B) # (x,p) -> (x,θ)
        B *= expU
        B = F_θtop(B) # (x,θ) -> (x,p)
        B = F_xtoλ(B) # (x,p) -> (λ,p)
        B *= expλ
        B = F_λtox(B) # (λ,p) -> (x,p)
        f_sol_spectral_v1[...,k+1] = real(B) # to avoid python warning
    return (time.time() - t_start, time.clock() - c_start)

# second order spectral split operator (alternative propagator ordering)
def solve_spectral_second_order_v2():
    global X, P, θ, λ, xx, pp, f_sol_spectral_v2, f_init
    expU = exp(dt*qd(U,X,1j*θ)/2.)
    expλ = exp(-1j*dt*P*λ/m)
    t_start,c_start = time.time(),time.clock()
    f_sol_spectral_v2[...,0] = f_init
    for k in range(Nt-1):
        B = F_ptoθ(f_sol_spectral_v2[...,k]) # (x,p) -> (x,θ)
        B *= expU
        B = F_θtop(B) # (x,θ) -> (x,p)
        B = F_xtoλ(B) # (x,p) -> (λ,p)
        B *= expλ
        B = F_ptoθ(B) # (λ,p) -> (λ,θ)
        B = F_λtox(B) # (λ,θ) -> (x,θ)
        B *= expU
        B = F_θtop(B) # (x,θ) -> (x,p)
        f_sol_spectral_v2[...,k+1] = real(B) # to avoid python warning
    return (time.time() - t_start, time.clock() - c_start)

f_analytic = f(xxx,ppp,ttt)

# first order method
t_end,c_end = solve_spectral()
f_numeric_spectral = ifftshift(f_sol_spectral, axes=(0,1))
err1 = amax(abs(f_analytic - f_numeric_spectral))
print("SS1: (%3.1fs/%3.1fs): (%dx%dx%d) Error=%07.6f" % (t_end, c_end, Nx, Np, Nt, err1))

# second order method
t_end,c_end = solve_spectral_second_order()
f_numeric_spectral_v1 = ifftshift(f_sol_spectral_v1, axes=(0,1))
err2 = amax(abs(f_analytic - f_numeric_spectral_v1))
print("SS2 v1: (%3.1fs/%3.1fs): (%dx%dx%d) Error=%07.6f" % (t_end, c_end, Nx, Np, Nt, err2))

# second order v2 method
t_end,c_end = solve_spectral_second_order_v2()
f_numeric_spectral_v2 = ifftshift(f_sol_spectral_v2, axes=(0,1))
err3 = amax(abs(f_analytic - f_numeric_spectral_v2))
print("SS2 v2: (%3.1fs/%3.1fs): (%dx%dx%d) Error=%07.6f" % (t_end, c_end, Nx, Np, Nt, err3))

#print("%d\t\t%d\t\t%d\t\t%07.6f\t\t%07.6f\t\t%07.6f" % (Nx, Np, Nt, err1, err2, err3))
#exit()

(Wmin_a,Wmax_a) = (amin(f_analytic), amax(f_analytic))
Wlevels1 = linspace(Wmin_a, Wmax_a, 100)
Wticks1 = linspace(Wmin_a, Wmax_a, 10)
print("Wmin_a=", Wmin_a, "Wmax_a=", Wmax_a)

(Wmin,Wmax) = (amin(f_numeric_spectral), amax(f_numeric_spectral))
Wlevels2 = linspace(Wmin, Wmax, 100)
Wticks2 = linspace(Wmin, Wmax, 10)
print("Wmin=", Wmin, "Wmax=", Wmax)

(Wmin_v1,Wmax_v1) = (amin(f_numeric_spectral_v1), amax(f_numeric_spectral_v1))
Wlevels3 = linspace(Wmin_v1, Wmax_v1, 100)
Wticks3 = linspace(Wmin_v1, Wmax_v1, 10)
print("Wmin_v1=", Wmin_v1, "Wmax_v1=", Wmax_v1)

(Wmin_v2,Wmax_v2) = (amin(f_numeric_spectral_v2), amax(f_numeric_spectral_v2))
Wlevels4 = linspace(Wmin_v2, Wmax_v2, 100)
Wticks4 = linspace(Wmin_v2, Wmax_v2, 10)
print("Wmin_v2=", Wmin_v2, "Wmax_v2=", Wmax_v2)

for i in range(Nt):
    draw_frame(i, 'frames/%04d.png' % i)
