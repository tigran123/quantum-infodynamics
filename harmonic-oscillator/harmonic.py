import matplotlib as mplt
mplt.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, simps
import time

# select FFT implementation
#from numpy.fft import ifftshift, fftshift, fft, ifft
from scipy.fftpack import fftshift, ifftshift, fft, ifft
#import pyfftw
#from pyfftw.interfaces.numpy_fft import fft, fftshift, ifftshift, ifft
#pyfftw.interfaces.cache.enable()
#pyfftw.interfaces.cache.set_keepalive_time(10)

ω  = 0.5 # cyclic frequency (radian/s)
π = np.pi # number π=3.1415926...

(Nx,Np,Nt) = (512,512,10)

(x1,x2,p1,p2) = (-7.0, 7.0, -10.0, 10.0)
(t1,t2) = (0.0, 2.0*π/ω) # simulate one period

#hbar = 1.0545718e-34 # Planck's constant in J*s (SI)
hbar = 1.0 # Planck's constant in 'natural units'

def qd(f, x, dx):
    global hbar
    return (f(x+1j*hbar*dx/2.) - f(x-1j*hbar*dx/2.))/(1j*hbar)

(x0,p0,σx,σp) = (0.0, 3.0, 0.7, 0.2)

def f0(x,p):
    global x0,p0,σx,σp
    return (1./(2.*π*σx*σp))*np.exp(-((x-x0)**2/(2.*σx**2)+(p-p0)**2/(2.*σp**2)))

# parameters for the potential energy
m  = 3.0 # mass (kg)

# this analytical solution is obtainable by the method of characteristics or by noticing the
# Lie group symmetry of the phase flow:
# x d/dy - y d/dx = d/dφ , where x = r cos φ, y = r sin φ
def f(x,p,t):
    global m, ω
    return f0(x*np.cos(ω*t) - (p/(m*ω))*np.sin(ω*t), p*np.cos(ω*t) + m*ω*x*np.sin(ω*t))

# potential energy U(x)
def U(x):
    global m, ω
    return m*ω**2*x**2/2.

def dUdx(x):
    global m, ω
    return m*ω**2*x

# non-relativistic Hamiltonian
def H(x,p):
    global m
    return p**2/(2.*m) + U(x)

xv,dx = np.linspace(x1, x2, Nx, endpoint=False, retstep=True)
pv,dp = np.linspace(p1, p2, Np, endpoint=False, retstep=True)
t,dt  = np.linspace(t1, t2, Nt, endpoint=False, retstep=True)
xx,pp = np.mgrid[x1:x2-dx:Nx*1j, p1:p2-dp:Np*1j]
xxx,ppp,ttt = np.mgrid[x1:x2-dx:Nx*1j, p1:p2-dp:Np*1j, t1:t2-dt:Nt*1j]

# ranges in Fourier image spaces (θ is conjugated to p)
dθ = 2.*π/(p2-p1)
θ_amp = dθ*Np/2.
θv = np.linspace(-θ_amp, θ_amp - dθ, Np)

# λ is conjugated to x
dλ = 2.*π/(x2-x1)
λ_amp = dλ*Nx/2.
λv = np.linspace(-λ_amp, λ_amp - dλ, Nx)

# now shift them all to center zero frequency
X = fftshift(xv)[:,np.newaxis]
P = fftshift(pv)[np.newaxis,:]
θ = fftshift(θv)[np.newaxis,:]
λ = fftshift(λv)[:,np.newaxis]

Hmatrix = H(xx,pp)
E_max = np.amax(Hmatrix)
E_min = np.amin(Hmatrix)
h_levels = np.linspace(E_min, E_max, 10) # the number of contour levels of H(x,p) to plot

Vx = pp/m
Vp = -dUdx(xx)

def vector_field(ax):
    ax.quiver(xx, pp, Vx, Vp, color='y', linewidth=.3)

def draw_frame(i, fname):
    global f_numeric_spectral, f_numeric_spectral_v1, f_numeric_fd, f_analytic

    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(19.2,10.8), dpi=100)

    ax1.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='w')
    rho_a = f_analytic[...,i]
    ax1.imshow(rho_a.T, origin='lower', interpolation='none', extent=[x1,x2-dx,p1,p2-dp],vmin=np.amin(rho_a), vmax=np.amax(rho_a))
    #vector_field(ax1)
    E_sum = np.sum(H(xx,pp)*rho_a)*dx*dp
    E_dblq = dblquad(lambda x, p: H(x,p)*f(x,p,t[i]), x1, x2-dx, lambda p: p1, lambda p: p2-dp)[0]
    title = 'AN, E_sum=% 6.4f, E_dblq= %6.4f' % (E_sum, E_dblq)
    ax1.set_title(title)

    ax2.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='w')
    rho = f_numeric_spectral[...,i]
    err = np.amax(np.abs(rho - rho_a))
    ax2.imshow(rho.T, origin='lower', interpolation='none', extent=[x1,x2-dx,p1,p2-dp],vmin=np.amin(rho), vmax=np.amax(rho))
    #vector_field(ax2)
    E_num = np.sum(H(xx,pp)*rho)*dx*dp
    E_num_simps = simps(simps(H(xx,pp)*rho,pv),xv)
    title = 'SS1, E_num=% 6.4f, E_num_simps= %6.4f\nError=% 6.4f' % (E_num, E_num_simps, err)
    ax2.set_title(title)

    ax3.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='w')
    rho = f_numeric_spectral_v1[...,i]
    err = np.amax(np.abs(rho - rho_a))
    ax3.imshow(rho.T, origin='lower', interpolation='none', extent=[x1,x2-dx,p1,p2-dp],vmin=np.amin(rho), vmax=np.amax(rho))
    #vector_field(ax3)
    E_num = np.sum(H(xx,pp)*rho)*dx*dp
    E_num_simps = simps(simps(H(xx,pp)*rho,pv),xv)
    title = 'SS2v1, E_num=% 6.4f, E_num_simps= %6.4f\nError=% 6.4f' % (E_num, E_num_simps, err)
    ax3.set_title(title)

    ax4.contour(xx, pp, Hmatrix, h_levels, linewidths=0.25, colors='w')
    rho = f_numeric_spectral_v2[...,i]
    err = np.amax(np.abs(rho - rho_a))
    ax4.imshow(rho.T, origin='lower', interpolation='none', extent=[x1,x2-dx,p1,p2-dp],vmin=np.amin(rho), vmax=np.amax(rho))
    #vector_field(ax4)
    E_num = np.sum(H(xx,pp)*rho)*dx*dp
    E_num_simps = simps(simps(H(xx,pp)*rho,pv),xv)
    title = 'SS2v2, E_num=% 6.4f, E_num_simps= %6.4f\nError=% 6.4f' % (E_num, E_num_simps, err)
    ax4.set_title(title)

    plt.tight_layout()
    fig.savefig(fname)
    fig.clf()
    plt.close('all')

# solve using Spectral Split Propagator (Exponential Integrator)
f_sol_spectral = np.zeros((Nx, Np, Nt))
f_sol_spectral_v1 = np.zeros((Nx, Np, Nt))
f_sol_spectral_v2 = np.zeros((Nx, Np, Nt))

def F_ptoθ(W):
    return fft(W, axis=1)

def F_θtop(W):
    return ifft(W, axis=1)

def F_xtoλ(W):
    return fft(W, axis=0)

def F_λtox(W):
    return ifft(W, axis=0)

# first order spectral split operator
def solve_spectral():
    global X, P, θ, λ, xx, pp
    t_start,c_start = time.time(),time.clock()
    expU = np.exp(dt*qd(U,X,1j*θ))
    expλ = np.exp(-1j*dt*P*λ/m)
    f_sol_spectral[...,0] = fftshift(f0(xx,pp))
    for k in range(Nt-1):
        B = F_ptoθ(f_sol_spectral[...,k]) # (x,p) -> (x,θ)
        B *= expU
        B = F_θtop(B) # (x,θ) -> (x,p)
        B = F_xtoλ(B) # (x,p) -> (λ,p)
        B *= expλ
        B = F_λtox(B) # (λ,p) -> (x,p)
        f_sol_spectral[...,k+1] = np.real(B) # to avoid python warning
    return (time.time() - t_start, time.clock() - c_start)

# second order spectral split operator
def solve_spectral_second_order():
    global X, P, θ, λ, xx, pp
    t_start,c_start = time.time(),time.clock()
    expU = np.exp(dt*qd(U,X,1j*θ))
    expλ = np.exp(-1j*dt*P*λ/(2.*m))
    f_sol_spectral_v1[...,0] = fftshift(f0(xx,pp))
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
        f_sol_spectral_v1[...,k+1] = np.real(B) # to avoid python warning
    return (time.time() - t_start, time.clock() - c_start)

# second order spectral split operator (alternative propagator ordering)
def solve_spectral_second_order_v2():
    global X, P, θ, λ, xx, pp
    expU = np.exp(dt*qd(U,X,1j*θ)/2.)
    expλ = np.exp(-1j*dt*P*λ/m)
    t_start,c_start = time.time(),time.clock()
    f_sol_spectral_v2[...,0] = fftshift(f0(xx,pp))
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
        f_sol_spectral_v2[...,k+1] = np.real(B) # to avoid python warning
    return (time.time() - t_start, time.clock() - c_start)

f_analytic = f(xxx,ppp,ttt)

# first order method
t_end,c_end = solve_spectral()
f_numeric_spectral = ifftshift(f_sol_spectral, axes=(0,1))
err1 = np.amax(np.abs(f_analytic - f_numeric_spectral))
print("SS1: (%3.1fs/%3.1fs): (%dx%dx%d) Error=%07.6f" % (t_end, c_end, Nx, Np, Nt, err1))

# second order method
t_end,c_end = solve_spectral_second_order()
f_numeric_spectral_v1 = ifftshift(f_sol_spectral_v1, axes=(0,1))
err2 = np.amax(np.abs(f_analytic - f_numeric_spectral_v1))
print("SS2 v1: (%3.1fs/%3.1fs): (%dx%dx%d) Error=%07.6f" % (t_end, c_end, Nx, Np, Nt, err2))

# second order v2 method
t_end,c_end = solve_spectral_second_order_v2()
f_numeric_spectral_v2 = ifftshift(f_sol_spectral_v2, axes=(0,1))
err3 = np.amax(np.abs(f_analytic - f_numeric_spectral_v2))
print("SS2 v2: (%3.1fs/%3.1fs): (%dx%dx%d) Error=%07.6f" % (t_end, c_end, Nx, Np, Nt, err3))

#print("%d\t\t%d\t\t%d\t\t%07.6f\t\t%07.6f\t\t%07.6f" % (Nx, Np, Nt, err1, err2, err3))
#exit()

for i in range(Nt):
    draw_frame(i, 'frames/%04d.png' % i)
