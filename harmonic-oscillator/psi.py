"""
 Author: Tigran Aivazian <aivazian.tigran@gmail.com>
 License: GPL
"""

import numpy as np
import matplotlib as mplt
mplt.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermval
from math import factorial

mplt.rc('text', usetex=True)
mplt.rc('font', family='serif', size=15)

hbar = 1.0
ω = 1.0
mass = 1.0

x1 = -20.
x2 = 20.
t1 = 0.
t2 = 2.*np.pi/ω
Nx = 2000
Nt = 300
xv,dx = np.linspace(x1, x2, Nx, endpoint=False, retstep=True)
tv,dt  = np.linspace(t1, t2, Nt, endpoint=False, retstep=True)
xx,tt = np.mgrid[x1:x2-dx:Nx*1j, t1:t2-dt:Nt*1j]

b = mass*ω/hbar
B = np.sqrt(b)

def psi(x,t,n):
    global b, B, ω
    a = np.exp(-1j*ω*t/2. - b*x**2/2.)
    c = np.array([np.exp(-1j*k*ω*t)/np.sqrt(float(2**k*factorial(k))) for k in range(n+1)])
    ret = a*hermval(x*B,c,tensor=False)
    return ret

# map the plot number to the quantum energy level n
nmap = [0, 3, 20, 60, 100]
nmap_len = len(nmap)

data = np.zeros((Nx,Nt,nmap_len))
maxv = np.zeros(nmap_len)

print("Calculating data:", end=" ", flush=True)
from time import time,clock
t_start,c_start = time(),clock()
for m in range(nmap_len):
    d = np.abs(psi(xx, tt, nmap[m]))**2
    norm = np.sum(d)*dx
    data[...,m] = d/norm
    maxv[m] = np.amax(data[...,m])
t_end,c_end = time() - t_start,clock() - c_start
print("%3.1fs/%3.1fs" % (t_end, c_end))

print("Plotting frames...")
for k in range(Nt):
    fig, axes = plt.subplots(3, 2, figsize=(19.2,10.8), dpi=100)

    axnum = 0
    for ax in axes.reshape(-1)[:-1]:
        ax.set_title(r"$|\Psi_{%d}(x,t)|^2$" % nmap[axnum])
        d = data[:,k,axnum]
        ax.plot(xv, d, color='black')
        ax.fill_between(xv, 0, d, where=d>0, color='red', interpolate=True)
        ax.grid()
        ax.axis()
        ax.set_ylabel(r'$|\Psi_n|^2$')
        ax.set_xlabel(r'$x$')
        ax.set_xlim([x1,x2-dx])
        ax.set_ylim(0.0, 1.02*maxv[m])
        axnum += 1

    ax = axes[2,1]
    ax.axis("off")
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    xstart = -0.05
    textline = r"Frame=%03d/%03d, $t=$%6.4f/%6.4f$, \Delta t=$%3.2f$, N_x\times N_t=%d\times %d$" % (k+1, Nt, tv[k], t2-dt, dt, Nx, Nt)
    ax.text(xstart, 0.99, textline)
    textline = r"$\hat{H} = -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + \frac{m\omega^2 x^2}{2}, m = $%2.1f$, \omega=$%3.1f$, \hbar=$%3.1f" % (mass, ω, hbar)
    ax.text(xstart, 0.75, textline)
    textline = r"$\hat{H}\varphi_n = E_n\varphi_n, E_n =\hbar\omega(n+\frac{1}{2}), \varphi_n(x) = (\frac{m\omega}{2\hbar})^{\frac{1}{4}}\frac{1}{\sqrt{2^n n!}}H_n(x\sqrt{\frac{m\omega}{\hbar}})$"
    ax.text(xstart, 0.55, textline)
    textline = r"$\Psi_n(x,t) = \sum\limits_{k=0}^n e^{-\frac{i E_k t}{\hbar}} \varphi_n(x) \propto \exp\left\{-\frac{i\omega t}{2} - \frac{m\omega x^2}{2\hbar}\right\} \sum\limits_{k=0}^n e^{-ik\omega t}\frac{H_k(x\sqrt{\frac{m\omega}{\hbar}})}{\sqrt{2^k k!}}$"
    ax.text(xstart, 0.30, textline)
    textline = r"$\tilde{E}_n = \sum\limits_{k=0}^n E_k$"
    ax.text(xstart, 0.05, textline)

    plt.tight_layout()
    fig.savefig('frames' + '/%04d.png' % k)
    fig.clf()
    plt.close('all')
