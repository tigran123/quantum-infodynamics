#!/usr/bin/env python3.7

"""
  solanim.py --- Quantum Infodynamics Tools - Solution Animator

  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  License: GPL
"""

import sys
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import load, linspace, mgrid, memmap, append, unique, sqrt
from argparse import ArgumentParser as argp
from time import time

# our own modules
from midnorm import MidpointNormalize
norm = MidpointNormalize(midpoint=0.0)

mplt.rc('font', family='serif', size=9)

p = argp(description="Quantum Infodynamics Tools - Solution Animator")
p.add_argument("-s", action="append", help="Solution data filename (multiple OK)", dest="sfilenames", required=True, default=[])
p.add_argument("-W",  action="store_true", help="Animate W(x,p,t) only", dest="Wonly")
p.add_argument("-c", action="store", help="Number of contour levels of W(x,p,t) to plot (default 100)", dest="clevels", type=int, default=100)
p.add_argument("-P", action="store", help="Number of parts to split the time range into", dest="nparts", type=int, default=1)
p.add_argument("-p", action="store", help="The part number to process in this instance", dest="part", type=int, default=1)
p.add_argument("-d", action="store", help="Frames directory", dest="framedir", required=True)
p.add_argument("-fw", action="store", help="Frame width in pixels (default 1920)", dest="framew", type=int, default=1920)
p.add_argument("-fh", action="store", help="Frame height in pixels (default 1080)", dest="frameh", type=int, default=1080)
args = p.parse_args()

framedir,Wonly,nparts,part,framew,frameh = args.framedir,args.Wonly,args.nparts,args.part,args.framew,args.frameh

def pr_exit(str):
    print("ERROR:" + str)
    sys.exit()

if nparts <= 0: pr_exit("Number of parts must be positive, but %d <= 0" % nparts)
if part <= 0 or part > nparts: pr_exit("The part number must lie between 1 and %d,  but %d <= 0" % (nparts, part))

(t,Nt,W,rho,phi,Wmin,Wmax,rho_min,rho_max,phi_min,phi_max,descr,H0,
  Wlevels,Wticks,Wfilenames,x1,x2,Nx,p1,p2,Np,H,Hmin,Hmax,E,Emin,Emax,deltaX,deltaP) = ([] for _ in range(30))

for sfilename in args.sfilenames:
    with load(sfilename + '.npz',allow_pickle=True) as data:
        t.append(data['t']);
        H.append(data['H']); H0.append(data['H0'])
        params = data['params'][()]
        E.append(data['E']);
        Emin.append(params['Emin']);
        Emax.append(params['Emax'])
        Wmin.append(params['Wmin']);
        Wmax.append(params['Wmax'])
        Wlevels.append(linspace(Wmin[-1], Wmax[-1], args.clevels));
        Wticks.append(linspace(Wmin[-1], Wmax[-1], 10))
        if not Wonly:
            rho.append(data['rho']);
            phi.append(data['phi']);
            rho_min.append(params['rho_min']);
            rho_max.append(params['rho_max'])
            phi_min.append(params['phi_min']);
            phi_max.append(params['phi_max'])
            deltaX.append(data['deltaX']);
            deltaP.append(data['deltaP']);
        Wfilenames.append(params['Wfilename']);
        Nt.append(params['Nt'])
        x1.append(params['x1']); x2.append(params['x2']); Nx.append(params['Nx'])
        p1.append(params['p1']); p2.append(params['p2']); Np.append(params['Np'])
        Hmin.append(params['Hmin']);
        Hmax.append(params['Hmax']);
        descr.append(params['descr'])

W = [memmap(filename, mode='r', dtype='float64', shape=(nt,nx,np)) for (filename,nt,nx,np) in zip(Wfilenames,Nt,Nx,Np)]

xvdx = [linspace(x1i, x2i, Nxi, endpoint=False, retstep=True) for (x1i,x2i,Nxi) in zip(x1,x2,Nx)]
pvdp = [linspace(p1i, p2i, Npi, endpoint=False, retstep=True) for (p1i,p2i,Npi) in zip(p1,p2,Np)]
dx = [a[1] for a in xvdx]
dp = [a[1] for a in pvdp]
xxpp = [mgrid[x1i:x2i-dxi:Nxi*1j, p1i:p2i-dpi:Npi*1j] for (x1i,x2i,dxi,Nxi,p1i,p2i,dpi,Npi) in zip(x1,x2,dx,Nx,p1,p2,dp,Np)]
Hlevels =  [unique(append(linspace(hmin, hmax, 10),h0)) for (hmin,hmax,h0) in zip(Hmin,Hmax,H0)]

def fmt(x, pos):
    return "%3.2f" % x

from mpl_toolkits.axes_grid1 import make_axes_locatable

def split(a, n, p):
    """Split the list 'a' into 'n' chunks and return chunk number 'p' (numbered from 1)"""
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))[p-1]

nsol = len(t)
t_longest = max(t, key=len)
time_steps = len(t_longest)

# split the entire time range into 'nparts' chunks and take chunk 'part'
time_range = split(list(range(time_steps)), nparts, part)
prog_prefix = "solanim: %d of %d: " %(part, nparts)

total_frames = len(time_range)
print(prog_prefix + "processing %d out of %d frames" % (total_frames, time_steps))
t_start = time()
frames = 0
nplots = 1 if Wonly else 3

for k in time_range:
    fig, axes = plt.subplots(nsol, nplots, figsize=(framew/100,frameh/100), dpi=100)
    s = 0
    if nsol == 1: axes_list = [axes]
    else: axes_list = axes
    for ax in axes_list:
        if Wonly:
            ax = [ax]
        xx,pp = xxpp[s][0],xxpp[s][1]
        xv = xvdx[s][0]
        pv = pvdp[s][0]
        time_index = abs(t[s] - t_longest[k]).argmin()
        ax[0].contour(xx, pp, H[s], levels=Hlevels[s], linewidths=0.5, colors='k')
        ax[0].set_title(descr[s] + ' $W(x,p,t)$')
        im = ax[0].contourf(xx, pp, W[s][time_index], levels=Wlevels[s], norm=norm, cmap=cm.bwr)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", "2%", pad="1%")
        plt.colorbar(im, cax = cax, ticks=Wticks[s], format=mplt.ticker.FuncFormatter(fmt))
        ax[0].set_ylabel('$p$')
        ax[0].set_xlabel('$x$')
        ax[0].set_xlim([x1[s],x2[s]-dx[s]])
        ax[0].set_ylim([p1[s],p2[s]-dp[s]])

        if not Wonly:
            ax[1].set_title(r"$\rho(x,t), E_0=$ % 6.3f, $E_{min}=$% 6.3f, $E_{max}=$% 6.3f" % (E[s][0],Emin[s],Emax[s]))
            rho_now = rho[s][time_index]
            ax[1].plot(xv, rho_now, color='black')
            ax[1].plot(xv, rho[s][0], color='green', label=r'$\rho(x,0)$')
            ax[1].legend(prop=dict(size=12),loc=1)
            ax[1].fill_between(xv, 0, rho_now, where=rho_now>0, color='red', interpolate=True)
            ax[1].fill_between(xv, 0, rho_now, where=rho_now<0, color='blue', interpolate=True)
            ax[1].set_ylabel(r'$\rho$')
            ax[1].set_xlabel('$x$')
            ax[1].set_xlim([x1[s],x2[s]-dx[s]])
            ax[1].text(0.05, 0.6, "t=% 6.4f\n$\Delta x$ = %.1f\n$\Delta p$ = %.1f\n$\Delta x\Delta p$=%.1f" %
                                  (t[s][time_index], deltaX[s][time_index], deltaP[s][time_index], deltaX[s][time_index]*deltaP[s][time_index]), transform=ax[1].transAxes)
            ax[1].set_ylim([1.02*rho_min[s],1.02*rho_max[s]])

            ax[2].set_title(r"Momentum density $\varphi(p,t)$")
            phi_now = phi[s][time_index]
            ax[2].plot(pv, phi_now, color='black')
            ax[2].plot(pv, phi[s][0], color='green', label=r'$\varphi(p,0)$')
            ax[2].legend(prop=dict(size=12),loc=1)
            ax[2].fill_between(pv, 0, phi_now, where=phi_now>0, color='red', interpolate=True)
            ax[2].fill_between(pv, 0, phi_now, where=phi_now<0, color='blue', interpolate=True)
            ax[2].set_ylabel(r'$\varphi$')
            ax[2].set_xlabel('$p$')
            ax[2].set_xlim([p1[s],p2[s]-dp[s]])
            ax[2].set_ylim([1.02*phi_min[s],1.02*phi_max[s]])
        s += 1

    plt.tight_layout()
    fig.savefig(framedir + '/%05d.png' % k, format='png')
    plt.close('all')
    frames += 1
    if frames%30 == 0: print(prog_prefix + "processed %d frames of %d" % (frames,total_frames))

t_end = time()
print(prog_prefix + "processed all %d frames in %.1fs (%.1f FPS)" % (total_frames,t_end-t_start,total_frames/(t_end-t_start)))
