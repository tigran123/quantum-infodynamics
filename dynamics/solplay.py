#!/usr/bin/env python3.7

"""
  solplay.py --- Quantum Infodynamics Tools - Solution Playback
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  License: GPL
"""
import matplotlib as mplt
from numpy import load, linspace, mgrid, memmap, append, unique, sqrt
from argparse import ArgumentParser as argp
from time import time

p = argp(description="Quantum Infodynamics Tools - Solution Playback")
p.add_argument("-s", action="append", help="Solution data filename (multiple OK)", dest="sfilenames", required=True, default=[])
p.add_argument("-o", action="store", help="Output animation to the file", dest="ofilename")
p.add_argument("-r", action="store", help="Number of frames per second [25]", dest="fps", type=int, default=25)
p.add_argument("-np",  action="store_true", help="Do not plot momentum distribution", dest="nophi")
p.add_argument("-W",  action="store_true", help="Animate W(x,p,t) only", dest="Wonly")
p.add_argument("-l",  action="store_true", help="Pre-load solution data before animation", dest="preload")
p.add_argument("-c", action="store", help="Number of contour levels of W(x,p,t) to plot [100]", dest="clevels", type=int, default=100)
p.add_argument("-fw", action="store", help="Frame width in pixels [1920]", dest="framew", type=int, default=1920)
p.add_argument("-fh", action="store", help="Frame height in pixels [1080]", dest="frameh", type=int, default=1080)
args = p.parse_args()

Wonly, fps, ofilename, preload, nophi = args.Wonly, args.fps, args.ofilename, args.preload, args.nophi
if ofilename: preload = True
assert fps > 0

import matplotlib.pyplot as plt
from matplotlib import cm, animation

# our own modules
from midnorm import MidpointNormalize
from progress import ProgressBar

mplt.rc('font', family='serif', size=10)

(t,Nt,W,rho,phi,Wmin,Wmax,rho_min,rho_max,phi_min,phi_max,descr,H0,
  Wlevels,Wticks,Wfilenames,x1,x2,Nx,p1,p2,Np,H,U,T,Hmin,Hmax,E,Emin,Emax,X,X2,P,P2) = ([] for _ in range(34))

for sfilename in args.sfilenames:
    with load(sfilename + '.npz', allow_pickle=True) as data:
        t.append(data['t']); rho.append(data['rho']); phi.append(data['phi']); H.append(data['H'])
        U.append(data['U']); T.append(data['T']); H0.append(data['H0'])
        E.append(data['E']); X.append(data['X']); X2.append(data['X2']); P.append(data['P']); P2.append(data['P2']);
        params = data['params'][()];
        Wmin.append(params['Wmin']); Wmax.append(params['Wmax'])
        Emin.append(params['Emin']); Emax.append(params['Emax'])
        Wlevels.append(linspace(Wmin[-1], Wmax[-1], args.clevels)); Wticks.append(linspace(Wmin[-1], Wmax[-1], 10))
        rho_min.append(params['rho_min']); rho_max.append(params['rho_max'])
        phi_min.append(params['phi_min']); phi_max.append(params['phi_max'])
        Wfilenames.append(params['Wfilename']); Nt.append(params['Nt'])
        x1.append(params['x1']); x2.append(params['x2']); Nx.append(params['Nx'])
        p1.append(params['p1']); p2.append(params['p2']); Np.append(params['Np'])
        Hmin.append(params['Hmin']); Hmax.append(params['Hmax']); descr.append(params['descr'])

W = [memmap(filename, mode='r', dtype='float64', shape=(nt,nx,np)) for (filename,nt,nx,np) in zip(Wfilenames,Nt,Nx,Np)]

xvdx = [linspace(x1i, x2i, Nxi, endpoint=False, retstep=True) for (x1i,x2i,Nxi) in zip(x1,x2,Nx)]
pvdp = [linspace(p1i, p2i, Npi, endpoint=False, retstep=True) for (p1i,p2i,Npi) in zip(p1,p2,Np)]
dx = [a[1] for a in xvdx]
dp = [a[1] for a in pvdp]
xxpp = [mgrid[x1i:x2i-dxi:Nxi*1j, p1i:p2i-dpi:Npi*1j] for (x1i,x2i,dxi,Nxi,p1i,p2i,dpi,Npi) in zip(x1,x2,dx,Nx,p1,p2,dp,Np)]
Hlevels =  [unique(append(linspace(hmin, hmax, 10),h0)) for (hmin,hmax,h0) in zip(Hmin,Hmax,H0)]

from mpl_toolkits.axes_grid1 import make_axes_locatable

t_longest = max(t, key=len)
s_longest = t.index(t_longest)
time_steps = len(t_longest)
nsol = len(t)
norm = [MidpointNormalize(midpoint=0.0) for _ in range(nsol)]

if Wonly:
    nplots = 1
else:
    if nophi:
        nplots = 2
    else:
        nplots = 3

fig, axes = plt.subplots(nsol, nplots, figsize=(args.framew/100,args.frameh/100), dpi=100)
if nsol == 1: axes = [axes]

s = 0
c_artists,h_artists,U_artists,T_artists,rho_init_artists,phi_init_artists = ([] for _ in range(6))
for ax in axes:
    if Wonly:
        ax = [ax]
    xx,pp = xxpp[s][0],xxpp[s][1]
    xv = xvdx[s][0]
    pv = pvdp[s][0]
    imh = ax[0].contour(xx, pp, H[s], levels=Hlevels[s], linewidths=0.5, colors='k')
    h_artists.append(imh)
    ax[0].set_title(descr[s])
    im = ax[0].contourf(xx, pp, W[s][0], levels=Wlevels[s], norm=norm[s], cmap=cm.bwr)
    c_artists.append(im)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im, cax = cax, ticks=Wticks[s], format=mplt.ticker.FuncFormatter(lambda x, pos: "%3.1f" % x))
    ax[0].set_ylabel('$p$')
    ax[0].set_xlabel('$x$')
    ax[0].set_xlim([x1[s],x2[s]-dx[s]])
    ax[0].set_ylim([p1[s],p2[s]-dp[s]])

    if not Wonly:
        ax[1].set_title(r"$\rho(x,t), E_0=$ % 6.3f, $E_{min}=$% 6.3f, $E_{max}=$% 6.3f" % (E[s][0],Emin[s],Emax[s]))
        rho_init_artists += [ax[1].plot(xv, rho[s][0], color='green', label=r'$\rho_0(x)$')[0]]
        U_artists += [ax[1].plot(xv, U[s], color='black', label='$U(x)$')[0]]
        ax[1].legend(prop=dict(size=12),loc=1)
        ax[1].set_xlabel('$x$')
        ax[1].set_xlim([x1[s],x2[s]-dx[s]])
        ax[1].set_ylim([min(-1,1.02*rho_min[s]),1.02*rho_max[s]])

        if not nophi:
            ax[2].set_title(r"$\varphi(p,t)$")
            phi_init_artists += [ax[2].plot(pv, phi[s][0], color='green', label=r'$\varphi_0(p)$')[0]]
            T_artists += [ax[2].plot(pv, T[s], color='blue', label='$T(p)$')[0]]
            ax[2].legend(prop=dict(size=12),loc=1)
            ax[2].set_xlabel('$p$')
            ax[2].set_xlim([p1[s],p2[s]-dp[s]])
            ax[2].set_ylim([1.02*phi_min[s],1.02*phi_max[s]])
    s += 1

fig.tight_layout()
if not ofilename: fig.show()

progress = ProgressBar(time_steps, msg="Preloading" if preload else "Playing back")

frame = 0

def next_frame():
    global frame
    frame += 1
    if frame == time_steps: frame = 0

def prev_frame():
    global frame
    frame -= 1
    if frame == -1: frame = time_steps - 1

def animate_all(k):
    s = 0
    artists = []
    for ax in axes:
        if s == s_longest:
            time_index = frame
        else: # find an element in t[s] closest to the current time value (i.e. t_longest[k])
            time_index = abs(t[s] - t_longest[frame]).argmin()
        if not preload:
            for c in c_artists[s].collections: c.remove()
        xx,pp = xxpp[s][0],xxpp[s][1]
        xv,pv = xvdx[s][0],pvdp[s][0]
        c_artists[s] = ax[0].contourf(xx, pp, W[s][time_index], levels=Wlevels[s], norm=norm[s], cmap=cm.bwr)
        rho_now = rho[s][time_index]
        rho_artist, = ax[1].plot(xv, rho_now, color='black', animated=True)
        rho_plus = ax[1].fill_between(xv, 0, rho_now, where=rho_now>0, color='red', interpolate=False, animated=True)
        rho_minus = ax[1].fill_between(xv, 0, rho_now, where=rho_now<0, color='blue', interpolate=False, animated=True)
        deltaX2 = X2[s][time_index] - (X[s][time_index])**2
        deltaP2 = P2[s][time_index] - (P[s][time_index])**2
        sxsp = sqrt(deltaX2*deltaP2)
        text_artist = ax[1].text(0.05, 0.8, "E=% 6.3f\nt=% 6.4f\n$\Delta x\Delta p$=% 6.4f" %
                                            (E[s][time_index],t[s][time_index], sxsp), transform=ax[1].transAxes, animated=True)
        if not nophi:
            phi_now = phi[s][time_index]
            phi_artist, = ax[2].plot(pv, phi_now, color='black', animated=True)
            phi_plus = ax[2].fill_between(pv, 0, phi_now, where=phi_now>0, color='red', interpolate=False, animated=True)
            phi_minus = ax[2].fill_between(pv, 0, phi_now, where=phi_now<0, color='blue', interpolate=False, animated=True)
            artists.extend([T_artists[s], phi_init_artists[s],phi_artist,phi_plus,phi_minus])
        artists.extend(c_artists[s].collections + h_artists[s].collections +
                    [U_artists[s],rho_init_artists[s],rho_artist,rho_plus,rho_minus, text_artist])
        s += 1
    if preload: progress.update(frame)
    if anim_running: next_frame()
    return artists

def animate_Wonly(k):
    s = 0
    artists = []
    for ax in axes:
        if s == s_longest:
            time_index = frame
        else: # find an element in t[s] closest to the current time value (i.e. t_longest[frame])
            time_index = abs(t[s] - t_longest[frame]).argmin()
        if not preload:
            for c in c_artists[s].collections: c.remove()
        xx,pp = xxpp[s][0],xxpp[s][1]
        c_artists[s] = ax.contourf(xx, pp, W[s][time_index], levels=Wlevels[s], norm=norm[s], cmap=cm.bwr)
        artists.extend(c_artists[s].collections + h_artists[s].collections)
        s += 1
    if preload: progress.update(frame)
    if anim_running: next_frame()
    return artists

animate = animate_Wonly if Wonly else animate_all

def keypress(event):
    global anim_running
    if event.key == ' ':
        if preload and anim and anim.event_source:
            anim.event_source.stop() if anim_running else anim.event_source.start()
        anim_running ^= True
    elif event.key == 't' and not anim_running: next_frame()
    elif event.key == 'T' and not anim_running: prev_frame()

anim_running = True
fig.canvas.mpl_connect('key_press_event', keypress)

if preload:
    t_start = time()
    anim = animation.ArtistAnimation(fig, [animate(k) for k in range(time_steps)], interval=0, repeat_delay = 0, blit=True)
    print(" OK [%.1fs]" % (time()-t_start))
else:
    anim = animation.FuncAnimation(fig, animate, frames=time_steps, interval=0, repeat_delay = 0, blit=True)

if ofilename:
    t_start = time()
    print("Saving animation:", end=' ', flush=True)
    anim.save(ofilename, fps=fps, extra_args=['-vcodec', 'libx264'])
    print("OK [%.1fs]" % (time()-t_start))
else:
    plt.show()
