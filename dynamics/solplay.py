"""
  solplay.py --- Solution Playback
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  License: GPL
"""
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from numpy import load, linspace, mgrid, amin, amax, memmap
from argparse import ArgumentParser as argp

# our own modules
from midnorm import MidpointNormalize
from progress import ProgressBar

mplt.rc('font', family='serif', size=10)

p = argp(description="Solution Animator")
p.add_argument("-s", action="append", help="Solution data filename (multiple OK)", dest="sfilenames", required=True, default=[])
p.add_argument("-c", action="store", help="Number of contour levels of W(x,p,t) to plot (default 20)", dest="clevels", type=int, default=20)
p.add_argument("-fw", action="store", help="Frame width in pixels (default 1920)", dest="framew", type=int, default=1920)
p.add_argument("-fh", action="store", help="Frame height in pixels (default 1080)", dest="frameh", type=int, default=1080)
args = p.parse_args()

(t,Nt,W,rho,phi,Wmin,Wmax,rho_min,rho_max,phi_min,phi_max,trajectory,descr,
  Wlevels,Wticks,Wfilenames,x1,x2,Nx,p1,p2,Np,H,Hmin,Hmax) = ([] for _ in range(25))

for sfilename in args.sfilenames:
    with load(sfilename + '.npz') as data:
        t.append(data['t']); rho.append(data['rho']); phi.append(data['phi']); H.append(data['H'])
        trajectory.append(data['trajectory']); params = data['params'][()]
        Wmin.append(params['Wmin']); Wmax.append(params['Wmax'])
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
Hlevels =  [linspace(hmin, hmax, 10) for (hmin,hmax) in zip(Hmin,Hmax)]

def fmt(x, pos):
    return "%3.1f" % x

from mpl_toolkits.axes_grid1 import make_axes_locatable

t_longest = max(t, key=len)
s_longest = t.index(t_longest)
time_steps = len(t_longest)
nsol = len(t)
norm = [MidpointNormalize(midpoint=0.0) for _ in range(nsol)]

fig, axes = plt.subplots(nsol, 3, figsize=(args.framew/100,args.frameh/100), dpi=100)
if nsol == 1: axes = [axes]

s = 0
c_artists,h_artists,traj_artists,rho_artists,phi_artists,text_artists = ([] for _ in range(6))
for ax in axes:
    xx,pp = xxpp[s][0],xxpp[s][1]
    xv = xvdx[s][0]
    pv = pvdp[s][0]
    x = trajectory[s][:,0]
    p = trajectory[s][:,1]
    imh = ax[0].contour(xx, pp, H[s], levels=Hlevels[s], linewidths=0.5, colors='k')
    h_artists.append(imh)
    ax[0].set_title(descr[s])
    im = ax[0].contourf(xx, pp, W[s][0], levels=Wlevels[s], norm=norm[s], cmap=cm.bwr)
    c_artists.append(im)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im, cax = cax, ticks=Wticks[s], format=mplt.ticker.FuncFormatter(fmt))
    traj, = ax[0].plot(x, p, color='g', linestyle='--')
    traj_artists.append(traj)
    ax[0].set_ylabel('$p$')
    ax[0].set_xlabel('$x$')
    ax[0].set_xlim([x1[s],x2[s]-dx[s]])
    ax[0].set_ylim([p1[s],p2[s]-dp[s]])

    ax[1].set_title(r"Spatial density $\rho(x,t)$")
    rho_artist, = ax[1].plot(xv, rho[s][0], color='black')
    rho_artists.append(rho_artist)
    text_artist = ax[1].text(0.8, 0.8, "", transform=ax[1].transAxes, animated=True)
    text_artists.append(text_artist)
    ax[1].set_xlabel('$x$')
    ax[1].set_xlim([x1[s],x2[s]-dx[s]])
    ax[1].set_ylim([1.02*rho_min[s],1.02*rho_max[s]])

    ax[2].set_title(r"Momentum density $\varphi(p,t)$")
    phi_artist, = ax[2].plot(pv, phi[s][0], color='black')
    phi_artists.append(phi_artist)
    ax[2].set_xlabel('$p$')
    ax[2].set_xlim([p1[s],p2[s]-dp[s]])
    ax[2].set_ylim([1.02*phi_min[s],1.02*phi_max[s]])
    s += 1

fig.tight_layout()
fig.show()

ims = []
progress = ProgressBar(time_steps, msg="Playing back frames")

def animate(k):
    s = 0
    artists = []
    for ax in axes:
        if s == s_longest:
            time_index = k
        else: # find an element in t[s] closest to the current time value (i.e. t_longest[k])
            time_index = abs(t[s] - t_longest[k]).argmin()
        for c in c_artists[s].collections: c.remove()
        c_artists[s] = ax[0].contourf(xx, pp, W[s][time_index], levels=Wlevels[s], norm=norm[s], cmap=cm.bwr, animated=True)
        text_artists[s].set_text("t=% 6.3f" % t[s][time_index])
        rho_artists[s].set_ydata(rho[s][time_index])
        rho_artists[s].set(color='blue')
        phi_artists[s].set_ydata(phi[s][time_index])
        phi_artists[s].set(color='blue')
        artists.extend(c_artists[s].collections + h_artists[s].collections +
                        [traj_artists[s],rho_artists[s],phi_artists[s],text_artists[s]])
        s += 1
    progress.update(k)
    return artists

ani = animation.FuncAnimation(fig, animate, frames=time_steps, interval=0, repeat_delay = 1000, blit=True)
plt.show()
