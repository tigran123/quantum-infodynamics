"""
  solanim.py --- Solution Animator
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  License: GPL
"""

import matplotlib as mplt
mplt.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import load, linspace, mgrid, amin, amax, ma, interp, where
import argparse as arg
from os import mkdir

mplt.rc('font', family='serif', size=12)

p = arg.ArgumentParser(description="Solution Animator")
p.add_argument("-i", action="store", help="Initial data filename", dest="ifilename", required=True)
p.add_argument("-s", action="append", help="Solution data filename (may be specified more than once)", dest="sfilenames", required=True, default=[])
p.add_argument("-d", action="store", help="Frames directory", dest="framedir", required=True)
args = p.parse_args()

framedir = args.framedir
mkdir(framedir)

with load(args.ifilename) as idata:
    params = idata['params']
    (x1,x2,Nx,p1,p2,Np) = params[:6]
    (Hmin,Hmax) = params[-2:]
    U = idata['U']; H = idata['H']

t = []; W = []; rho = []; phi = []; Wmin = []; Wmax = []; rho_min = []; rho_max = []; phi_min = []; phi_max = []
Wlevels = []
Wticks = []

for sfilename in args.sfilenames:
    with load(sfilename) as data:
        t.append(data['t']); W.append(data['W']); rho.append(data['rho']); phi.append(data['phi'])
        params = data['params']
        Wmin.append(params[0]); Wmax.append(params[1])
        Wlevels.append(linspace(Wmin[-1], Wmax[-1], 100))
        Wticks.append(linspace(Wmin[-1], Wmax[-1], 10))
        rho_min.append(params[2]); rho_max.append(params[3])
        phi_min.append(params[4]); phi_max.append(params[5])

xv,dx = linspace(x1, x2, Nx, endpoint=False, retstep=True)
pv,dp = linspace(p1, p2, Np, endpoint=False, retstep=True)
xx,pp = mgrid[x1:x2-dx:Nx*1j, p1:p2-dp:Np*1j]

Hlevels = linspace(Hmin, Hmax, 10)

def fmt(x, pos):
    return "%3.2f" % x

from mpl_toolkits.axes_grid1 import make_axes_locatable

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

t_longest = max(t, key=len)
s_longest = t.index(t_longest)
time_steps = len(t_longest)

nsol = len(t)

for k in range(time_steps):
    fig, axes = plt.subplots(nsol, 3, figsize=(19.2,10.8), dpi=100)
    
    if k%20 == 0: print("Time index k=", k)
    s = 0
    for ax in axes:
        if s == s_longest:
            time_index = k
        else:
            # find an element in t[s] closest to current time value (i.e. t_longest[k])
            time_index = abs(t[s] - t_longest[k]).argmin()
            print("current time = ", t_longest[k], "nearest time=", t[s][time_index])
        ax[0].contour(xx, pp, H, levels=Hlevels, linewidths=0.5, colors='k')
        ax[0].set_title("Information field $W(x,p,t)$")
        ax[0].grid(True)
        im = ax[0].contourf(xx, pp, W[s][time_index], levels=Wlevels[s], norm=norm, cmap=cm.bwr)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", "2%", pad="1%")
        plt.colorbar(im, cax = cax, ticks=Wticks[s], format=mplt.ticker.FuncFormatter(fmt))
        ax[0].set_ylabel('$p$')
        ax[0].set_xlabel('$x$')
        ax[0].set_xlim([x1,x2-dx])
        ax[0].set_ylim([p1,p2-dp])

        ax[1].set_title(r"Spatial density $\rho(x,t)$")
        ax[1].grid(True)
        rho_now = rho[s][time_index]
        ax[1].plot(xv, rho_now, color='black')
        ax[1].fill_between(xv, 0, rho_now, where=rho_now>0, color='red', interpolate=True)
        ax[1].fill_between(xv, 0, rho_now, where=rho_now<0, color='blue', interpolate=True)
        ax[1].set_ylabel(r'$\rho$')
        ax[1].set_xlabel('$x$')
        ax[1].set_xlim([x1,x2-dx])
        ax[1].text(3.0, 3.2, "t=% 6.3f" % t[s][time_index])
        ax[1].set_ylim([1.02*rho_min[s],1.02*rho_max[s]])

        ax[2].set_title(r"Momentum density $\varphi(p,t)$")
        ax[2].grid(True)
        phi_now = phi[s][time_index]
        ax[2].plot(pv, phi_now, color='black')
        ax[2].fill_between(pv, 0, phi_now, where=phi_now>0, color='red', interpolate=True)
        ax[2].fill_between(pv, 0, phi_now, where=phi_now<0, color='blue', interpolate=True)
        ax[2].set_ylabel(r'$\varphi$')
        ax[2].set_xlabel('$p$')
        ax[2].set_xlim([p1,p2-dp])
        ax[2].set_ylim([1.02*phi_min[s],1.02*phi_max[s]])
        s += 1

    plt.tight_layout()
    fig.savefig(framedir + '/%05d.png' % k)
    fig.clf()
    plt.close('all')
