"""
  solanim.py --- Solution Animator
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  License: GPL
"""

import matplotlib as mplt
mplt.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import load, linspace, mgrid, amin, amax, ma, interp
import argparse as arg

mplt.rc('font', family='serif', size=12)

p = arg.ArgumentParser(description="Solution Animator")
p.add_argument("-i", action="store", help="Initial data filename", dest="ifilename", required=True)
p.add_argument("-s", action="store", help="Solution data file ame", dest="sfilename")
p.add_argument("-a", action="store", help="Save animation to filename", dest="sfilename")
args = p.parse_args()

with load(args.ifilename) as idata:
    U = idata['U']; H = idata['H']; Hmin = idata['Hmin']; Hmax = idata['Hmax']

with load(args.sfilename) as data:
    x1 = data['x1']; x2 = data['x2']; Nx = data['Nx']
    p1 = data['p1']; p2 = data['p2']; Np = data['Np']
    t1 = data['t1']; t2 = data['t2']; Nt = data['Nt']
    W = data['W']; Wmin = data['Wmin']; Wmax = data['Wmax']
    rho = data['rho']; rho_min = data['rho_min']; rho_max = data['rho_max']
    phi = data['phi']; phi_min = data['phi_min']; phi_max = data['phi_max']

xv,dx = linspace(x1, x2, Nx, endpoint=False, retstep=True)
pv,dp = linspace(p1, p2, Np, endpoint=False, retstep=True)
xx,pp = mgrid[x1:x2-dx:Nx*1j, p1:p2-dp:Np*1j]

Hlevels = linspace(Hmin, Hmax, 10)
Wlevels = linspace(Wmin, Wmax, 100)
Wticks = linspace(Wmin, Wmax, 10)

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

for k in range(Nt):
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(19.2,10.8), dpi=100)
    
    ax1.contour(xx, pp, H, levels=Hlevels, linewidths=0.5, colors='k')
    ax1.set_title("Information field $W(x,p,t)$")
    ax1.grid(True)
    im1 = ax1.contourf(xx, pp, W[...,k], levels=Wlevels, norm=norm, cmap=cm.bwr)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", "2%", pad="1%")
    plt.colorbar(im1, cax = cax, ticks=Wticks, format=mplt.ticker.FuncFormatter(fmt))
    ax1.set_ylabel(r'$p$')
    ax1.set_xlabel(r'$x$')
    ax1.set_xlim([x1,x2-dx])
    ax1.set_ylim([p1,p2-dp])

    ax2.set_title(r"Spatial density $\rho(x,t)$")
    ax2.grid(True)
    rho_now = rho[...,k]
    ax2.plot(xv, rho_now, color='black')
    ax2.fill_between(xv, 0, rho_now, where=rho_now>0, color='red', interpolate=True)
    ax2.fill_between(xv, 0, rho_now, where=rho_now<0, color='blue', interpolate=True)
    ax2.set_ylabel(r'$\rho$')
    ax2.set_xlabel(r'$x$')
    ax2.set_xlim([x1,x2-dx])
    ax2.set_ylim([1.03*rho_min,1.03*rho_max])

    ax3.set_title(r"Momentum density $\varphi(p,t)$")
    ax3.grid(True)
    phi_now = phi[...,k]
    ax3.plot(pv, phi_now, color='black')
    ax3.fill_between(pv, 0, phi_now, where=phi_now>0, color='red', interpolate=True)
    ax3.fill_between(pv, 0, phi_now, where=phi_now<0, color='blue', interpolate=True)
    ax3.set_ylabel(r'$\varphi$')
    ax3.set_xlabel(r'$p$')
    ax3.set_xlim([p1,p2-dx])
    ax3.set_ylim([1.03*phi_min,1.03*phi_max])

    plt.tight_layout()
    fig.savefig("frames/%05d.png" % k)
    fig.clf()
    plt.close('all')