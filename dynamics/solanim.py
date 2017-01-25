"""
  solanim.py --- Solution Animator
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  License: GPL
"""
import matplotlib as mplt
mplt.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import load, linspace, mgrid, amin, amax, memmap
from argparse import ArgumentParser as argp

# our own modules
from midnorm import MidpointNormalize
norm = MidpointNormalize(midpoint=0.0)

mplt.rc('font', family='serif', size=11)

p = argp(description="Solution Animator")
p.add_argument("-s", action="append", help="Solution data filename (multiple OK)", dest="sfilenames", required=True, default=[])
p.add_argument("-c", action="store", help="Number of contour levels of W(x,p,t) to plot (default 100)", dest="clevels", type=int, default=100)
p.add_argument("-P", action="store", help="Number of parts to split the time range into", dest="nparts", type=int, default=1)
p.add_argument("-p", action="store", help="The part number to process in this instance", dest="part", type=int, default=1)
p.add_argument("-d", action="store", help="Frames directory", dest="framedir", required=True)
p.add_argument("-fw", action="store", help="Frame width in pixels (default 1920)", dest="framew", type=int, default=1920)
p.add_argument("-fh", action="store", help="Frame height in pixels (default 1080)", dest="frameh", type=int, default=1080)
args = p.parse_args()

(framedir,nparts,part,framew,frameh) = (args.framedir,args.nparts,args.part,args.framew,args.frameh)

def pr_exit(str):
    print("ERROR:" + str)
    exit()

if nparts <= 0: pr_exit("Number of parts must be positive, but %d <= 0" % nparts)
if part <= 0 or part > nparts: pr_exit("The part number must lie between 1 and %d,  but %d <= 0" % (nparts, part))

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
    return "%3.2f" % x

from mpl_toolkits.axes_grid1 import make_axes_locatable

def split(a, n, p):
    """Split the list 'a' into 'n' chunks and return chunk number 'p' (numbered from 1)"""
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))[p-1]

t_longest = max(t, key=len)
s_longest = t.index(t_longest)
time_steps = len(t_longest)

nsol = len(t)

# split the entire time range into 'nparts' chunks and take chunk 'part'
time_range = split(list(range(time_steps)), nparts, part)
prog_prefix = "solanim: %d of %d: " %(part, nparts)

total_frames = len(time_range)
print(prog_prefix + "processing %d frames" % total_frames)
frames = 0
for k in time_range:
    fig, axes = plt.subplots(nsol, 3, figsize=(framew/100,frameh/100), dpi=100)
    
    s = 0
    if nsol == 1: axes_list = [axes]
    else: axes_list = axes
    for ax in axes_list:
        xx,pp = xxpp[s][0],xxpp[s][1]
        xv = xvdx[s][0]
        pv = pvdp[s][0]
        x = trajectory[s][:,0]
        p = trajectory[s][:,1]
        if s == s_longest:
            time_index = k
        else: # find an element in t[s] closest to the current time value (i.e. t_longest[k])
            time_index = abs(t[s] - t_longest[k]).argmin()
        ax[0].contour(xx, pp, H[s], levels=Hlevels[s], linewidths=0.5, colors='k')
        ax[0].set_title(descr[s] + ' $W(x,p,t)$')
        im = ax[0].contourf(xx, pp, W[s][time_index], levels=Wlevels[s], norm=norm, cmap=cm.bwr)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", "2%", pad="1%")
        plt.colorbar(im, cax = cax, ticks=Wticks[s], format=mplt.ticker.FuncFormatter(fmt))
        ax[0].plot(x, p, color='g', linestyle='--')
        ax[0].set_ylabel('$p$')
        ax[0].set_xlabel('$x$')
        ax[0].set_xlim([x1[s],x2[s]-dx[s]])
        ax[0].set_ylim([p1[s],p2[s]-dp[s]])

        ax[1].set_title(r"Spatial density $\rho(x,t)$")
        rho_now = rho[s][time_index]
        ax[1].plot(xv, rho_now, color='black')
        ax[1].fill_between(xv, 0, rho_now, where=rho_now>0, color='red', interpolate=True)
        ax[1].fill_between(xv, 0, rho_now, where=rho_now<0, color='blue', interpolate=True)
        ax[1].set_ylabel(r'$\rho$')
        ax[1].set_xlabel('$x$')
        ax[1].set_xlim([x1[s],x2[s]-dx[s]])
        ax[1].text(0.8, 0.8, "t=% 6.3f" % t[s][time_index], transform=ax[1].transAxes)
        ax[1].set_ylim([1.02*rho_min[s],1.02*rho_max[s]])

        ax[2].set_title(r"Momentum density $\varphi(p,t)$")
        phi_now = phi[s][time_index]
        ax[2].plot(pv, phi_now, color='black')
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
print(prog_prefix + "processed all %d frames" % total_frames)
