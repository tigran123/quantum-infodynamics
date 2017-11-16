"""
  Mathematical Pendulum Simulator (main program)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  This program was written in 2017 and placed in public domain.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pendulum import Pendulum
from numpy import pi, linspace, mgrid, append, unique

def init():
    """initialize animation"""
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    time_text.set_text('')
    energy1_text.set_text('')
    energy2_text.set_text('')
    energy3_text.set_text('')
    return line1, line2, line3, time_text, energy1_text, energy2_text, energy3_text

def animate(i):
    """perform animation step"""
    global pend1, pend2, pend3, points
    dt = 0.005
    pend1.step(dt)
    pend2.step(dt)
    pend3.step(dt)
    
    line1.set_data(*pend1.position())
    line2.set_data(*pend2.position())
    line3.set_data(*pend3.position())
    time_text.set_text('Time = %.1f s' % pend1.t)
    energy1_text.set_text('E = %.3f J' % pend1.energy())
    energy2_text.set_text('E = %.3f J' % pend2.energy())
    energy3_text.set_text('E = %.3f J' % pend3.energy())

    points.set_offsets([[pend1.phi, pend1.phidot],
                      [pend2.phi, pend2.phidot],
                      [pend3.phi, pend3.phidot]]) 
    return line1, line2, line3, time_text, energy1_text, energy2_text, energy3_text, points

pend1 = Pendulum(phi=pi, phidot=3)
pend2 = Pendulum(phi=pi)
pend3 = Pendulum(phi=pi/6)

# for saving the animation fig,(ax1,ax2) = plt.subplots(2, 1, figsize=(19.2,10.8), dpi=100)
fig,(ax1,ax2) = plt.subplots(2, 1, figsize=(19.2,10.8), dpi=100)
fig.canvas.set_window_title("Mathematical Pendulum Simulator v0.1")

ax1.set_aspect('equal')
ax1.set_title("Mathematical Pendulum")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
space_range = 1.5*max(pend1.L, pend2.L, pend3.L)
ax1.set_xlim([-space_range,space_range])
ax1.set_ylim([-space_range,space_range])

line1, = ax1.plot([], [], 'o-', lw=2, color='b')
line2, = ax1.plot([], [], 'o-', lw=2, color='r')
line3, = ax1.plot([], [], 'o-', lw=2, color='g')
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
energy1_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes, color='b')
energy2_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes, color='r')
energy3_text = ax1.text(0.02, 0.80, '', transform=ax1.transAxes, color='g')

ax2.set_title("Phase Space")
ax2.set_xlabel(r"$\varphi$")
ax2.set_ylabel(r"$\dot{\varphi}$")
phirange = 1.1*pi
ax2.set_xlim([-phirange,phirange])
ax2.set_ylim([-10,10])
points = ax2.scatter([],[], color=['b','r','g'])
phiv = linspace(-phirange, phirange, 200)
phidotv = linspace(-10, 10, 200)
phim,phidotm = mgrid[-phirange:phirange:200j,-20:20:200j]
H = pend1.Hamiltonian(phim, phidotm)

# include the separatrix and the phase curves of our particles
extra_values = [pend1.Hamiltonian(pi,0), pend1.energy(), pend2.energy(), pend3.energy()]
Hlevels = unique(append(linspace(-9,20,8), extra_values))

#import pdb ; pdb.set_trace()
cn = ax2.contour(phim, phidotm, H, levels=Hlevels, linewidths=0.8, colors='k')
plt.clabel(cn, fontsize=9, inline=False)

ani = animation.FuncAnimation(fig, animate, blit=True, init_func=init, interval=0, frames=5000)
#ani.save('pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
