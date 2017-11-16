"""
  Mathematical Pendulum Simulator (main program)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  This program was written in 2017 and placed in public domain.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pendulum import Pendulum
from numpy import pi, mgrid

def animate(i):
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

pend1 = Pendulum(phi=pi, phidot=3, L=1.0)
pend2 = Pendulum(phi=pi, L=0.9)
pend3 = Pendulum(phi=pi/6, L=0.4)

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
phi_range = 1.1*pi
phi_points = 200
phidot_range = 10.0
phidot_points = 200
ax2.set_xlim([-phi_range,phi_range])
ax2.set_ylim([-phidot_range,phidot_range])
points = ax2.scatter([],[], color=['b','r','g'])
phim,phidotm = mgrid[-phi_range:phi_range:phi_points*1j,-phidot_range:phidot_range:phidot_points*1j]

cn1 = ax2.contour(phim, phidotm, pend1.Hamiltonian(phim,phidotm), levels=pend1.energy(), linewidths=0.8, colors='b')
plt.clabel(cn1, fontsize=9, inline=False)
cn2 = ax2.contour(phim, phidotm, pend2.Hamiltonian(phim,phidotm), levels=pend2.energy(), linewidths=0.8, colors='r')
plt.clabel(cn2, fontsize=9, inline=False)
cn3 = ax2.contour(phim, phidotm, pend3.Hamiltonian(phim,phidotm), levels=pend3.energy(), linewidths=0.8, colors='g')
plt.clabel(cn3, fontsize=9, inline=False)

ani = animation.FuncAnimation(fig, animate, blit=True, interval=0, frames=500)
#ani.save('pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
