"""
  Mathematical Pendulum Simulator (main program)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  This program was written in 2017 and placed in public domain.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pendulum import Pendulum
from numpy import pi, mgrid

t = 0.0 # global simulation time (has to be the same for all pendulums)
dt = 0.005 # ODE integration fixed timestep
anim_running = False # change to True to start the animation immediately
print("Loading simulation in the %s state..." % ("RUNNING" if anim_running else "STOPPED"))

def keypress(event):
    global anim_running, dt

    if event.key == ' ':
        ani.event_source.stop() if anim_running else ani.event_source.start()
        anim_running = not anim_running
    elif event.key == '.':
        ani.event_source.start()
        dt = abs(dt)
        anim_running = False
    elif event.key == ',':
        dt = -abs(dt)
        ani.event_source.start()
        anim_running = False
    elif event.key == "delete":
        if pendulums:
            ani.event_source.stop()
            p = pendulums.pop()
            p.line.remove()
            del(p.line)
            p.energy_text.remove()
            del(p.energy_text)
            ani.event_source.start()

def animate(i):
    global t

    if not anim_running: ani.event_source.stop()
    time_text.set_text('Time = %.3f s' % t)
    offsets = []
    for p in pendulums:
        offsets.append([p.phi,p.phidot])
        p.line.set_data(p.position())
        p.energy_text.set_text(r'E = %.3f J, $\varphi$=%.3f' % (p.energy(), p.phi))
    points.set_offsets(offsets)
    if i != 0: # don't evolve on the 0'th frame because animate(0) is called THREE times by matplotlib!
       for p in pendulums: p.step(dt)
       t += dt
    return tuple(p.line for p in pendulums) + tuple(p.energy_text for p in pendulums) + (time_text, points)
    # the commented version below is about 15% more efficient, but 10 times more unreadable
    # return sum(tuple((p.line,p.energy_text) for p in pendulums),()) + (time_text, points)

pendulums = [Pendulum(phi=pi, phidot=3, L=1.0, color='b'),
             Pendulum(phi=pi, L=0.9, color='r'),
             Pendulum(phi=pi/3, L=0.6, color='g'),
             Pendulum(phi=0.9*pi/3, L=0.6, color='m')]

fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(19.2,10.8), dpi=100)
fig.canvas.set_window_title("Mathematical Pendulum Simulator v0.2")

ax1.set_aspect('equal')
ax1.set_title("Mathematical Pendulum Simulator v0.2")
ax1.set_xlabel("$x$ (m)")
ax1.set_ylabel("$y$ (m)")
space_range = 1.5*max(p.L for p in pendulums)
ax1.set_xlim([-space_range,space_range])
ax1.set_ylim([-space_range,space_range])
texty = 0.95
time_text = ax1.text(0.02, texty, '', transform=ax1.transAxes)
texty -= 0.05

ax2.set_title("Phase Space (SPACE = pause/resume, './,' = step forward/backward)")
ax2.set_xlabel(r"$\varphi$ (rad)")
ax2.set_ylabel(r"$\dot{\varphi}$ (rad/s)")
phi_range = 1.1*pi
phi_points = 200
phidot_range = 10.0
phidot_points = 200
ax2.set_xlim([-phi_range,phi_range])
ax2.set_ylim([-phidot_range,phidot_range])
phim,phidotm = mgrid[-phi_range:phi_range:phi_points*1j,-phidot_range:phidot_range:phidot_points*1j]

colors = []
for p in pendulums:
    colors.append(p.color)
    p.line, = ax1.plot([], [], 'o-', lw=2, color=p.color)
    p.energy_text = ax1.text(0.02, texty, '', transform=ax1.transAxes, color=p.color)
    texty -= 0.05
    p.cs = ax2.contour(phim, phidotm, p.Hamiltonian(phim,phidotm), levels=p.energy(), linewidths=0.8, colors=p.color)
    p.cs.clabel(fontsize=9, inline=False)
points = ax2.scatter([],[], color=colors)

fig.canvas.mpl_connect('key_press_event', keypress)
ani = animation.FuncAnimation(fig, animate, blit=True, interval=0, frames=2000)
fig.tight_layout()

#ani.save('pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
