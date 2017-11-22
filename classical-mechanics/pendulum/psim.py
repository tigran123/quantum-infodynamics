#!/usr/local/bin/python3

"""
  Mathematical Pendulum Simulator (main program)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  This program was written in 2017 and placed in public domain.
"""

import sys
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.qt_compat import QtWidgets, QtCore
from numpy import pi, mgrid
from pendulum import Pendulum

t = 0.0 # global simulation time (has to be the same for all pendulums)
dt = 0.005 # ODE integration fixed timestep
anim_running = False # change to True to start the animation immediately
print("Loading simulation in the %s state..." % ("RUNNING" if anim_running else "STOPPED"))

def evolve_pendulums(dt):
    global t
    for p in pendulums: p.step(dt)
    t += dt

def keypress(event):
    global anim_running, dt

    if event.key == ' ':
        ani.event_source.stop() if anim_running else ani.event_source.start()
        anim_running = not anim_running
    elif event.key == '.':
        dt = abs(dt)
        evolve_pendulums(dt)
        anim_running = False
        ani.event_source.start()
    elif event.key == ',':
        dt = -abs(dt)
        evolve_pendulums(dt)
        anim_running = False
        ani.event_source.start()
    elif event.key == "delete":
        if pendulums:
            ani.event_source.stop()
            p = pendulums.pop()
            p.free()
            ani._handle_resize()
            ani._end_redraw(None)
            ani.event_source.start()

def animate(i):
    global t

    if not anim_running: ani.event_source.stop()
    time_text.set_text('Time = %.3f s' % t)
    offsets = []
    for p in pendulums:
        offsets.append([p.phi, p.phidot])
        p.line.set_data(p.position())
        p.energy_text.set_text(r'E = %.3f J, $\varphi$=%.3f' % (p.energy(), p.phi))
    points.set_offsets(offsets)

    # ignore 0'th frame because animate(0) is called THRICE by matplotlib!
    if i != 0 and anim_running: evolve_pendulums(dt)

    return tuple(p.line for p in pendulums) + tuple(p.energy_text for p in pendulums) + (time_text, points)

pendulums = [Pendulum(phi=pi, phidot=3, L=1.0, color='b'),
             Pendulum(phi=pi, L=0.9, color='r'),
             Pendulum(phi=pi/3, L=0.6, color='g'),
             Pendulum(phi=0.9*pi/3, L=0.6, color='m')]

app = QtWidgets.QApplication(sys.argv)
win = QtWidgets.QMainWindow()
fig = Figure(figsize=(19.2,10.8), dpi=100)
canvas = FigureCanvas(fig)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_aspect('equal')
ax1.set_title("Mathematical Pendulum")
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

canvas.mpl_connect('key_press_event', keypress)
ani = animation.FuncAnimation(fig, animate, blit=True, interval=0, frames=200)
#fig.tight_layout(); anim_running = True ; ani.save('pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

win.resize(int(fig.bbox.width), int(fig.bbox.height))
win.setWindowTitle("Mathematical Pendulum Simulator v0.3 (Qt)")
canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
canvas.setFocus()
win.setCentralWidget(canvas)
win.show()
sys.exit(app.exec_())
