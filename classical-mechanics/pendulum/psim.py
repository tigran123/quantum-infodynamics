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

QMessageBox = QtWidgets.QMessageBox
QMainWindow = QtWidgets.QMainWindow
QApplication = QtWidgets.QApplication
Qt = QtCore.Qt

t = 0.0 # global simulation time (has to be the same for all pendulums)
dt = 0.005 # ODE integration fixed timestep
anim_running = False # change to True to start the animation immediately

def main_exit():
    print("Exiting the application")
    sys.exit()

class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(19.2,10.8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.ax1.set_aspect('equal')
        self.ax1.set_title("Mathematical Pendulum")
        self.ax1.set_xlabel("$x$ (m)")
        self.ax1.set_ylabel("$y$ (m)")
        space_range = 2.0
        self.ax1.set_xlim([-space_range,space_range])
        self.ax1.set_ylim([-space_range,space_range])
        texty = 0.95
        self.time_text = self.ax1.text(0.02, texty, '', transform=self.ax1.transAxes)
        texty -= 0.05
        self.ax2.set_title("Phase Space (SPACE = pause/resume, './,' = step forward/backward)")
        self.ax2.set_xlabel(r"$\varphi$ (rad)")
        self.ax2.set_ylabel(r"$\dot{\varphi}$ (rad/s)")
        self.phi_range = 1.1*pi
        self.phi_points = 200
        self.phidot_range = 10.0
        self.phidot_points = 200
        self.ax2.set_xlim([-self.phi_range, self.phi_range])
        self.ax2.set_ylim([-self.phidot_range, self.phidot_range])
        phim,phidotm = mgrid[-self.phi_range:self.phi_range:self.phi_points*1j,-self.phidot_range:self.phidot_range:self.phidot_points*1j]
        colors = []
        for p in pendulums:
            colors.append(p.color)
            p.line, = self.ax1.plot([], [], 'o-', lw=2, color=p.color)
            p.energy_text = self.ax1.text(0.02, texty, '', transform=self.ax1.transAxes, color=p.color)
            texty -= 0.05
            p.cs = self.ax2.contour(phim, phidotm, p.Hamiltonian(phim,phidotm), levels=p.energy(), linewidths=0.8, colors=p.color)
            p.cs.clabel(fontsize=9, inline=False)
        self.points = self.ax2.scatter([],[], color=colors)
        self.canvas.mpl_connect('key_press_event', keypress)
        self.ani = animation.FuncAnimation(self.fig, animate, blit=True, interval=0, frames=200)
        self.setWindowTitle("Output Window")
        self.resize(int(self.fig.bbox.width), int(self.fig.bbox.height))
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        self.setCentralWidget(self.canvas)
        self.show()

class ControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mathematical Pendulum Simulator v0.3 (Qt)")
        self.resize(640, 480)
        self.show()

    def closeEvent(self, event): # this doesn't catch the press on the Quit button
        reply = QMessageBox.warning(self, 'Warning', "Are you sure you want to quit?", 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
            main_exit()
        else:
            event.ignore()

def evolve_pendulums(dt):
    global t
    for p in pendulums: p.step(dt)
    t += dt

def keypress(event):
    global anim_running, dt

    if event.key == ' ':
        win1.ani.event_source.stop() if anim_running else win1.ani.event_source.start()
        anim_running = not anim_running
    elif event.key == '+':
        win1.ax1.set_xlim([-2,2])
        win1.ax1.set_ylim([-2,2])
        win1.canvas.draw()
        win1.ani._handle_resize()
        win1.ani._end_redraw(None)
    elif event.key == '-':
        win1.ax1.set_xlim([-1,1])
        win1.ax1.set_ylim([-1,1])
        win1.canvas.draw()
        win1.ani._handle_resize()
        win1.ani._end_redraw(None)
    elif event.key == '.':
        dt = abs(dt)
        evolve_pendulums(dt)
        anim_running = False
        win1.ani.event_source.start()
    elif event.key == ',':
        dt = -abs(dt)
        evolve_pendulums(dt)
        anim_running = False
        win1.ani.event_source.start()
    elif event.key == "delete":
        if pendulums:
            win1.ani.event_source.stop()
            p = pendulums.pop()
            p.free()
            win1.ani._handle_resize()
            win1.ani._end_redraw(None)
            win1.ani.event_source.start()

def animate(i):
    global t

    if not anim_running: win1.ani.event_source.stop()
    win1.time_text.set_text('Time = %.3f s' % t)
    offsets = []
    for p in pendulums:
        offsets.append([p.phi, p.phidot])
        p.line.set_data(p.position())
        p.energy_text.set_text(r'E = %.3f J, $\varphi$=%.3f' % (p.energy(), p.phi))
    win1.points.set_offsets(offsets)

    # ignore 0'th frame because animate(0) is called THRICE by matplotlib!
    if i != 0 and anim_running: evolve_pendulums(dt)

    return tuple(p.line for p in pendulums) + tuple(p.energy_text for p in pendulums) + (win1.time_text, win1.points)

pendulums = [Pendulum(phi=pi, phidot=3, L=1.0, color='b'),
             Pendulum(phi=pi, L=0.9, color='r'),
             Pendulum(phi=pi/3, L=0.6, color='g'),
             Pendulum(phi=0.9*pi/3, L=0.6, color='m')]

app = QApplication(sys.argv)
win1 = PlotWindow()
win2 = ControlWindow()

#win1.fig.tight_layout(); anim_running = True ; win1.ani.save('pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
sys.exit(app.exec_())
