#!/usr/bin/env python3.8

"""
  Mathematical Pendulum Simulator (main program)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  Released under GPLv3, 2017
"""

import sys

from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

import numpy as np

from qtapi import *
from pendulum import Pendulum

COMPANY = 'QuantumInfodynamics.com'
PROGRAM = 'Mathematical Pendulum Simulator v0.3 (Qt)'
PROG = 'MathematicalPendulum'
LOGO = 'icons/Logo.jpg'

t = 0.0 # global simulation time (has to be the same for all pendulums)
dt = 0.005 # ODE integration fixed timestep
anim_running = False # change to True to start the animation immediately

def main_exit():
     global settings
     settings.setValue('plot_geometry', winp.saveGeometry())
     settings.setValue('plot_windowState', winp.saveState())
     settings.setValue('control_geometry', winc.saveGeometry())
     settings.setValue('control_windowState', winc.saveState())
     del settings # to force the writing of settings to storage
     print('Exiting the program')
     sys.exit()

class PlotWindow(QMainWindow):
    def __init__(self, geometry = None, state = None):
        super().__init__()
        self.fig = Figure(figsize=(19.2,10.8))
        self.canvas = FigureCanvas(self.fig)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Mathematical Pendulum')
        self.ax1.set_xlabel('$x$ (m)')
        self.ax1.set_ylabel('$y$ (m)')
        space_range = 2.0
        self.ax1.set_xlim([-space_range,space_range])
        self.ax1.set_ylim([-space_range,space_range])
        self.ax2.set_title('Phase Space Trajectories')
        self.ax2.set_xlabel(r'$\varphi$ (rad)')
        self.ax2.set_ylabel(r'$\dot{\varphi}$ (rad/s)')
        self.phi_range = 1.1*np.pi
        self.phi_points = 200
        self.phidot_range = 10.0
        self.phidot_points = 200
        self.ax2.set_xlim([-self.phi_range, self.phi_range])
        self.ax2.set_ylim([-self.phidot_range, self.phidot_range])
        phim,phidotm = np.mgrid[-self.phi_range:self.phi_range:self.phi_points*1j,-self.phidot_range:self.phidot_range:self.phidot_points*1j]
        colors = []
        texty = 0.95
        for p in pendulums:
            colors.append(p.color)
            p.line, = self.ax1.plot([], [], 'o-', lw=2, color=p.color)
            p.energy_text = self.ax1.text(0.02, texty, '', transform=self.ax1.transAxes, color=p.color)
            texty -= 0.05
            p.cs = self.ax2.contour(phim, phidotm, p.Hamiltonian(phim,phidotm), levels=[p.energy()], linewidths=0.8, colors=p.color)
            p.cs.clabel(fontsize=9, inline=False)
        self.points = self.ax2.scatter([None]*len(colors),[None]*len(colors), color=colors)
        self.canvas.mpl_connect('key_press_event', self.keypress)
        self.ani = FuncAnimation(self.fig, animate, blit=True, interval=0, frames=1000) # frames= used only for saving to file
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        self.setCentralWidget(self.canvas)

        self.setWindowIcon(QIcon(LOGO))
        self.setWindowTitle('Plotting Window')
        if geometry: self.restoreGeometry(geometry)
        if state: self.restoreState(state)
        self.show()

    def keypress(self, event):
        """Handler for key presses, registered with matplotlib in the constructor of PlotWindow()"""
        global anim_running, dt

        if event.key == ' ':
            self.ani.event_source.stop() if anim_running else self.ani.event_source.start()
            anim_running = not anim_running
        elif event.key == 'ctrl+q':
            main_exit()
        elif event.key == '+':
            self.ax1.set_xlim([-2,2])
            self.ax1.set_ylim([-2,2])
            self.canvas.draw()
            self.ani._handle_resize()
            self.ani._end_redraw(None)
        elif event.key == '-':
            self.ax1.set_xlim([-1,1])
            self.ax1.set_ylim([-1,1])
            self.canvas.draw()
            self.ani._handle_resize()
            self.ani._end_redraw(None)
        elif event.key == '.':
            dt = abs(dt)
            evolve_pendulums()
            anim_running = False
            self.ani.event_source.start()
        elif event.key == ',':
            dt = -abs(dt)
            evolve_pendulums()
            anim_running = False
            self.ani.event_source.start()
        elif event.key == 'delete':
            if pendulums:
                self.ani.event_source.stop()
                p = pendulums.pop()
                p.free()
                self.ani._handle_resize()
                self.ani._end_redraw(None)
                self.ani.event_source.start()


class ControlWindow(QMainWindow):
    def __init__(self, geometry = None, state = None):
        super().__init__()
        self.create_tabs()
        self.create_menus()
        self.create_time_indicator()
        self.create_statusbar()

        self.playicon = QIcon('icons/play.png')
        self.pauseicon = QIcon('icons/pause.png')
        self.playpausebtn = QToolButton(self, icon=self.playicon)
        self.frameforwardbtn = QToolButton(self, icon=QIcon('icons/forward.png'))
        self.framebackbtn = QToolButton(self, icon=QIcon('icons/rewind'))
        self.playpausebtn.clicked.connect(self.playpause_animation)
        self.frameforwardbtn.clicked.connect(self.frameforward)
        self.framebackbtn.clicked.connect(self.frameback)

        self.setup_layout()

        self.setWindowIcon(QIcon(LOGO))
        self.setWindowTitle('Mathematical Pendulum')
        if geometry: self.restoreGeometry(geometry)
        if state: self.restoreState(state)
        self.show()

    def create_tabs(self):
        """Create tab widgets and set the container to be the central widget"""
        self.tabs = QTabWidget()
        self.controls = QWidget()
        self.pend1 = QWidget()
        self.pend2 = QWidget()
        self.tabs.addTab(self.controls, 'Control &Panel')
        self.tabs.addTab(self.pend1, 'Pendulum &1')
        self.tabs.addTab(self.pend2, 'Pendulum &2')
        self.setCentralWidget(self.tabs)

    def create_menus(self):
        """Create menubar, menu actions and attach them to the menubar"""
        self.menubar = self.menuBar()
        self.menubar.setNativeMenuBar(False)
        self.fileMenu = self.menubar.addMenu('File')
        self.viewMenu = self.menubar.addMenu('View')
        self.helpMenu = self.menubar.addMenu('Help')

        self.exitAction = QAction(QIcon('icons/exit.png'), 'E&xit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Save current state and exit')
        self.exitAction.triggered.connect(main_exit)
        self.fileMenu.addAction(self.exitAction)

        self.tooltipsAction = QAction('Show &tooltips', self, checkable=True, checked=False)
        self.tooltipsAction.setStatusTip('Toggle showing tooltip popups')
        self.tooltipsAction.triggered.connect(self.tooltips_toggle)
        self.viewMenu.addAction(self.tooltipsAction)

        self.aboutAction = QAction(QIcon('icons/about.png'), '&About', self)
        self.aboutAction.setStatusTip('Information about the program')
        self.aboutAction.triggered.connect(self.about)
        self.helpMenu.addAction(self.aboutAction)

        self.aboutQtAction = QAction(QIcon('icons/qt.png'), 'About &Qt', self)
        self.aboutQtAction.setStatusTip('Information about the Qt version')
        self.aboutQtAction.triggered.connect(self.aboutQt)
        self.helpMenu.addAction(self.aboutQtAction)

    def setup_layout(self):
        """Create and connect the layouts for the main control panel"""
        self.controls.grid = QGridLayout()
        self.controls.setLayout(self.controls.grid)
        self.controls.grid.addWidget(self.framebackbtn, 0, 0)
        self.controls.grid.addWidget(self.playpausebtn, 0, 1)
        self.controls.grid.addWidget(self.frameforwardbtn, 0, 2)
        self.controls.grid.addWidget(self.time_label, 1, 0)
        self.controls.grid.addWidget(self.time_lcd, 1, 1)

    def create_time_indicator(self):
        """Create the label and LCD window for the current time"""
        self.time_label = QLabel('Time (s):')
        self.time_lcd = QLCDNumber(self)
        self.time_lcd.setDigitCount(8)
        self.time_lcd.setSegmentStyle(QLCDNumber.Flat)
        self.time_lcd.setStyleSheet('QLCDNumber {background: #8CB398;}')

    def create_statusbar(self):
        """Create status bar and permanent message widget for the status info"""
        self.statusbar = self.statusBar()
        self.statusbar.setStyleSheet('QStatusBar {border-top: 1px outset grey;}')
        self.status_msg = QLabel('Program ready')
        self.statusbar.addPermanentWidget(self.status_msg) # to prevent ovewriting status by other widgets

    def tooltips_toggle(self, state):
        if state:
            self.playpausebtn.setToolTip('Start/pause the animation')
            self.frameforwardbtn.setToolTip('Step forward one time step')
            self.framebackbtn.setToolTip('Step back one time step')
        else:
            self.playpausebtn.setToolTip(None)
            self.frameforwardbtn.setToolTip(None)
            self.framebackbtn.setToolTip(None)

    def playpause_animation(self):
        global anim_running
        self.frameforwardbtn.setEnabled(anim_running)
        self.framebackbtn.setEnabled(anim_running)
        self.status_msg.setText('Animation ' + ('paused' if anim_running else 'running'))
        self.playpausebtn.setIcon(self.playicon if anim_running else self.pauseicon)
        winp.ani.event_source.stop() if anim_running else winp.ani.event_source.start()
        anim_running = not anim_running

    def frameforward(self):
        global anim_running, dt
        dt = abs(dt)
        evolve_pendulums()
        self.status_msg.setText('Animation frame forward')
        anim_running = False
        winp.ani.event_source.start()

    def frameback(self):
        global anim_running, dt
        dt = -abs(dt)
        evolve_pendulums()
        self.status_msg.setText('Animation frame backward')
        anim_running = False
        winp.ani.event_source.start()

    def about(self):
        QMessageBox.about(self, PROGRAM, "<p>Computer simulation of mathematical pendulums with the analysis of the trajectory in the phase space.</p><p>To make a suggestion or to report a bug, please visit our github repository at: <A HREF='https://github.com/tigran123/quantum-infodynamics'>https://github.com/tigran123/quantum-infodynamics</A></p>")

    def aboutQt(self):
        QMessageBox.aboutQt(self, PROGRAM)

def evolve_pendulums():
    global t
    for p in pendulums: p.evolve(t, t+dt)
    t += dt

def animate(i):
    if not anim_running: winp.ani.event_source.stop()
    winc.time_lcd.display('%.3f' % t)
    offsets = []
    for p in pendulums:
        offsets.append([p.phi, p.phidot])
        p.line.set_data(p.position())
        p.energy_text.set_text(r'E/m = %.3f, $\varphi$=%.3f' % (p.energy(), p.phi))
    winp.points.set_offsets(offsets)

    # ignore 0'th frame because animate(0) is called THRICE by matplotlib!
    if i != 0 and anim_running: evolve_pendulums()

    return tuple(p.line for p in pendulums) + tuple(p.energy_text for p in pendulums) + (winp.points,)

pendulums = [Pendulum(phi=np.pi, phidot=0, L=1.0, color='b'),
             Pendulum(phi=0.1*np.pi/2, color='k'),
             Pendulum(phi=0.1*np.pi/2 + 0.01*np.pi/2, color='r'),
             Pendulum(phi=np.pi/2, color='g'),
             Pendulum(phi=np.pi/2 + 0.01*np.pi/2, color='m')]

app = QApplication(sys.argv)
settings = QSettings(COMPANY, PROG)
winp = PlotWindow(geometry = settings.value('plot_geometry'), state = settings.value('plot_windowState'))
winc = ControlWindow(geometry = settings.value('control_geometry'), state = settings.value('control_windowState'))
app.aboutToQuit.connect(main_exit)

# Uncomment this line only for saving animation to file (as per 'frames=' of FuncAnimation() constructor
#winp.fig.tight_layout(); anim_running = True ; winp.ani.save('pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264']) ; sys.exit()

sys.exit(app.exec_())
