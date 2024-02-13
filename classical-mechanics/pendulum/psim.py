#!/usr/bin/env python3
"""
  Mathematical Pendulum Simulator (main program)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  Released under GPLv3, 2017
"""

import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from time import time
from numpy import pi, mgrid
from qtapi import *
from pendulum import Pendulum

COMPANY = 'Bibles.org.uk'
PROGRAM = 'Mathematical Pendulum Simulator v1.0'
PROG = 'MathematicalPendulumSimulator'
LOGO = 'icons/Logo.jpg'

t = 0.0 # global simulation time (the same for all pendulums)
dt = 0.005 # initial ODE integration timestep
dtlim = 1.0 #  -dtlim <= dt <= +dtlim
anim_running = False # if True start the animation immediately
anim_save = True # set to True to save animation to disk
save_frames=100 # number of frames to save, fps set in ani.save() call

# for calculating FPS in animate()
frames = 0 ; fps = 0 ; start_time = time()

def update_dt(value):
    global dt
    dt = dtlim*value/1000
    winc.label_dt.setText('Δt = %.4f s' % dt)

def main_exit():
    global settings
    settings.setValue('plot_geometry', winp.saveGeometry())
    settings.setValue('plot_windowState', winp.saveState())
    settings.setValue('control_geometry', winc.saveGeometry())
    settings.setValue('control_windowState', winc.saveState())
    del settings # to force the writing of settings to storage
    print('Exiting the program')
    sys.exit()

def single_step(dir):
    global dt, anim_running
    dt = abs(dt) if dir == 'forward' else -abs(dt)
    winc.status_msg.setText('Step ' + dir)
    anim_running = False
    winp.ani.event_source.start()

def playpause():
    global anim_running
    #winc.frameforwardbtn.setEnabled(anim_running)
    #winc.framebackbtn.setEnabled(anim_running)
    winc.status_msg.setText('Animation ' + ('paused' if anim_running else 'running'))
    winc.playpausebtn.setIcon(winc.playicon if anim_running else winc.pauseicon)
    winp.ani.event_source.stop() if anim_running else winp.ani.event_source.start()
    anim_running = not anim_running

class PlotWindow(QMainWindow):
    def __init__(self, geometry = None, state = None):
        super().__init__()
        self.fig = Figure(figsize=(19.2,10.8))
        self.canvas = FigureCanvas(self.fig)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.ax1.grid()
        self.ax2.grid()
        self.ax1.set_aspect('equal')
        self.ax1.set_title(PROGRAM)
        self.ax1.set_xlabel('$x$ (m)')
        self.ax1.set_ylabel('$y$ (m)')
        space_range = 1.5
        self.ax1.set_xlim([-space_range,space_range])
        self.ax1.set_ylim([-space_range,space_range])
        self.ax2.set_title('Phase Space Trajectories')
        self.ax2.set_xlabel(r'$\varphi$ (rad)')
        self.ax2.set_ylabel(r'$\dot{\varphi}$ (rad/s)')
        self.phi_range = 1.1*pi
        self.phi_points = 200
        self.phidot_range = 8.0
        self.phidot_points = 200
        self.ax2.set_xlim([-self.phi_range, self.phi_range])
        self.ax2.set_ylim([-self.phidot_range, self.phidot_range])
        phim,phidotm = mgrid[-self.phi_range:self.phi_range:self.phi_points*1j,-self.phidot_range:self.phidot_range:self.phidot_points*1j]
        colors = []
        self.fps_text = self.ax1.text(0.02, 0.05, '', transform=self.ax1.transAxes)
        self.time_text = self.ax1.text(0.02, 0.1, '', transform=self.ax1.transAxes)
        texty = 0.95
        for p in pendulums:
            colors.append(p.color)
            p.line, = self.ax1.plot([], [], 'o-', lw=2, color=p.color)
            p.energy_text = self.ax1.text(0.02, texty, '', transform=self.ax1.transAxes, color=p.color)
            texty -= 0.05
            p.cs = self.ax2.contour(phim, phidotm, p.Hamiltonian(phim,phidotm), levels=[p.energy()], linewidths=0.8, colors=p.color)
        self.points = self.ax2.scatter([None]*len(colors),[None]*len(colors), color=colors)
        self.canvas.mpl_connect('key_press_event', self.keypress)
        self.ani = FuncAnimation(self.fig, animate, blit=True, interval=0, frames=save_frames) # frames= used only for saving to file
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        self.setCentralWidget(self.canvas)

        self.setWindowIcon(QIcon(LOGO))
        self.setWindowTitle('Plotting Window')
        if geometry: self.restoreGeometry(geometry)
        if state: self.restoreState(state)
        self.fig.tight_layout();
        self.show()

    def keypress(self, event):
        """Handler for key presses, registered with matplotlib in the constructor of PlotWindow()"""
        if event.key == ' ':
            playpause()
        elif event.key == 'ctrl+q':
            main_exit()
        elif event.key == '+':
            self.ax1.set_xlim([-2,2])
            self.ax1.set_ylim([-2,2])
            self.canvas.draw()
            self.ani._end_redraw(None)
        elif event.key == '-':
            self.ax1.set_xlim([-1,1])
            self.ax1.set_ylim([-1,1])
            self.canvas.draw()
            self.ani._end_redraw(None)
        elif event.key == '.':
            single_step('forward')
        elif event.key == ',':
            single_step('backward')
        elif event.key == 'delete':
            if pendulums:
                self.ani.event_source.stop()
                p = pendulums.pop()
                p.line.remove()
                del(p.line)
                p.energy_text.remove()
                del(p.energy_text)
                while True:
                    try: p.cs.pop_label()
                    except IndexError: break
                for c in p.cs.collections: c.remove()
                self.ani._end_redraw(None)
                self.ani.event_source.start()

class ControlWindow(QMainWindow):
    def __init__(self, geometry = None, state = None):
        super().__init__()
        self.create_tabs()
        self.create_menus()
        self.create_time_indicator()
        self.create_statusbar()
        self.create_slider()

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
        self.setWindowTitle(PROGRAM)
        if geometry: self.restoreGeometry(geometry)
        if state: self.restoreState(state)
        self.show()

    def create_tabs(self):
        """Create tab widgets and set the container to be the central widget"""
        self.tabs = QTabWidget()
        self.controls = QWidget()
        self.tabs.addTab(self.controls, 'Control &Panel')
        #self.pendtabs = []
        #i = 0
        #for p in pendulums:
        #    self.pendtabs.append(QWidget())
        #    i += 1
        #    self.tabs.addTab(self.pendtabs[-1], 'Pendulum &%d' % (i))
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

    def create_slider(self):
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(-1000, 1000)
        self.slider.setPageStep(5)
        self.slider.valueChanged.connect(update_dt)
        self.label_dt = QLabel('Δt = %.4f s' % dt)
        self.label_dt.setAlignment(Qt.AlignLeft)
        self.label_dt.setMinimumWidth(120)

    def setup_layout(self):
        """Create and connect the layouts for the main control panel"""
        self.controls.grid = QGridLayout()
        self.controls.setLayout(self.controls.grid)
        self.controls.grid.addWidget(self.framebackbtn, 0, 0)
        self.controls.grid.addWidget(self.playpausebtn, 0, 1)
        self.controls.grid.addWidget(self.frameforwardbtn, 0, 2)
        self.controls.grid.addWidget(self.time_label, 0, 3)
        self.controls.grid.addWidget(self.time_lcd, 0, 4)
        self.controls.grid.addWidget(self.label_dt, 1, 0, 1, 3)
        self.controls.grid.addWidget(self.slider, 1, 3, 1, 2)

    def create_time_indicator(self):
        """Create the label and LCD window for the current time"""
        self.time_label = QLabel('Time (s):')
        self.time_lcd = QLCDNumber(self)
        self.time_lcd.setDigitCount(12)
        self.time_lcd.setSegmentStyle(QLCDNumber.Flat)
        self.time_lcd.setStyleSheet('QLCDNumber {background: #8CB398;}')

    def create_statusbar(self):
        """Create status bar and permanent message widget for the status info"""
        self.statusbar = self.statusBar()
        self.statusbar.setStyleSheet('QStatusBar {border-top: 1px outset grey;}')
        self.status_msg = QLabel('Program ready')
        self.statusbar.addPermanentWidget(self.status_msg) # prevent ovewriting status by other widgets

    def tooltips_toggle(self, state):
        if state:
            self.playpausebtn.setToolTip('Start/pause the animation')
            self.frameforwardbtn.setToolTip('Step forward one time step')
            self.framebackbtn.setToolTip('Step back one time step')
            self.time_lcd.setToolTip('Simulation time in seconds')
            self.label_dt.setToolTip('Current value of ODE integration time step Δt in seconds')
            self.slider.setToolTip('Control ODE integration time step Δt')
        else:
            self.playpausebtn.setToolTip(None)
            self.frameforwardbtn.setToolTip(None)
            self.framebackbtn.setToolTip(None)
            self.time_lcd.setToolTip(None)
            self.label_dt.setToolTip(None)
            self.slider.setToolTip(None)

    def playpause_animation(self):
        playpause()

    def frameforward(self):
        single_step('forward')

    def frameback(self):
        single_step('backward')

    def about(self):
        QMessageBox.about(self, PROGRAM, "<p>Computer simulation of mathematical pendulums in the the phase space.</p><p>To report a bug, please visit our github repository at: <A HREF='https://github.com/tigran123/quantum-infodynamics'>https://github.com/tigran123/quantum-infodynamics</A></p>")

    def aboutQt(self):
        QMessageBox.aboutQt(self, PROGRAM)

def evolve_pendulums():
    global t
    for p in pendulums: p.evolve(t, t+dt)
    t += dt

def animate(i):
    global frames, fps, start_time
    if i == 0: return tuple(p.line for p in pendulums) # ignore 0'th frame as animate(0) is called THRICE by matplotlib
    if not anim_running: winp.ani.event_source.stop()
    evolve_pendulums()
    winc.time_lcd.display('%.3f' % t)

    if not anim_save: # don't update or show FPS when saving animation to file
        frames += 1
        now = time()
        deltaT = now - start_time
        if deltaT > 3: # update FPS every 3 seconds
            winp.fps_text.set_text("FPS: %.1f" % float(frames/deltaT))
            start_time = now
            frames = 0

    offsets = []
    for p in pendulums:
        offsets.append([p.phi, p.phidot])
        p.line.set_data(p.position())
        p.energy_text.set_text(r'E=%.2f J/kg, $\varphi$=%.1f°, $\dot{\varphi}$=%.1f rad/s' % (p.energy(), p.phi*180/pi, p.phidot))
    winp.time_text.set_text("Time t=%.3f s" % t)
    winp.points.set_offsets(offsets)

    return tuple(p.line for p in pendulums) + (winp.fps_text,) + (winp.time_text,) + tuple(p.energy_text for p in pendulums) + (winp.points,)

pendulums = [Pendulum(phi=pi, phidot=0, L=1.0, color='b'),
             Pendulum(phi=0.3*pi/2, color='k'),
             Pendulum(phi=0.3*pi/2 + 0.01*pi/2, color='r'),
             Pendulum(phi=pi/2, phidot=5.1, color='g'),
             Pendulum(phi=pi/2, color='m')]

app = QApplication(sys.argv)
settings = QSettings(COMPANY, PROG)
winp = PlotWindow(geometry = settings.value('plot_geometry'), state = settings.value('plot_windowState'))
winc = ControlWindow(geometry = settings.value('control_geometry'), state = settings.value('control_windowState'))
app.aboutToQuit.connect(main_exit)

if anim_save:
    anim_running = True
    winp.ani.save('pendulum.mp4', dpi=150, fps=30, extra_args=['-vcodec', 'libx264'])
    sys.exit() # bypass our exit handler main_exit()
else:
	sys.exit(app.exec_())
