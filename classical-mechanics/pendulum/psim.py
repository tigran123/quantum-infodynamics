#!/usr/local/bin/python3

"""
  Mathematical Pendulum Simulator (main program)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  Released under GPLv3, 2017
"""

import sys
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.qt_compat import QtWidgets, QtCore, QtGui
from numpy import pi, mgrid
from pendulum import Pendulum

QMessageBox = QtWidgets.QMessageBox
QMainWindow = QtWidgets.QMainWindow
QWidget = QtWidgets.QWidget
QTabWidget = QtWidgets.QTabWidget
QAction = QtWidgets.QAction
QLCDNumber = QtWidgets.QLCDNumber
QLabel = QtWidgets.QLabel
QApplication = QtWidgets.QApplication
QPushButton = QtWidgets.QPushButton
QToolButton = QtWidgets.QToolButton
QGridLayout = QtWidgets.QGridLayout
Qt = QtCore.Qt
QSettings = QtCore.QSettings
QIcon = QtGui.QIcon

PROGRAM = "Mathematical Pendulum Simulator v0.3 (Qt)"
ICON = "icons/Logo.jpg"

t = 0.0 # global simulation time (has to be the same for all pendulums)
dt = 0.005 # ODE integration fixed timestep
anim_running = False # change to True to start the animation immediately

def main_exit():
    print("main_exit(): Exiting the application")
    sys.exit()

class NamedWindow(QMainWindow):
    """NamedWindow is based on QMainWindow and allow the user to specify the name to be used
       as a key for saving/restoring the window state on exit/starup.
    """
    def __init__(self, name="Unnamed", descr="Unknown"):
        super().__init__()
        self.name = name
        self.descr = descr
        self.settings = QSettings("QuantumInfodynamics.com", "MathematicalPendulum_" + self.name)
        self.setWindowIcon(QIcon(ICON))
        self.setWindowTitle(self.descr)
        if not self.settings.value("geometry") == None:
            self.restoreGeometry(self.settings.value("geometry"))
        if not self.settings.value("windowState") == None:
            self.restoreState(self.settings.value("windowState"))

    def closeEvent(self, event):
        reply = QMessageBox.warning(self, 'Warning', "Are you sure you want to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            settings = QSettings("QuantumInfodynamics.com", "MathematicalPendulum_" + self.name)
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState())
            event.accept()
            print("Exiting the application")
            sys.exit()
        else:
            event.ignore()

class PlotWindow(NamedWindow):
    def __init__(self):
        super().__init__("plot", "Plotting Window")
        self.fig = Figure()
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
        texty = 0.95
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
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        self.setCentralWidget(self.canvas)
        self.show()

class ControlWindow(NamedWindow):
    def __init__(self):
        super().__init__("control", PROGRAM)

        self.tabs = QTabWidget()
        self.controls = QWidget()
        self.pend1 = QWidget()
        self.pend2 = QWidget()
        self.tabs.addTab(self.controls, "Control &Panel")
        self.tabs.addTab(self.pend1, "Pendulum &1")
        self.tabs.addTab(self.pend2, "Pendulum &2")
        self.setCentralWidget(self.tabs)

        self.grid = QGridLayout()
        self.controls.setLayout(self.grid)

        self.menubar = self.menuBar()
        self.menubar.setNativeMenuBar(False)
        self.fileMenu = self.menubar.addMenu('File')
        self.viewMenu = self.menubar.addMenu('View')
        self.helpMenu = self.menubar.addMenu('Help')

        self.exitAction = QAction(QIcon('icons/exit.png'), 'E&xit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip("Exit the program")
        self.exitAction.triggered.connect(app.quit)
        self.fileMenu.addAction(self.exitAction)

        self.tooltipsAction = QAction('Show &tooltips', self, checkable=True, checked=False)
        self.tooltipsAction.setStatusTip("Toggle showing tooltip popups")
        self.tooltips_enabled = False
        self.tooltipsAction.triggered.connect(self.tooltips_toggle)
        self.viewMenu.addAction(self.tooltipsAction)

        self.aboutAction = QAction(QIcon('icons/about.png'), '&About', self)
        self.aboutAction.setStatusTip("Information about the program")
        self.aboutAction.triggered.connect(self.about)
        self.helpMenu.addAction(self.aboutAction)

        self.aboutQtAction = QAction(QIcon('icons/qt.png'), 'About &Qt', self)
        self.aboutQtAction.setStatusTip("Information about the Qt version")
        self.aboutQtAction.triggered.connect(self.aboutQt)
        self.helpMenu.addAction(self.aboutQtAction)

        self.time_label = QLabel('Time (s):')
        self.time_lcd = QLCDNumber(self)
        self.time_lcd.setDigitCount(8)
        self.time_lcd.setSegmentStyle(QLCDNumber.Flat)
        self.time_lcd.setStyleSheet('QLCDNumber {background: #8CB398;}')

        self.statusbar = self.statusBar()
        self.statusbar.setStyleSheet('QStatusBar {border-top: 1px outset grey;}')
        self.status_msg = QLabel('Program ready')
        self.statusbar.addPermanentWidget(self.status_msg) # to prevent ovewriting status by other widgets

        self.playicon = QIcon('icons/play.png')
        self.pauseicon = QIcon('icons/pause.png')
        self.playpausebtn = QToolButton(self, icon=self.playicon)
        self.playpausebtn.setToolTip('Start the animation')
        self.frameforwardbtn = QToolButton(self, icon=QIcon('icons/forward.png'))
        self.frameforwardbtn.setToolTip('Step forward one time step')
        self.framebackbtn = QToolButton(self, icon=QIcon('icons/rewind'))
        self.framebackbtn.setToolTip('Step back one time step')

        self.playpausebtn.clicked.connect(self.playpause_animation)
        self.frameforwardbtn.clicked.connect(self.frameforward)
        self.framebackbtn.clicked.connect(self.frameback)
        self.grid.addWidget(self.framebackbtn, 0, 0)
        self.grid.addWidget(self.playpausebtn, 0, 1)
        self.grid.addWidget(self.frameforwardbtn, 0, 2)
        self.grid.addWidget(self.time_label, 1, 0)
        self.grid.addWidget(self.time_lcd, 1, 1)
        self.show()

    def tooltips_toggle(self, state):
        self.tooltips_enabled = state

    def playpause_animation(self):
        global anim_running
        if anim_running:
            anim_running = False
            self.status_msg.setText("Animation paused")
            self.playpausebtn.setIcon(self.playicon)
            if self.tooltips_enabled: self.playpausebtn.setToolTip('Start the animation')
            winp.ani.event_source.stop()
        else:
            anim_running = True
            self.status_msg.setText("Animation running")
            self.playpausebtn.setIcon(self.pauseicon)
            if self.tooltips_enabled: self.playpausebtn.setToolTip('Pause the animation')
            winp.ani.event_source.start()

    def frameforward(self):
        global anim_running, dt
        dt = abs(dt)
        evolve_pendulums()
        anim_running = False
        self.playpausebtn.setIcon(self.playicon)
        if self.tooltips_enabled: self.playpausebtn.setToolTip('Start the animation')
        self.status_msg.setText("Animation frame forward")
        winp.ani.event_source.start()

    def frameback(self):
        global anim_running, dt
        dt = -abs(dt)
        evolve_pendulums()
        anim_running = False
        self.playpausebtn.setIcon(self.playicon)
        if self.tooltips_enabled: self.playpausebtn.setToolTip('Start the animation')
        self.status_msg.setText("Animation frame backward")
        winp.ani.event_source.start()

    def about(self):
        QMessageBox.about(self, PROGRAM, "<p>Computer simulation of mathematical pendulums with the analysis of the trajectory in the phase space.</p><p>To make a suggestion or to report a bug, please visit our github repository at: <A HREF='https://github.com/tigran123/quantum-infodynamics'>https://github.com/tigran123/quantum-infodynamics</A></p>")

    def aboutQt(self):
        QMessageBox.aboutQt(self, PROGRAM)

def evolve_pendulums():
    global t
    for p in pendulums: p.evolve(t, t+dt)
    t += dt

def keypress(event):
    global anim_running, dt

    if event.key == ' ':
        winp.ani.event_source.stop() if anim_running else winp.ani.event_source.start()
        anim_running = not anim_running
    elif event.key == '+':
        winp.ax1.set_xlim([-2,2])
        winp.ax1.set_ylim([-2,2])
        winp.canvas.draw()
        winp.ani._handle_resize()
        winp.ani._end_redraw(None)
    elif event.key == '-':
        winp.ax1.set_xlim([-1,1])
        winp.ax1.set_ylim([-1,1])
        winp.canvas.draw()
        winp.ani._handle_resize()
        winp.ani._end_redraw(None)
    elif event.key == '.':
        dt = abs(dt)
        evolve_pendulums()
        anim_running = False
        winp.ani.event_source.start()
    elif event.key == ',':
        dt = -abs(dt)
        evolve_pendulums()
        anim_running = False
        winp.ani.event_source.start()
    elif event.key == "delete":
        if pendulums:
            winp.ani.event_source.stop()
            p = pendulums.pop()
            p.free()
            winp.ani._handle_resize()
            winp.ani._end_redraw(None)
            winp.ani.event_source.start()

def animate(i):
    if not anim_running: winp.ani.event_source.stop()
    winc.time_lcd.display('%.3f' % t)
    offsets = []
    for p in pendulums:
        offsets.append([p.phi, p.phidot])
        p.line.set_data(p.position())
        p.energy_text.set_text(r'E = %.3f J, $\varphi$=%.3f' % (p.energy(), p.phi))
    winp.points.set_offsets(offsets)

    # ignore 0'th frame because animate(0) is called THRICE by matplotlib!
    if i != 0 and anim_running: evolve_pendulums()

    return tuple(p.line for p in pendulums) + tuple(p.energy_text for p in pendulums) + (winp.points,)

pendulums = [Pendulum(phi=pi, phidot=3, L=1.0, color='b'),
             Pendulum(phi=pi, L=0.9, color='r'),
             Pendulum(phi=pi/3, L=0.6, color='g'),
             Pendulum(phi=0.9*pi/3, L=0.6, color='m')]

app = QApplication(sys.argv)
app.aboutToQuit.connect(main_exit)
winp = PlotWindow()
winc = ControlWindow()

#winp.fig.tight_layout(); anim_running = True ; winp.ani.save('pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
sys.exit(app.exec_())
