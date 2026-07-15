#!.venv/bin/python
"""
  Mathematical Pendulum Simulator (main program)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  Released under GPLv3, 2017
"""

import sys
from os import environ
# Qt6 runs natively on Wayland, but Wayland gives applications no way to position
# their own windows, so the saved window positions could never be restored there.
# Prefer X11 (XWayland), where positioning works; override with QT_QPA_PLATFORM=wayland.
environ.setdefault('QT_QPA_PLATFORM', 'xcb')
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.animation import FuncAnimation
from subprocess import Popen, PIPE, DEVNULL
from matplotlib.figure import Figure
from matplotlib.colors import to_hex
from time import perf_counter, strftime
from gc import collect
from numpy import pi, mgrid, empty
from qtapi import *
from pendulum import Pendulum

COMPANY = 'Bibles.org.uk'
PROGRAM = 'Mathematical Pendulum Simulator v1.0'
PROG = 'MathematicalPendulumSimulator'
LOGO = 'icons/Logo.jpg'

t = 0.0 # global simulation time (the same for all pendulums)
dt = 0.005 # initial ODE integration timestep
dtlim = 1.0 #  -dtlim <= dt <= +dtlim, adjustable on the Control tab
anim_running = False # if True start the animation immediately
single_step = False # set by step_forward()/step_backward(): the next frame may advance the simulation while paused

# for calculating FPS in pw_animate()
frames = 0 ; start_time = perf_counter()

def update_dt(value):
    global dt
    dt = value

def update_dtlim(value):
    global dtlim
    dtlim = value
    cw.dt_spin.setRange(-dtlim, dtlim) # clamps dt (via its valueChanged) if now out of range

def main_exit():
    """Save the state before quitting; connected to app.aboutToQuit, when the windows
       are still alive. NB: quitting is always via app.quit(), never sys.exit() from
       inside a Qt slot: raising SystemExit there finalises the interpreter (destroying
       the QApplication) while the C++ event loop is still on the stack below, which
       segfaults in sip depending on the destruction order."""
    pw.ani.event_source.stop() # no more animation callbacks during teardown
    if pw.writer: # finalise the video file of an unfinished recording
        try: pw.stop_recording()
        except Exception: pass
    settings.setValue('plot_geometry', pw.saveGeometry())
    settings.setValue('plot_windowState', pw.saveState())
    settings.setValue('control_geometry', cw.saveGeometry())
    settings.setValue('control_windowState', cw.saveState())
    settings.setValue('show_text', cw.textcheck.isChecked())
    settings.sync() # force the writing of settings to storage
    print('Exiting the program')

def stop_recording_ui():
    """Finalise the current recording and reflect it in the control window"""
    (nframes, filename) = pw.stop_recording()
    cw.cw_record_reset('Saved %d frames to %s' % (nframes, filename))

def commit_dt_edits():
    """Apply a Δt or |Δt|-limit value still being typed. Keyboard tracking is off on these
       spinboxes, so a typed value is normally applied on Enter/focus-out only — but the
       transport buttons never take keyboard focus, so clicking them commits any pending
       edit here, or the animation would run with the old Δt while the field shows the new one."""
    cw.dtlim_spin.interpretText() # the limit first: a new limit must be in force before Δt is parsed
    cw.dt_spin.interpretText()

def step_forward():
    global dt, anim_running, single_step
    commit_dt_edits()
    recorded = bool(pw.writer)
    if recorded: stop_recording_ui() # stopping the animation finishes the recording
    dt = abs(dt)
    cw.dt_spin.setValue(dt)
    if not recorded: cw.status_msg.setText('Step forward') # keep the 'Saved ...' message visible
    anim_running = False
    single_step = True
    pw.ani.resume()

def step_backward():
    global dt, anim_running, single_step
    commit_dt_edits()
    recorded = bool(pw.writer)
    if recorded: stop_recording_ui() # stopping the animation finishes the recording
    dt = -abs(dt)
    cw.dt_spin.setValue(dt)
    if not recorded: cw.status_msg.setText('Step backward') # keep the 'Saved ...' message visible
    anim_running = False
    single_step = True
    pw.ani.resume()

def playpause():
    global anim_running, frames, start_time
    commit_dt_edits()
    recorded = anim_running and pw.writer
    if recorded: stop_recording_ui() # stopping the animation finishes the recording
    if not recorded: cw.status_msg.setText('Animation ' + ('paused' if anim_running else 'running')) # keep the 'Saved ...' message visible
    cw.playpausebtn.setIcon(cw.playicon if anim_running else cw.pauseicon)
    if anim_running:
        pw.ani.pause()
    else:
        (frames, start_time) = (0, perf_counter()) # a fresh FPS measurement window
        pw.ani.resume()
    anim_running = not anim_running

def reset_time():
    """Reset the simulation time to zero. The states of the pendulums are unaffected:
       the dynamics is autonomous, so t is only a label on the current state."""
    global t
    t = 0.0
    cw.time_lcd.display('%.3f' % t)
    if pw.show_text: pw.time_text.set_text('Time t=%.3f s' % t)
    if not anim_running: pw.refresh() # repaint the new time now, no frame will do it for us
    cw.status_msg.setText('Time reset to zero')

PALETTE = ['b', 'k', 'r', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']

def next_color():
    """Return the first palette colour not yet used by any pendulum"""
    used = {p.color for p in pendulums}
    for c in PALETTE:
        if c not in used: return c
    return 'k'

def color_icon(color):
    """Return a small solid rectangle of the given colour, used as the icon of a
       pendulum's tab so the user can see which tab controls which pendulum"""
    pm = QPixmap(12, 12)
    pm.fill(QColor(to_hex(color)))
    return QIcon(pm)

def state_str(p):
    """Return the state of pendulum p formatted for its state text artist.
       Plain text, NOT mathtext: mathtext would be re-parsed on every frame
       (the string changes every frame), costing ~half of the total frame time."""
    return 'φ=%.3f°, φ̇=%.3f rad/s' % (p.phi*180/pi, p.phidot)

def add_pendulum(p):
    """Add pendulum p to the simulation and (if present) the control window"""
    pw.ani.pause()
    pendulums.append(p)
    pw.add_artists(p)
    pw.restack_texts()
    pw.update_scatter()
    pw.update_limits()
    pw.refresh()
    cw.add_tab(p)
    pw.ani.resume()

def delete_pendulum(p):
    """Remove pendulum p from the simulation and (if present) the control window"""
    pw.ani.pause()
    pendulums.remove(p)
    pw.remove_artists(p)
    pw.restack_texts()
    pw.update_scatter()
    pw.update_limits()
    pw.refresh()
    cw.remove_tab(p)
    pw.ani.resume()

CAPTION_ACTIVE = '#1e50c8'   # caption bar colour of the active window (blue)
CAPTION_INACTIVE = '#2e8b57' # caption bar colour of an inactive window (green)

class CaptionBar(QWidget):
    """Custom window caption bar with our own colours (the windows are frameless,
       so the desktop theme has no say in how our caption bars look)"""
    def __init__(self, win, title):
        super().__init__(win)
        self.win = win
        self.setAutoFillBackground(True)
        self.title = QLabel(title)
        self.title.setStyleSheet('color: white; font-weight: bold; background: transparent;')
        btnstyle = """QToolButton {color: white; background: transparent; border: none; font-weight: bold;}
                      QToolButton:hover {background: rgba(255,255,255,0.3);}"""
        self.minbtn = QToolButton(self)
        self.minbtn.setText('–')
        self.minbtn.clicked.connect(win.showMinimized)
        self.maxbtn = QToolButton(self)
        self.maxbtn.setText('□')
        self.maxbtn.clicked.connect(self.toggle_maximized)
        self.closebtn = QToolButton(self)
        self.closebtn.setText('✕')
        self.closebtn.clicked.connect(win.close)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 4, 2)
        layout.addWidget(self.title)
        layout.addStretch(1)
        for b in (self.minbtn, self.maxbtn, self.closebtn):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setStyleSheet(btnstyle)
            layout.addWidget(b)
        self.set_active(True)

    def set_active(self, active):
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(CAPTION_ACTIVE if active else CAPTION_INACTIVE))
        self.setPalette(pal)

    def toggle_maximized(self):
        self.win.showNormal() if self.win.isMaximized() else self.win.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.win.windowHandle().startSystemMove()

    def mouseDoubleClickEvent(self, event):
        self.toggle_maximized()

RESIZE_MARGIN = 6 # width (in pixels) of the edge strip of a frameless window that resizes it

class CaptionedWindow(QMainWindow):
    """Frameless QMainWindow with a CaptionBar, resizeable by dragging any edge or corner;
       subclasses call setup_caption() before show()"""
    def setup_caption(self, title, below = None):
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        # the margin strip around the content belongs to the window itself and acts
        # as the resize handle (mouse events over child widgets never reach us)
        self.setContentsMargins(RESIZE_MARGIN, RESIZE_MARGIN, RESIZE_MARGIN, RESIZE_MARGIN)
        self.setMouseTracking(True)
        self.caption = CaptionBar(self, title)
        if below:
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(0)
            vbox.addWidget(self.caption)
            vbox.addWidget(below)
            self.setMenuWidget(container)
        else:
            self.setMenuWidget(self.caption)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.ActivationChange and hasattr(self, 'caption'):
            self.caption.set_active(self.isActiveWindow())
        super().changeEvent(event)

    def edges_at(self, pos):
        """Return the combination of window edges the point is within RESIZE_MARGIN of"""
        edges = Qt.Edge(0)
        r = self.rect()
        if pos.x() <= RESIZE_MARGIN: edges |= Qt.Edge.LeftEdge
        if pos.x() >= r.width() - RESIZE_MARGIN: edges |= Qt.Edge.RightEdge
        if pos.y() <= RESIZE_MARGIN: edges |= Qt.Edge.TopEdge
        if pos.y() >= r.height() - RESIZE_MARGIN: edges |= Qt.Edge.BottomEdge
        return edges

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.isMaximized():
            edges = self.edges_at(event.position().toPoint())
            if edges:
                self.windowHandle().startSystemResize(edges)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Give cursor feedback over the resize strip (requires mouse tracking)"""
        E = Qt.Edge
        edges = self.edges_at(event.position().toPoint()) if not self.isMaximized() else E(0)
        if edges in (E.LeftEdge, E.RightEdge):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif edges in (E.TopEdge, E.BottomEdge):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif edges in (E.LeftEdge|E.TopEdge, E.RightEdge|E.BottomEdge):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif edges in (E.LeftEdge|E.BottomEdge, E.RightEdge|E.TopEdge):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)

class CursorReset(QObject):
    """Application-wide event filter (installed on the QApplication) resetting the
       resize-feedback cursor of a CaptionedWindow as soon as the pointer enters any
       of its child widgets. The window itself receives no mouse-move events while
       the pointer is over a child, so without this the double-arrow cursor set over
       the resize strip would stay stuck over the whole window."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Enter and isinstance(obj, QWidget):
            win = obj.window()
            if isinstance(win, CaptionedWindow) and obj is not win: win.unsetCursor()
        return False

class RecordableFuncAnimation(FuncAnimation):
    """FuncAnimation which pipes every frame composited on the screen to the recording
       ffmpeg process (if one is active) of the PlotWindow owning it. NB: this overrides
       the private _post_draw() because the frame is complete in the canvas buffer only
       after it has run and matplotlib offers no public post-compositing hook; blitted
       frames do not emit draw_event."""
    def __init__(self, window, *args, **kwargs):
        self.window = window # must be set before super().__init__(), which calls _post_draw()
        super().__init__(*args, **kwargs)

    def _post_draw(self, framedata, blit):
        super()._post_draw(framedata, blit)
        if self.window.writer and framedata is not None: self.window.grab_frame()

    def pause(self):
        """Stop the animation timer. Overrides Animation.pause() WITHOUT its flipping of
           the drawn artists to animated=False: our artists must stay animated forever,
           or any full draw while paused (refresh(), a resize) would bake them into the
           blitting background, leaving permanent ghosts on the plot."""
        self.event_source.stop()

    def resume(self):
        """Restart the animation timer (counterpart of pause(), same NB applies)"""
        self.event_source.start()

class PlotWindow(CaptionedWindow):
    def __init__(self, geometry = None, state = None):
        super().__init__()
        self.writer = None # ffmpeg process active while the animation is being recorded
        self.show_text = True # render the state and time texts in the plot
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
        self.update_limits()
        self.ax2.set_title('Phase Space Trajectories')
        self.ax2.set_xlabel(r'$\varphi$ (rad)')
        self.ax2.set_ylabel(r'$\dot{\varphi}$ (rad/s)')
        self.phi_range = 1.1*pi
        self.phi_points = 200
        self.phidot_range = 8.0
        self.phidot_points = 200
        self.ax2.set_xlim([-self.phi_range, self.phi_range])
        self.ax2.set_ylim([-self.phidot_range, self.phidot_range])
        self.phim,self.phidotm = mgrid[-self.phi_range:self.phi_range:self.phi_points*1j,-self.phidot_range:self.phidot_range:self.phidot_points*1j]
        self.time_text = self.ax1.text(0.02, 0.05, 'Time t=%.3f s' % t, transform=self.ax1.transAxes, animated=True)
        for p in pendulums: self.add_artists(p)
        self.restack_texts()
        self.points = self.ax2.scatter([], [], animated=True)
        self.update_scatter()
        self.canvas.mpl_connect('key_press_event', self.pw_keypress)

        def init_animate(): return tuple(p.line for p in pendulums)
        def animate(i):
            global frames, start_time, t, single_step
            artists = tuple(p.line for p in pendulums) + (pw.time_text,) + tuple(p.state_text for p in pendulums) + (pw.points,)
            if i == 0: return artists
            if not anim_running:
                pw.ani.pause()
                # while paused the timer may still fire once: at startup, and whenever
                # refresh() runs (its _end_redraw() restarts the event source). Unless a
                # single step was explicitly requested, the simulation must not advance.
                if not single_step: return artists
            # consume the step request; also clear one left stale by a step request
            # that was superseded by free running before its frame could fire
            single_step = False
            # evolve first, then render: everything below shows the state at the new time.
            # Evolving after rendering would leave the state one Δt ahead of the display,
            # revealed only by the NEXT frame: the first step after a Δt change would
            # visibly advance by the old Δt, and anything reading the state while paused
            # (pendulum tabs, Stop) would disagree with the plot.
            for p in pendulums:
                if p.live: p.evolve(t, t+dt)
            t += dt
            cw.time_lcd.display('%.3f' % t)
            cw.sync_current_tab()
            frames += 1
            now = perf_counter()
            deltaT = now - start_time
            if deltaT > 3: # update FPS every 3 seconds
                cw.fps_label.setText('FPS: %.1f' % (frames/deltaT))
                start_time = now
                frames = 0
            for p in pendulums:
                p.line.set_data(p.position())
            if pw.show_text:
                for p in pendulums: p.state_text.set_text(state_str(p))
                pw.time_text.set_text("Time t=%.3f s" % t)
            pw.update_scatter()
            return artists

        self.ani = RecordableFuncAnimation(self, self.fig, animate, init_func=init_animate, blit=True, interval=0, cache_frame_data=False)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()
        self.setCentralWidget(self.canvas)
        self.setup_caption('Plotting Window')
        self.statusBar() # gives the frameless window a size grip for resizing

        self.setWindowIcon(QIcon(LOGO))
        self.setWindowTitle('Plotting Window')
        if geometry: self.restoreGeometry(geometry)
        if state: self.restoreState(state)
        self.fig.tight_layout();
        self.show()

    def make_contour(self, p):
        """Create the contour of the energy level of pendulum p in the phase space.
           NB: matplotlib draws negative levels of monochrome contours dashed by default,
           and the energy of a pendulum near the bottom is negative, hence linestyles."""
        return self.ax2.contour(self.phim, self.phidotm, p.Hamiltonian(self.phim, self.phidotm), levels=[p.energy()], linewidths=0.8, colors=p.color, linestyles='solid')

    def add_artists(self, p):
        """Create all the artists of pendulum p: the bob line, the state text and the energy contour.
           The artists redrawn on every frame are 'animated', so that they are never baked into
           the background captured for blitting (which would leave ghosts on the plot)."""
        p.line, = self.ax1.plot([], [], 'o-', lw=2, color=p.color, animated=True)
        p.line.set_data(p.position())
        p.state_text = self.ax1.text(0.02, 0.95, state_str(p), transform=self.ax1.transAxes, color=p.color, animated=True, visible=self.show_text)
        p.cs = self.make_contour(p)

    def remove_artists(self, p):
        """Remove all the artists of pendulum p from the axes"""
        p.line.remove()
        p.state_text.remove()
        p.cs.remove()

    def update_artists(self, p):
        """Reflect the changed state/colour of pendulum p in its artists"""
        p.line.set_color(p.color)
        p.line.set_data(p.position())
        p.state_text.set_color(p.color)
        p.state_text.set_text(state_str(p))
        p.cs.remove()
        p.cs = self.make_contour(p)
        self.update_scatter()
        self.update_limits()
        self.refresh()

    def restack_texts(self):
        """Stack the state texts of all pendulums in the top left corner of the plot"""
        texty = 0.95
        for p in pendulums:
            p.state_text.set_position((0.02, texty))
            texty -= 0.05

    def update_scatter(self):
        """Sync the phase space points with the current states of the pendulums"""
        if pendulums:
            self.points.set_offsets([[p.phi, p.phidot] for p in pendulums])
            self.points.set_color([p.color for p in pendulums])
        else:
            self.points.set_offsets(empty((0,2)))

    def set_show_text(self, show):
        """Show/hide the state and time texts in the plot: superfluous interactively
           (the control window shows the same data, and not rendering them makes the
           animation faster), but valuable in a recording."""
        self.show_text = show
        if show: # bring the texts up to date: they are not updated while hidden
            for p in pendulums: p.state_text.set_text(state_str(p))
            self.time_text.set_text('Time t=%.3f s' % t)
        self.time_text.set_visible(show)
        for p in pendulums: p.state_text.set_visible(show)
        self.refresh()

    def update_limits(self):
        """Rescale the space plot to fit the longest pendulum entirely"""
        r = max(1.5, 1.25*max((p.L for p in pendulums), default=0))
        self.ax1.set_xlim([-r, r])
        self.ax1.set_ylim([-r, r])

    def refresh(self):
        """Redraw the canvas and recapture the blitting background of the animation.
           Without invalidating ani._blit_cache the stale background (with removed
           artists still on it) would be restored on every subsequent frame.
           NB: ani._blit_cache and ani._end_redraw() are private matplotlib APIs,
           kept in this one method only."""
        self.ani._blit_cache.clear()
        self.canvas.draw()
        self.ani._end_redraw(None)

    def start_recording(self, filename):
        """Start recording the animation into a video file. Every frame composited on
           the screen is piped raw to ffmpeg, so the video contains exactly what is
           seen, at the current window resolution, with no extra rendering cost."""
        (w, h) = self.canvas.get_width_height(physical=True)
        proc = Popen(['ffmpeg', '-loglevel', 'quiet', '-y',
                      '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', '%dx%d' % (w, h), '-r', '30', '-i', '-',
                      '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2', # h264 requires even dimensions
                      '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', filename],
                     stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
        (self.rec_filename, self.rec_frames, self.rec_size, self.writer) = (filename, 0, (w, h), proc)

    def grab_frame(self):
        """Pipe the current contents of the canvas to the recording ffmpeg process"""
        try:
            if self.canvas.get_width_height(physical=True) != self.rec_size:
                raise RuntimeError('the window was resized')
            self.writer.stdin.write(self.canvas.buffer_rgba())
            self.rec_frames += 1
        except Exception as err:
            (proc, self.writer) = (self.writer, None)
            try:
                proc.stdin.close()
                proc.wait()
            except Exception: pass
            cw.cw_record_reset('Recording stopped: %s' % err)

    def stop_recording(self):
        """Finalise the video file and return (number of frames, filename)"""
        (proc, self.writer) = (self.writer, None)
        proc.stdin.close()
        proc.wait()
        return (self.rec_frames, self.rec_filename)

    def pw_keypress(self, event):
        """Handler for key presses, registered with matplotlib in the constructor of PlotWindow()"""
        if event.key == ' ':
            playpause()
        elif event.key == 'ctrl+q':
            app.quit()
        elif event.key == '+':
            self.ax1.set_xlim([-2,2])
            self.ax1.set_ylim([-2,2])
            self.refresh()
        elif event.key == '-':
            self.ax1.set_xlim([-1,1])
            self.ax1.set_ylim([-1,1])
            self.refresh()
        elif event.key == '.':
            step_forward()
        elif event.key == ',':
            step_backward()
        elif event.key == 'delete':
            if pendulums: delete_pendulum(pendulums[-1])

RAD2DEG = 180/pi
PHIDOT_MAX = 20.0 # maximal |dφ/dt| settable in a pendulum tab, in rad/s

class PendulumTab(QWidget):
    """Tab of the control window with the initial conditions and controls of a single pendulum"""
    def __init__(self, p, cw):
        super().__init__()
        self.p = p
        self.cw = cw
        self.degrees = False # angle units of this tab's φ and φ̇ fields: radians (False) or degrees (True)
        self.color = to_hex(p.color)

        self.phi_spin = QDoubleSpinBox(decimals=3)
        self.phidot_spin = QDoubleSpinBox(decimals=3)
        self.set_unit_ranges()
        self.set_phi(p.phi)
        self.set_phidot(p.phidot)
        self.L_spin = QDoubleSpinBox(decimals=3, minimum=0.01, maximum=10.0, singleStep=0.1, suffix=' m')
        self.L_spin.setValue(p.L)
        self.colorbtn = QPushButton()
        self.colorbtn.setFixedWidth(60)
        self.colorbtn.setStyleSheet('background-color: %s;' % self.color)
        self.colorbtn.clicked.connect(self.choose_color)

        self.applybtn = QPushButton('&Apply')
        self.applybtn.clicked.connect(self.apply)
        self.startbtn = QPushButton('&Stop' if p.live else '&Start')
        self.startbtn.clicked.connect(self.startstop)
        self.set_editable(not p.live)
        self.delbtn = QPushButton('De&lete')
        self.delbtn.clicked.connect(self.delete)
        # the units toggle is per-tab: it changes only how this tab displays φ and φ̇
        # (the plot window always uses radians)
        self.radbtn = QRadioButton('&Radians')
        self.degbtn = QRadioButton('&Degrees')
        self.radbtn.setChecked(True)
        self.unitsgroup = QButtonGroup(self)
        self.unitsgroup.addButton(self.radbtn)
        self.unitsgroup.addButton(self.degbtn)
        self.degbtn.toggled.connect(self.set_units)
        # the buttons must never take keyboard focus, or pressing Space would 'click'
        # whichever happened to be focused; they are operated by mouse or Alt+mnemonic
        for b in (self.applybtn, self.startbtn, self.delbtn, self.colorbtn, self.radbtn, self.degbtn):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        fields = QHBoxLayout()
        for (label, spin) in (('φ', self.phi_spin), ('φ̇', self.phidot_spin), ('L', self.L_spin)):
            fields.addWidget(QLabel(label))
            fields.addWidget(spin)
        fields.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.applybtn)
        buttons.addWidget(self.startbtn)
        buttons.addWidget(self.delbtn)
        buttons.addWidget(self.colorbtn)
        buttons.addWidget(self.radbtn)
        buttons.addWidget(self.degbtn)
        buttons.addStretch(1)
        vbox = QVBoxLayout()
        vbox.addLayout(fields)
        vbox.addLayout(buttons)
        vbox.addStretch(1)
        self.setLayout(vbox)

    def set_unit_ranges(self):
        """Set the ranges/steps/suffixes of the φ and φ̇ fields according to the current angle units"""
        if self.degrees:
            self.phi_spin.setRange(-180.0, 180.0); self.phi_spin.setSingleStep(1.0); self.phi_spin.setSuffix(' °')
            self.phidot_spin.setRange(-PHIDOT_MAX*RAD2DEG, PHIDOT_MAX*RAD2DEG); self.phidot_spin.setSingleStep(5.0); self.phidot_spin.setSuffix(' °/s')
        else:
            self.phi_spin.setRange(-pi, pi); self.phi_spin.setSingleStep(0.01); self.phi_spin.setSuffix(' rad')
            self.phidot_spin.setRange(-PHIDOT_MAX, PHIDOT_MAX); self.phidot_spin.setSingleStep(0.1); self.phidot_spin.setSuffix(' rad/s')

    def phi(self):
        """Return the value of the φ field in radians"""
        v = self.phi_spin.value()
        return v/RAD2DEG if self.degrees else v

    def phidot(self):
        """Return the value of the φ̇ field in rad/s"""
        v = self.phidot_spin.value()
        return v/RAD2DEG if self.degrees else v

    def set_phi(self, rad):
        self.phi_spin.setValue(rad*RAD2DEG if self.degrees else rad)

    def set_phidot(self, rad):
        self.phidot_spin.setValue(rad*RAD2DEG if self.degrees else rad)

    def set_units(self, degrees):
        """Redisplay the φ and φ̇ fields in the given angle units (the internal state is always radians)"""
        # The fields hold display-rounded values, so when a field is merely mirroring the
        # pendulum (equal to it within display precision), convert the exact state instead
        # of the field: otherwise φ=0.4π displayed as 1.257 rad would become 72.021°.
        tol = 0.0005/RAD2DEG if self.degrees else 0.0005 # half a display step, in radians
        phi, phidot = self.phi(), self.phidot()
        if abs(phi - self.p.phi) < tol: phi = self.p.phi
        if abs(phidot - self.p.phidot) < tol: phidot = self.p.phidot
        self.degrees = degrees
        self.set_unit_ranges()
        self.set_phi(phi)
        self.set_phidot(phidot)

    def sync_state(self):
        """Show the current state of the pendulum in the φ and φ̇ fields"""
        self.set_phi(self.p.phi)
        self.set_phidot(self.p.phidot)

    def set_editable(self, editable):
        """The φ and φ̇ fields track the running pendulum and are editable only when it is stopped"""
        self.phi_spin.setReadOnly(not editable)
        self.phidot_spin.setReadOnly(not editable)

    def choose_color(self):
        color = QColorDialog.getColor(QColor(self.color), self, 'Select the colour of the pendulum')
        if color.isValid():
            self.color = color.name()
            self.colorbtn.setStyleSheet('background-color: %s;' % self.color)

    def apply(self):
        """Apply the values entered in this tab to the pendulum. The φ and φ̇ fields are
           applied only when the pendulum is stopped (while it is live they just track it),
           so applying e.g. a new colour or length does not disturb the motion."""
        p = self.p
        pw.ani.pause()
        if not p.live:
            p.phi = self.phi()
            p.phidot = self.phidot()
        p.L = self.L_spin.value()
        p.color = self.color
        self.cw.tabs.setTabIcon(self.cw.tabs.indexOf(self), color_icon(p.color))
        pw.update_artists(p)
        pw.ani.resume()
        self.cw.status_msg.setText('Pendulum updated')

    def startstop(self):
        """Toggle this pendulum between live (simulated) and stopped (frozen in place)"""
        p = self.p
        p.live = not p.live
        self.startbtn.setText('&Stop' if p.live else '&Start')
        self.sync_state() # snap the fields to the exact state the pendulum is in right now
        self.set_editable(not p.live)
        self.cw.status_msg.setText('Pendulum ' + ('started' if p.live else 'stopped'))

    def delete(self):
        delete_pendulum(self.p)
        self.cw.status_msg.setText('Pendulum deleted')

class ControlWindow(CaptionedWindow):
    def __init__(self, geometry = None, state = None):
        super().__init__()
        self.cw_create_tabs()
        self.cw_create_menus()
        self.setup_caption(PROGRAM, below=self.menubar)
        self.cw_create_time_indicator()
        self.cw_create_statusbar()
        self.cw_create_dt_control()

        self.playicon = QIcon('icons/play.png')
        self.pauseicon = QIcon('icons/pause.png')
        self.playpausebtn = QToolButton(self, icon=self.playicon)
        self.frameforwardbtn = QToolButton(self, icon=QIcon('icons/forward.png'))
        self.framebackbtn = QToolButton(self, icon=QIcon('icons/rewind.png'))
        self.playpausebtn.clicked.connect(playpause)
        self.frameforwardbtn.clicked.connect(step_forward)
        self.framebackbtn.clicked.connect(step_backward)
        self.recordbtn = QPushButton('&Record')
        self.recordbtn.clicked.connect(self.cw_record)
        self.zerobtn = QPushButton('&Zero')
        self.zerobtn.clicked.connect(reset_time)
        self.textcheck = QCheckBox('Show state &text')
        self.textcheck.setChecked(settings.value('show_text', True, type=bool))
        if not self.textcheck.isChecked(): pw.set_show_text(False) # the plot shows the texts by default
        self.textcheck.toggled.connect(lambda checked: pw.set_show_text(checked))
        # the buttons must never take keyboard focus, or pressing Space would 'click'
        # whichever happened to be focused (stepping the animation, reversing dt via the
        # rewind button, creating pendulums via '+'); mouse or Alt+mnemonic only
        for b in (self.playpausebtn, self.frameforwardbtn, self.framebackbtn,
                  self.recordbtn, self.zerobtn, self.textcheck, self.addbtn):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.cw_setup_layout()

        self.setWindowIcon(QIcon(LOGO))
        self.setWindowTitle(PROGRAM)
        if geometry: self.restoreGeometry(geometry)
        if state: self.restoreState(state)
        self.show()

    def cw_create_tabs(self):
        """Create the Control tab, one tab per pendulum and the '+' button, set the container as the central widget"""
        self.tabs = QTabWidget()
        self.controls = QWidget()
        self.tabs.addTab(self.controls, '&Control')
        self.pendtabs = []
        for p in pendulums: self.add_tab(p)
        self.addbtn = QToolButton(self)
        self.addbtn.setText('+')
        self.addbtn.clicked.connect(self.cw_new_pendulum)
        self.tabs.setCornerWidget(self.addbtn, Qt.Corner.TopRightCorner)
        self.tabs.currentChanged.connect(lambda i: self.sync_current_tab())
        self.setCentralWidget(self.tabs)

    def add_tab(self, p):
        """Create a new tab for pendulum p, marked with a swatch of its colour"""
        tab = PendulumTab(p, self)
        self.pendtabs.append(tab)
        i = self.tabs.addTab(tab, 'Pendulum &%d' % len(self.pendtabs))
        self.tabs.setTabIcon(i, color_icon(p.color))
        return tab

    def remove_tab(self, p):
        """Remove the tab of pendulum p and renumber the remaining tabs"""
        for tab in self.pendtabs:
            if tab.p is p:
                self.pendtabs.remove(tab)
                self.tabs.removeTab(self.tabs.indexOf(tab))
                tab.deleteLater()
                break
        for i, tab in enumerate(self.pendtabs, start=1):
            self.tabs.setTabText(self.tabs.indexOf(tab), 'Pendulum &%d' % i)

    def cw_new_pendulum(self):
        """Create a new pendulum in the staged (frozen) state and open its tab"""
        p = Pendulum(phi=pi/3, phidot=0.0, L=1.0, color=next_color(), live=False)
        add_pendulum(p)
        self.tabs.setCurrentWidget(self.pendtabs[-1])
        self.status_msg.setText('Pendulum %d created, press Start on its tab to simulate it' % len(self.pendtabs))

    def cw_record(self):
        """Start recording the animation into a new video file, from the current moment.
           The recording is finished by stopping the animation (pause or single-step)."""
        filename, _ = QFileDialog.getSaveFileName(self, 'Record the animation to',
                          strftime('pendulum-%Y%m%d-%H%M%S.mp4'), 'Video files (*.mp4);;All files (*)')
        if not filename: return
        try:
            pw.start_recording(filename)
        except Exception as err:
            self.status_msg.setText('Recording failed: %s' % err)
            return
        self.recordbtn.setText('Recording...')
        self.recordbtn.setEnabled(False)
        self.status_msg.setText('Recording to %s, pause the animation to finish' % filename)
        if not anim_running: playpause() # frames are captured only while the animation runs

    def cw_record_reset(self, msg):
        """Return the record button to its idle state and show msg in the status bar"""
        self.recordbtn.setEnabled(True)
        self.recordbtn.setText('&Record')
        self.status_msg.setText(msg)

    def sync_current_tab(self):
        """Keep the φ and φ̇ fields of the visible pendulum tab tracking the simulation"""
        tab = self.tabs.currentWidget()
        if isinstance(tab, PendulumTab) and tab.p.live: tab.sync_state()

    def cw_create_menus(self):
        """Create menubar, menu actions and attach them to the menubar.
           The menubar is standalone (not QMainWindow.menuBar()) because it is stacked
           under the caption bar in the menu widget slot by setup_caption()."""
        self.menubar = QMenuBar(self)
        self.menubar.setNativeMenuBar(False)
        self.fileMenu = self.menubar.addMenu('File')
        self.viewMenu = self.menubar.addMenu('View')
        self.helpMenu = self.menubar.addMenu('Help')

        self.exitAction = QAction(QIcon('icons/exit.png'), 'E&xit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Save current state and exit')
        self.exitAction.triggered.connect(app.quit)
        self.fileMenu.addAction(self.exitAction)

        self.tooltipsAction = QAction('Show &tooltips', self, checkable=True, checked=False)
        self.tooltipsAction.setStatusTip('Toggle showing tooltip popups')
        self.tooltipsAction.triggered.connect(self.cw_tooltips_toggle)
        self.viewMenu.addAction(self.tooltipsAction)

        def about(): QMessageBox.about(self, PROGRAM, "<p>Computer simulation of mathematical pendulums in the the phase space.</p><p>To report a bug, please visit our github repository at: <A HREF='https://github.com/tigran123/quantum-infodynamics'>https://github.com/tigran123/quantum-infodynamics</A></p>")
        self.aboutAction = QAction(QIcon('icons/about.png'), '&About', self)
        self.aboutAction.setStatusTip('Information about the program')
        self.aboutAction.triggered.connect(about)
        self.helpMenu.addAction(self.aboutAction)

        def aboutQt(): QMessageBox.aboutQt(self, PROGRAM)
        self.aboutQtAction = QAction(QIcon('icons/qt.png'), 'About &Qt', self)
        self.aboutQtAction.setStatusTip('Information about the Qt version')
        self.aboutQtAction.triggered.connect(aboutQt)
        self.helpMenu.addAction(self.aboutQtAction)

    def cw_create_dt_control(self):
        """Create the spinboxes controlling the ODE integration timestep Δt (negative runs
           time backwards) and the limit of its allowed range"""
        self.dt_spin = QDoubleSpinBox(decimals=3, minimum=-dtlim, maximum=dtlim, singleStep=0.001, suffix=' s')
        self.dt_spin.setValue(dt)
        # apply typed values only on Enter/focus-out, not on every keystroke: the
        # intermediate values (e.g. the '0' while typing '0.01') would hit the running simulation
        self.dt_spin.setKeyboardTracking(False)
        self.dt_spin.valueChanged.connect(update_dt)
        self.label_dt = QLabel('Δt:')
        self.dtlim_spin = QDoubleSpinBox(decimals=3, minimum=0.001, maximum=100.0, singleStep=0.1, suffix=' s')
        self.dtlim_spin.setValue(dtlim)
        self.dtlim_spin.setKeyboardTracking(False)
        self.dtlim_spin.valueChanged.connect(update_dtlim) # NB: connected after setValue(), while cw is not yet bound
        self.label_dtlim = QLabel('|Δt| ≤')

    def cw_setup_layout(self):
        """Create and connect the layouts for the main control panel: one hbox per row
           (a shared grid would couple column widths across unrelated rows)"""
        transport = QHBoxLayout()
        transport.setSpacing(4)
        for b in (self.framebackbtn, self.playpausebtn, self.frameforwardbtn):
            b.setIconSize(QSize(24, 24))
            transport.addWidget(b)
        row1 = QHBoxLayout()
        row1.addLayout(transport)
        row1.addSpacing(24)
        row1.addWidget(self.time_label)
        row1.addWidget(self.time_lcd)
        row1.addWidget(self.zerobtn)
        row1.addSpacing(24)
        row1.addWidget(self.label_dt)
        row1.addWidget(self.dt_spin)
        row1.addWidget(self.label_dtlim)
        row1.addWidget(self.dtlim_spin)
        row1.addStretch(1)
        row2 = QHBoxLayout()
        row2.addWidget(self.recordbtn)
        row2.addWidget(self.textcheck)
        row2.addStretch(1)
        # the FPS readout is about rendering, not physics: tuck it into the bottom-right corner
        fpsrow = QHBoxLayout()
        fpsrow.addStretch(1)
        fpsrow.addWidget(self.fps_label)
        vbox = QVBoxLayout(self.controls)
        vbox.addLayout(row1)
        vbox.addLayout(row2)
        vbox.addStretch(1)
        vbox.addLayout(fpsrow)

    def cw_create_time_indicator(self):
        """Create the label and LCD window for the current time and the FPS readout"""
        self.time_label = QLabel('Time (s):')
        self.time_lcd = QLCDNumber(self)
        self.time_lcd.setDigitCount(9) # up to '99999.999'
        self.time_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.time_lcd.setStyleSheet('QLCDNumber {background: #8CB398;}')
        self.time_lcd.setFixedSize(150, 28) # keep the LCD from ballooning with the window
        self.fps_label = QLabel('FPS: —')
        self.fps_label.setMinimumWidth(80)
        # right-aligned so its right edge stays put in the corner as the number changes width
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)

    def cw_create_statusbar(self):
        """Create status bar and permanent message widget for the status info"""
        self.statusbar = self.statusBar()
        self.statusbar.setStyleSheet('QStatusBar {border-top: 1px outset grey;}')
        self.status_msg = QLabel('Program ready')
        self.statusbar.addPermanentWidget(self.status_msg) # prevent ovewriting status by other widgets

    def cw_tooltips_toggle(self, state):
        if state:
            self.playpausebtn.setToolTip('Start/pause the animation')
            self.frameforwardbtn.setToolTip('Step forward one time step')
            self.framebackbtn.setToolTip('Step back one time step')
            self.time_lcd.setToolTip('Simulation time in seconds')
            self.zerobtn.setToolTip('Reset the simulation time to zero')
            self.dt_spin.setToolTip('ODE integration time step Δt in seconds; negative runs time backwards')
            self.dtlim_spin.setToolTip('Largest |Δt| settable in the Δt field, in seconds')
            self.recordbtn.setToolTip('Record the animation to a video file until it is paused')
            self.textcheck.setToolTip('Render the state and time texts in the plotting window')
            self.fps_label.setToolTip('Animation frames rendered per second')
            self.addbtn.setToolTip('Create a new pendulum')
        else:
            self.playpausebtn.setToolTip(None)
            self.frameforwardbtn.setToolTip(None)
            self.framebackbtn.setToolTip(None)
            self.time_lcd.setToolTip(None)
            self.zerobtn.setToolTip(None)
            self.dt_spin.setToolTip(None)
            self.dtlim_spin.setToolTip(None)
            self.addbtn.setToolTip(None)
            self.recordbtn.setToolTip(None)
            self.textcheck.setToolTip(None)
            self.fps_label.setToolTip(None)

pendulums = [
             Pendulum(phi=pi, phidot=0, L=1.0, color='b'),
             Pendulum(phi=0, phidot=0, L=1.0, color='b'),
             Pendulum(phi=0.4*pi, color='k'),
             Pendulum(phi=0.4*pi + 0.01*pi/2, color='r'),
             Pendulum(phi=pi/2, phidot=4.42869, color='g'),
             Pendulum(phi=pi/2, phidot=4.8, color='m'),
             Pendulum(phi=pi/2, phidot=4, color='c')
            ]

app = QApplication(sys.argv)
cursor_reset = CursorReset()
app.installEventFilter(cursor_reset)
settings = QSettings(COMPANY, PROG)
pw = PlotWindow(geometry = settings.value('plot_geometry'), state = settings.value('plot_windowState'))

cw = ControlWindow(geometry = settings.value('control_geometry'), state = settings.value('control_windowState'))
app.aboutToQuit.connect(main_exit)
rc = app.exec()
# Destroy the windows NOW, while the QApplication still exists. Left to the interpreter
# shutdown, the garbage collector destroys the module globals in arbitrary order, and
# whenever the QApplication went before the windows, destroying a QWidget after its
# QApplication is a use-after-free inside Qt/sip: an intermittent SIGSEGV on exit.
# collect() is needed because the windows sit in reference cycles (window <-> animation).
del pw, cw
collect()
sys.exit(rc)
