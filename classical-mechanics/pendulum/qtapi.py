"""
  Interface to Qt API (PyQt6, or whatever binding matplotlib's qt_compat selects)
  Author: Tigran Aivazian
  Released under GPLv3, 2017
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.qt_compat import QtWidgets, QtCore, QtGui

QMessageBox = QtWidgets.QMessageBox
QMainWindow = QtWidgets.QMainWindow
QWidget = QtWidgets.QWidget
QTabWidget = QtWidgets.QTabWidget
# QAction lives in QtGui since Qt6, in QtWidgets in Qt5
QAction = QtGui.QAction if hasattr(QtGui, 'QAction') else QtWidgets.QAction
QLCDNumber = QtWidgets.QLCDNumber
QLabel = QtWidgets.QLabel
QSlider = QtWidgets.QSlider
QApplication = QtWidgets.QApplication
QPushButton = QtWidgets.QPushButton
QToolButton = QtWidgets.QToolButton
QGridLayout = QtWidgets.QGridLayout
QFormLayout = QtWidgets.QFormLayout
QVBoxLayout = QtWidgets.QVBoxLayout
QHBoxLayout = QtWidgets.QHBoxLayout
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QColorDialog = QtWidgets.QColorDialog
QRadioButton = QtWidgets.QRadioButton
QButtonGroup = QtWidgets.QButtonGroup
QMenuBar = QtWidgets.QMenuBar
QFileDialog = QtWidgets.QFileDialog
Qt = QtCore.Qt
QSettings = QtCore.QSettings
QEvent = QtCore.QEvent
QIcon = QtGui.QIcon
QColor = QtGui.QColor
QPalette = QtGui.QPalette
