"""
  Interface to PyQt5 API
  Author: Tigran Aivazian
  Released under GPLv3, 2017
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.qt_compat import QtWidgets, QtCore, QtGui

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
