import os, pathlib, shutil, sys, warnings

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QGridLayout,
    QCheckBox,
    QLineEdit,
    QLabel,
)

import menu


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("pyCellReg the cool app to do cell registration")
        self.setStyleSheet("QMainWindow {background: 'black';}")
        # -- Main menu
        menu.fist_menu(self)

        # to remove macos weird menu at the top
        main_menu = self.menuBar()
        main_menu.setNativeMenuBar(False)

        # -- Main frame
        cwidget = QWidget()
        self.l0 = QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)

        # -- Main variable
        self.root_path = ""
        l1 = QLabel()
        l1.setText("Hello World")
        l1.setStyleSheet("background-color: lightgreen")
        self.l0.addWidget(l1)


def run_gui():
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    GUI = MainWindow()
    GUI.show()
    ret = app.exec_()
    # GUI.save_gui_data()
    sys.exit(ret)


if __name__ == "__main__":
    run_gui()


# """
# nOTE FOR NON AFINE
# https://stackoverflow.com/questions/41133614/how-to-use-qpainter-in-a-qwidget-in-pyqt4
# """

# import numpy as np
# import pyqtgraph as pg
# import sys
# from PyQt5 import QtGui, QtCore


# class MainWindow(QtGui.QMainWindow):
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         self.setWindowTitle("load data or process")
#         self.setGeometry(100, 100, 1500, 800)

#         # Layout
#         cwidget = QtGui.QWidget()
#         self.l0 = QtGui.QGridLayout()
#         cwidget.setLayout(self.l0)
#         self.setCentralWidget(cwidget)

#         # populate layout
#         self.make_list()
#         # self.make_images()

#     def make_list(self):
#         self.list_left = ListSession()
#         self.list_left.insertItem(0, "Red")
#         self.list_left.insertItem(1, "blue")
#         self.list_left.insertItem(2, "bulue")
#         self.l0.addWidget(self.list_left, 0, 0, 1, 2)

#         self.list_right = ListSession()
#         self.list_right.insertItem(0, "Red2")
#         self.list_right.insertItem(1, "blue")
#         self.list_right.insertItem(2, "green")
#         self.list_right.itemClicked.connect(self.list_right.clicked)

#         self.l0.addWidget(self.list_right, 0, 1, 1, 2)

#     def make_images(self):
#         p = "/home/bouromain/Sync/tmpData/crossReg/4453/20201217/ops.npy"
#         ops1 = np.load(p, allow_pickle=True)
#         ops1 = ops1.item()

#         self.win = pg.GraphicsLayoutWidget()
#         self.l0.addWidget(self.win, 0, 1)
#         self.viewIm1 = pg.ViewBox()
#         self.win.addItem(self.viewIm1, 0, 1)

#         self.im1 = pg.ImageItem(ops1["meanImgE"])
#         self.viewIm1.addItem(self.im1)


# class ListSession(QtGui.QListWidget):
#     def clicked(self, item):
#         print("yo")


# def main():
#     app = QtGui.QApplication(sys.argv)
#     GUI = MainWindow()
#     GUI.show()
#     sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()
