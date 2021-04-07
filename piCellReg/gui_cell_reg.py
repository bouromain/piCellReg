import numpy as np
import pyqtgraph as pg
import sys
from PyQt5 import QtGui, QtCore

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("yo mama")
        self.setGeometry(100,100,1500,800)

        # Layout

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        hour = [1,2,3,4,5,6,7,8,9,10]
        temperature = [30,32,34,32,33,31,29,32,35,45]

        self.graphWidget.plot(hour, temperature)

        # show stuff
        self.show()


def main():
    app = QtGui.QApplication(sys.argv)
    GUI = MainWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
