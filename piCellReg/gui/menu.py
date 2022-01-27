from PyQt5 import QtGui
from PyQt5.QtWidgets import QAction, QMenu, QFileDialog


def fist_menu(parent):
    # create the action
    test_act = QAction("&test", parent)
    test_act.setShortcut("Ctrl+R")
    test_act.triggered.connect(lambda: load_dialog(parent))
    parent.addAction(test_act)

    # make the menu at the end
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(test_act)


def load_dialog(parent):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    name = QFileDialog.getOpenFileName(
        parent, "Open processed file", filter="*.npy", options=options
    )
    parent.root_path = name[0]
