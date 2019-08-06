import sys
from controller import Controller
from gui_controller import GUIController

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    controller = Controller()
    gui_controller = GUIController(controller)
    gui_controller.import_videos()

    app.exec_()