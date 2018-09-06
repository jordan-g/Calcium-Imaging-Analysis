import sys
from controller import Controller
from gui_controller import GUIController

# import the Qt library
try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    pyqt_version = 4
except:
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    pyqt_version = 5

if __name__ == "__main__":
    app = QApplication(sys.argv)

    if pyqt_version == 5:
        # enable high DPI scaling
        app.setAttribute(Qt.AA_UseHighDpiPixmaps)
        app.setAttribute(Qt.AA_EnableHighDpiScaling)

    # create controllers
    controller     = Controller()
    gui_controller = GUIController(controller)

    # import videos
    gui_controller.import_videos()

    app.exec_()