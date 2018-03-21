import sys
from controller import Controller

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
        app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    controller = Controller()
    controller.select_videos_to_import()
    app.exec_()