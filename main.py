import sys
from motion_correction.controller import Controller as MotionCorrectionController
from watershed.controller import Controller as WatershedController
from cnmf.controller import Controller as CNMFController

# import the Qt library
try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except:
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *

import numpy as np

class Controller():
    def __init__(self):
        # self.image_path = "Videos/Michael July 28 - z 6.tif"
        self.image_path = "Videos/Michael July 28 (3) - z 1.tif"
        # self.image_path = "Michael July 28 (2) - z 5_mc.npy"
        # self.image_path = "Michael July 28 (3) - z 1_mc.npy"
        # self.image_path = "Michael July 28 - z 6.npy"
        self.image_path = "movie_3.npy"
        # self.image_path = "../Aug.15.17/Aug.15.17_long_oom_0.1toBF_lv25_0001-1.tif"
        self.image_path = "Aug.15.17_long_oom_0.1toBF_lv25_0001-1_mc.npy"
        self.image_path = "../Aug.15.17/Aug.15.17_long_oom_0.1toBF_lv25_laser6.5pct-1.tif"
        self.image_path = "Aug.15.17_long_oom_0.1toBF_lv25_laser6.5pct-1_mc.npy"

        # video = np.load("movie_3.npy")
        # print(np.amin(video))
        # np.save("movie_3.npy", np.transpose(video, (1, 0, 2)))
        # print(video.shape)

        self.run_watershed()

    def run_motion_correction(self, image_path):
        self.app = QApplication(sys.argv)
        self.motion_correction_controller = MotionCorrectionController(self)
        self.motion_correction_controller.open_video(image_path)
        self.app.exec_()

    def run_watershed(self):
        # self.app = QCoreApplication.instance()
        self.app = QApplication(sys.argv)
        self.app.setAttribute(Qt.AA_UseHighDpiPixmaps)
        self.watershed_controller = WatershedController()
        self.watershed_controller.select_and_open_video()
        self.app.exec_()

    def run_cnmf(self, image_path):
        self.app = QCoreApplication.instance()
        self.cnmf_controller = CNMFController(self)
        self.cnmf_controller.open_image(image_path)
        self.app.exec_()

    def motion_correction_done(self, motion_corrected_image_path):
        self.run_watershed(motion_corrected_image_path)

    def motion_correction_done_cnmf(self, motion_corrected_image_path):
        self.run_cnmf(motion_corrected_image_path)

    def close(self):
        self.app.quit()

if __name__ == "__main__":
    controller = Controller()