# Calcium Imaging Analysis

User-friendly software for motion-correcting calcium imaging videos, automatically finding ROIs using the CNMF method and manually refining them. This software uses the [CaImAn](https://github.com/flatironinstitute/CaImAn) package under the hood for motion correction, ROI finding, and refinement of ROIs. Optionally, the [suite2p](https://github.com/MouseLand/suite2p) package can also be used for ROI finding.

## Required Modules
This software requires Python 3 and the following modules:

- CaImAn (and all of its required modules – see [the CaImAn repository](https://github.com/flatironinstitute/CaImAn) for installation instructions)
- (Optional) suite2p (see [the suite2p repository](https://github.com/MouseLand/suite2p) for installation instructions)
- numpy
- matplotlib
- scikit-image
- scipy
- opencv-python
- PyQt5
- keras
