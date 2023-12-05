# AndorZL41-python-control

This graphical user interface of Andor ZL41 Wave 5.5 scientific CMOS camera is designed for applications in atomic, molecular and optical (AMO) physics experiments.

*Forked from the SrF branch of [pixelfly-python-control](https://github.com/SrFDeMilleGroup/pixelfly-python-control) repository.* 

Key features include:
- camera configuration
- real time image display
- region of interest (ROI) selection
- real time image analysis (including 2D Gaussian fit and gaussian filter)
- target system parameter scan (in coordinate with other programs/devices)
- socket network communication (as server)
- save and load program settings

Written in Python 3.8.6. Frequently tested on Window 7. A 24-inch 1080p monitor is recommended for the best graphical display.

## Install the camera
- Install SDK3 from Andor: https://andor.oxinst.com/downloads/view/andor-sdk3-3.15.30092.0
- Maybe driver pack? https://andor.oxinst.com/downloads/view/andor-driver-pack-3.15.30092.0-(scmos)

## Useful third party software resources
Python (for SDK3)
- https://pylablib.readthedocs.io/en/stable/devices/Andor.html
- https://gitlab.com/ptapping/andor3
- https://github.com/hamidohadi/pyandor/tree/master/Camera
- https://github.com/esrf-bliss/LImA

Other softwares
- https://andor.oxinst.com/tools/third-party-software-matrix

## To-do
- check widgets scripts and make sure it's the latest
- oprn file explorer when saving settings
- better way to save/load settings?
- camera take multiple images and only readout at the end of the cycle
- interface with Andor ZL41, and add widgets for its settings