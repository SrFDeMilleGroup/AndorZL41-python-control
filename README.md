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

Written in Python 3.10. Frequently tested on Window 10. A 24-inch 1080p monitor is recommended for the best graphical display.

## Install the camera
- Install SDK3 from Andor: https://andor.oxinst.com/downloads/view/andor-sdk3-3.15.30092.0
- Navigate to *C:/Program Files/Andor SDK3/Python/pyAndorSDK3/* to find the python wrapper manual and install the python package.
- Navigate to *C:/Program Files/Andor SDK3/Python/pyAndorSDK3/docs/* to find more python wrapper and SDK information.
- Navigate to *C:/Program Files/Andor SDK3/Docs/Camera Feature Reference/* to find available camera features (for programing).

## GPU acceleration
Here we use a GPU to accelerate real time 2D Gaussian fit and Gaussian fit. Note that Andor also offers a [GPU express package](https://andor.oxinst.com/products/gpu-express/gpu-express) but it is not free.

## Useful third party software resources
Python (for SDK3)
- https://pylablib.readthedocs.io/en/stable/devices/Andor.html
- https://gitlab.com/ptapping/andor3
- https://github.com/hamidohadi/pyandor/tree/master/Camera
- https://github.com/esrf-bliss/LImA (only for Linux)

Other softwares
- https://andor.oxinst.com/tools/third-party-software-matrix

## To-do
- check widgets scripts and make sure it's the latest
- oprn file explorer when saving settings
- better way to save/load settings?
- camera take multiple images and only readout at the end of the cycle
- interface with Andor ZL41, and add widgets for its settings
- GPU acceleration (Andor GPU Express is not free, but maybe we can do 2D gaussian fit and gaussain filter on GPU to accelerate)
- apply to Apogee camera