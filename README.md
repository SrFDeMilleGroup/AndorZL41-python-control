# AndorZL41-python-control

This graphical user interface GUI of Andor ZL41 Wave 5.5 scientific CMOS camera is designed for applications in atomic, molecular and optical (AMO) physics experiments.

*Forked from the SrF branch of [pixelfly-python-control](https://github.com/SrFDeMilleGroup/pixelfly-python-control) repository.* 

Key features include:
- camera configuration
- real time image display
- region of interest (ROI) selection
- real time image analysis (including 2D Gaussian fit and gaussian filter)
- target system parameter scan (in coordinate with other programs/devices)
- socket network communication (as server)
- save and load program settings

Developed in Python 3.11 and Window 10.

## Install the camera
- Install SDK3 from Andor: https://andor.oxinst.com/downloads/view/andor-sdk3-3.15.30092.0
- Navigate to *C:/Program Files/Andor SDK3/Python/pyAndorSDK3/* to find the python wrapper manual and install the python package.
- Navigate to *C:/Program Files/Andor SDK3/Python/pyAndorSDK3/docs/* to find more python wrapper and SDK information.
- Navigate to *C:/Program Files/Andor SDK3/Docs/Camera Feature Reference/* to find available camera features (for programing).

## Camera user guide
### Sensor cooling
To reduce dark current noise, the sCMOS sensor is cooled by a thermoelectric cooler (TEC) to reach the target temperature of 0 degree Celsius (non-adjustable). Fan cooling or water cooling is further implemented in the camera to cool down the TEC or electronics heatsink. TEC and fan can be individually turned on or off from the camera software. Sensor temperature and cooling status can also be read out.

### Noise filter
Two noise filters are implemented in the camera by the manufacturer Andor, spurious noise filter and static blemish correciton. Spurious noise, in the context of sCMOS cameras, mainly refers to the random telegraph noise, which originates from the defects in sCMOS pixels that can randomly trap or release photoelectrons and cause fluactuation. (Just imagine a semiconductor defect with a dangling chemical bond that have the tendency to catch an electron to form full electronic shell.) It is also called salt-and-pepper noise, because of its effect of causing bright and dark pixels in images mimicing and the color of salt and pepper. Such noise is random and dynamic, but Andor provides built-in algorithm to detect and filter it. The second noise filter, static blemish correction, compensates the readings of identified faulty pixels (hot pixels, noisy pixels, unresponsive pixels) by replacing them with the average readings of surrounding 8 pixels. Both filters can be individually turned on or off from software. For more information, see the [Hardware User Guide](https://andor.oxinst.com/downloads/view/zyla-scmos-hardware-guide).

### Area of interest (AOI)
Andor provides (nearly) full hardware AOI control of the sCMOS sensor, including AOI size, location and binning. The following table lists the range of each parameter.

|Direction|horizontal|vertical|
|:--:|:--:|:--:|
|sensor size (unbinned pixel)|2560|2160|
|physical pixel size (um)|6.5|6.5|
|min binning (unbinned pixel)|1|1|
|max binning (unbinned pixel)|640|2160|
|min AOI size|4 *binned* pixels|8 *unbinned* pixels|
|max AOI size (unbinned pixel)|2560|2160|
|AOI location|anywhere on sensor|anywhere on snesor|

Location of AOI can be specified by the row (column) index of its top-most row (left-most column), and the size of AOI. Alternatively, one can set AOI to be centered at the sCMOS sensor, and only use AOI size to specify. In this case, indices of the top and left of AOI are disabled. Setting AOI to be vertically centered is a built-in function of Andor SDK3, but setting it horizontally centered is coded by ourselves. Following Andor SDK3, size of AOI is specified in number of *binned* pixels in this GUI, while all other values are in the number of *unbinned* pixels. 

Changing binning can have minor effects on AOI size and location, just to make corresponding values integers and ensure AOI fit on sensor. Otherwise, the new AOI after binning change should mostly overlap with the orginal one on the physical sCMOS sensor. Similarly, changing AOI size can also overwrite AOI top and left indices, to keep AOI within the sensor. AOI top and left position have the least priority here, so they shouldn't trigger the change of other values. More information can be found in [Hardware User Guide](https://andor.oxinst.com/downloads/view/zyla-scmos-hardware-guide).

Lastly, it's worth mentioning that unlike CCD cameras, [hardware binning in sCMOS cameras](https://andor.oxinst.com/learning/view/article/binning-in-the-neo-and-zl41-wave-scmos-cameras) doesn't really bin adjacent pixels together, but all pixels are still read out individually. This is limited by the architecture of sCMOS sensor. Hardware binning actually happens in FPGA post-processing where adjacent pixel readings are summed up. Nevertheless, binning still increases frame rate by reducing the size of image to transfer. 

## GPU acceleration
Here we use a GPU to accelerate real time 2D Gaussian fit. Note that Andor also offers a [GPU express package](https://andor.oxinst.com/products/gpu-express/gpu-express) but it is not free, and not used here.

## Useful third party software resources
Python (for SDK3)
- https://pylablib.readthedocs.io/en/stable/devices/Andor.html
- https://gitlab.com/ptapping/andor3
- https://github.com/hamidohadi/pyandor/tree/master/Camera
- https://github.com/esrf-bliss/LImA (only for Linux)

Other softwares
- https://andor.oxinst.com/tools/third-party-software-matrix

## To-do
- start and close functions
- dummy camera: exposure time in different shutter and trigger mode. overlap in different shutter and exposure mode.
- reconnect camera
- save/load program settings
- acquisition thread
- camera take multiple images and only readout at the end of the cycle
- apply to Apogee camera (need to switch to Andor sdk2)
- better readme.md file