import pylablib as pll
from pylablib.devices import Andor
cam = Andor.AndorSDK3Camera()
print(cam.get_attribute_value("CameraAcquiring"))

cam.set_attribute_value("ExposureTime", 0.1)  # set the exposure to 100ms
print(cam.cav["ExposureTime"])  # get the exposure; could also use cam.get_attribute_value("ExposureTime")

cam.close()  # close the camera