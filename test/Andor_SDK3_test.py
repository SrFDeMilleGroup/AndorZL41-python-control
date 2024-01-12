from pyAndorSDK3 import AndorSDK3
import matplotlib.pyplot as plt
import time
import numpy as np

sdk3 = AndorSDK3()
print(f"Using SDK version {sdk3.SoftwareVersion}.")
print("Found {:d} camera(s).".format(sdk3.DeviceCount))

cam_index = 0
print("Connecting to camera (index = {:d}) ...".format(cam_index))
cam = sdk3.GetCamera(cam_index)
print(f"Connected to camera. Camera model: {cam.CameraModel}. Serial number: {cam.SerialNumber}. Interface: {cam.InterfaceType}. Firmware version: {cam.FirmwareVersion}.")

print("Camera sensor temperature: {:.2f} C.".format(cam.SensorTemperature))

cam.ElectronicShutteringMode = 'Global'
cam.TriggerMode = 'Internal'
cam.Overlap = 0
cam.ExposureTime = 2.5e-3

cam.aoivbin = 1
cam.aoihbin = 2

cam.VerticallyCenterAOI = True
cam.aoiheight = 500
# cam.aoitop = 500
cam.aoiwidth = int(750/cam.aoihbin)
cam.aoileft = int((2560-cam.aoiwidth*cam.aoihbin)/2) + 1

cam.AuxiliaryOutSource = 'FireAll'
cam.PixelReadoutRate = '280 MHz'
cam.SimplePreAmpGainControl = '16-bit (low noise & high well capacity)'

print("Getting image data...")
acqs = cam.acquire_series(frame_count=10, timeout=20000)
print(f"Performed {len(acqs)} acquisition(s).")

# cam.close()

# plt.imshow(np.array(acqs[-1].image, dtype=np.float64))
# plt.colorbar()
# plt.show()