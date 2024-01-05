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
# time.sleep(0.001)
# cam.SensorCooling = True
# cam.FanSpeed = 'On'

# waiting for temperature to stabilise
# while(cam.TemperatureStatus != "Stabilised"):
#     time.sleep(5)
#     print("Temperature: {:.5f}C".format(cam.SensorTemperature), end="  ")
#     print("Status: '{}'".format(cam.TemperatureStatus))
#     if cam.TemperatureStatus == "Fault":
#         err_str = "Camera faulted when cooling to target temperature"
#         raise RuntimeError(err_str)

# print("Sensor Temperature now Stabilised and Camera is ready to use")

# cam.ExposedPixelHeight = 1000
# print(cam.ExposedPixelHeight)

# cam.ShutterMode = 'Auto'
# print(cam.ShutterMode)
cam.ElectronicShutteringMode = 'Global'
cam.TriggerMode = 'External'
cam.AuxiliaryOutSource = 'FireAll'
cam.PixelReadoutRate = '280 MHz'
cam.SimplePreAmpGainControl = '16-bit (low noise & high well capacity)'
cam.VerticallyCenterAOI = True
cam.AOIHeight = 1000
cam.Overlap = 0
cam.ExposureTime = 0.005

print(cam.ExposureTime)
print(cam.exposedpixelheight)
print(cam.ElectronicShutteringMode)
print(cam.TriggerMode)
print(cam.Overlap)
# cam.aoiheight = 1000
# cam.aoitop = 500
# cam.aoiwidth = 1000
# cam.aoileft = 500

# cam.ExposedPixelHeight = 2000
# print(cam.ExposedPixelHeight)

print("Getting image data...")
# cam.ExposureTime = 0.1
# cam.CycleMode = 'Fixed'
# cam.ShutterMode = 'Open'
# cam.ElectronicShutteringMode = 'Global'
# acq = cam.acquire(timeout=20000)
acqs = cam.acquire_series(frame_count=10, timeout=20000)
print("Performed single acquisition.")
cam.close()

plt.imshow(np.array(acqs[-1].image, dtype=np.float64))
plt.colorbar()
plt.show()