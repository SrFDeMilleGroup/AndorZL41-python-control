from pyAndorSDK3 import AndorSDK3
import matplotlib.pyplot as plt

sdk3 = AndorSDK3()
print(f"Using SDK version {sdk3.SoftwareVersion}.")
print("Found {:d} camera(s).".format(sdk3.DeviceCount))

cam_index = 0
print("Connecting to camera (index = {:d}) ...".format(cam_index))
cam = sdk3.GetCamera(0)
print(f"Connected to camera. Camera model: {cam.CameraModel}. Serial number: {cam.SerialNumber}. Interface: {cam.InterfaceType}. Firmware version: {cam.FirmwareVersion}.")

print(cam.CoolerPower)

print("Camera sensor temperature: {:.2f} C.".format(cam.SensorTemperature))

print("Getting image data...")
acq = cam.acquire(timeout=20000)
print("Performed single acqition")
cam.close()

plt.imshow(acq.image)
plt.show()