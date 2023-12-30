from pyAndorSDK3 import AndorSDK3
import logging
import matplotlib.pyplot as plt

class AndorZL41Wave:
    def __init__(self, parent):
        self.parent = parent

        self.sdk3 = AndorSDK3()
        logging.info(f"Using SDK version {self.sdk3.SoftwareVersion}.")
        logging.info("Found {:d} camera(s) in the system.".format(self.sdk3.DeviceCount))

        cam_index = 0
        logging.info("Connecting to camera of index = {:d} ...".format(cam_index))
        try:
            self.camera = self.sdk3.GetCamera(cam_index)
            logging.info("Connected to camera of index = {:d}.".format(cam_index))
            logging.info(f"Camera model: {self.camera.CameraModel}. Serial number: {self.camera.SerialNumber}. Interface: {self.camera.InterfaceType}.")
            logging.info(f"Firmware version: {self.camera.FirmwareVersion}.")
        except Exception as err:
            logging.error(f"Failed to connect to camera of index = {cam_index}.")
            logging.error(err)
            self.camera = None

        print(self.camera.CoolerPower)

        print("Camera sensor temperature: {:.2f} C.".format(self.camera.SensorTemperature))

        print("Getting image data...")
        acq = self.camera.acquire(timeout=20000)
        print("Performed single acqition")
        self.camera.close()

        plt.imshow(acq.image)
        plt.show()


cam = AndorZL41Wave(None)

