from pyAndorSDK3 import AndorSDK3
import logging

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

    def set_cooling(self, type: str, cooling: bool):
        """Turn on or off camera sensor cooling."""

        assert type in ['sensor', 'fan']
        assert type(cooling) == bool

        if self.camera is not None:
            if type == 'sensor':
                # Sensor cooler refers to the TE cooler on sCMOS sensor.
                self.camera.SensorCooling = cooling
            elif type == 'fan':
                self.camera.FanSpeed = 'On' if cooling else 'Off'

    def get_cooling_status(self):
        """Get camera cooling status, including fan status, sensor cooler status, sensor cooling status, and sensor temperature."""

        if self.camera is not None:
            fan_status = self.camera.FanSpeed == 'On' # "on" or "Off"
            sensor_cooler_status = self.camera.SensorCooling == 1 # 1 or 0, Sensor cooler refers to the TE cooler on sCMOS sensor.
            sensor_cooling_staus = self.camera.TemperatureStatus # options: Cooler off, Stabilised, Cooling, Drift, Not Stabilised, Fault
            sensor_temp = self.camera.SensorTemperature

            return fan_status, sensor_cooler_status, sensor_cooling_staus, sensor_temp
        
        else:
            return None, None, None, None


logging.getLogger().setLevel("INFO")
cam = AndorZL41Wave(None)

