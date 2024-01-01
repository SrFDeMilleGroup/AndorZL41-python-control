from pyAndorSDK3 import AndorSDK3
import numpy as np
import logging, time

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

    def init_AOI_settings(self):
        """Initialize AOI settings. When the cemera is just opened, the AOI settings initialized to the default values."""

        self.AOI_settings = {"horizontal": {"sensor_size": 2560, # in unbinned pixels
                                            "AOI_binning": 1, # in unbinned pixels
                                            "AOI_binning_min": 1, # in unbinned pixels
                                            "AOI_binning_max": 640, # in unbinned pixels
                                            "AOI_size": 2560, # in BINNED pixels
                                            "AOI_size_min": 4, # in BINNED pixels
                                            "AOI_size_max": 2560, # in BINNED pixels
                                            "AOI_start": 1, # in unbinned pixels, top-most pixel index
                                            "AOI_start_min": 1, # in unbinned pixels
                                            "AOI_start_max": 1, # in unbinned pixels
                                            "AOI_centered": False,
                                            },
                             "vertical":   {"sensor_size": 2160, # in unbinned pixels
                                            "AOI_binning": 1, # in unbinned pixels
                                            "AOI_binning_min": 1, # in unbinned pixels
                                            "AOI_binning_max": 2160, # in unbinned pixels
                                            "AOI_size": 2160, # in BINNED pixels
                                            "AOI_size_min": 8, # in BINNED pixels
                                            "AOI_size_max": 2160, # in BINNED pixels
                                            "AOI_start": 1, # in unbinned pixels, left-most pixel index
                                            "AOI_start_min": 1, # in unbinned pixels
                                            "AOI_start_max": 1, # in unbinned pixels
                                            "AOI_centered": False,
                                            }
                            }
        
    def load_settings(self):
        # load settigns
        pass

    def close(self):
        if self.camera is not None:
            self.camera.close()
            logging.info("Camera closed.")

    def enable_cooler(self, cooler_type: str, enable: bool) -> tuple:
        """
        Turn on or off camera sensor cooling.
        
        returns: (current cooler status: bool, success: bool)
        """

        assert cooler_type in ['sensor', 'fan']
        assert type(enable) == bool

        if self.camera is None:
            return (None, None)
        
        if cooler_type == 'sensor':
            # Sensor cooler refers to the TE cooler on sCMOS sensor.
            self.camera.SensorCooling = enable
            current_value = bool(self.camera.SensorCooling)
            if current_value != enable:
                logging.warning(f"Failed to set sensor cooler to {enable}. Current value is {current_value}.")
                return (current_value, False)
            else:
                return (current_value, True)
            
        elif cooler_type == 'fan':
            self.camera.FanSpeed = 'On' if enable else 'Off'
            current_value = self.camera.FanSpeed == 'On'
            if current_value != enable:
                logging.warning(f"Failed to set fan to {enable}. Current value is {current_value}.")
                return (current_value, False)
            else:
                return (current_value, True)

    def read_cooling_status(self) -> tuple:
        """Get camera cooling status, including fan status, sensor cooler status, sensor cooling status, and sensor temperature."""

        if self.camera is not None:
            fan_status = self.camera.FanSpeed == 'On' # "on" or "Off"
            sensor_cooler_status = self.camera.SensorCooling == 1 # 1 or 0, Sensor cooler refers to the TE cooler on sCMOS sensor.
            sensor_cooling_status = self.camera.TemperatureStatus # options: Cooler off, Stabilised, Cooling, Drift, Not Stabilised, Fault
            sensor_temp = self.camera.SensorTemperature

            return fan_status, sensor_cooler_status, sensor_cooling_status, sensor_temp
        
        else:
            return None, None, None, None

    def enable_noise_filter(self, filter_type: str,  enable: bool) -> tuple:
        """
        Turn on or off camera noise filter.
        
        returns: (current filter status: bool, success: bool)
        """

        assert filter_type in ['spurious', 'blemish']
        assert type(enable) == bool

        if self.camera is None:
            return (None, None)
        
        if filter_type == 'spurious':
            self.camera.SpuriousNoiseFilter = enable
            current_value = bool(self.camera.SpuriousNoiseFilter)
            if current_value != enable:
                logging.warning(f"Failed to set spurious noise filter to {enable}. Current value is {current_value}.")
                return (current_value, False)
            else:
                return (current_value, True)
            
        elif filter_type == 'blemish':
            self.camera.StaticBlemishCorrection = enable
            current_value = bool(self.camera.StaticBlemishCorrection)
            if current_value != enable:
                logging.warning(f"Failed to set blemish noise filter to {enable}. Current value is {current_value}.")
                return (current_value, False)
            else:
                return (current_value, True)
            
    def set_AOI_binning(self, direction: str, binning: int) -> tuple:
        """
        Set binning in AOI.
        
        returns: (current binning: int, success: bool)
        """

        assert direction in ['horizontal', 'vertical']
        assert type(binning) == int

        if self.camera is None:
            return (self.AOI_settings[direction]['AOI_binning'], None)
        
        new_binning = np.clip(binning, self.AOI_settings[direction]["AOI_binning_min"], self.AOI_settings[direction]["AOI_binning_max"])
        if new_binning != binning:
            logging.warning(f"Invalid {direction} binning value {binning}. Round it to {new_binning} instead.")
            binning = new_binning

        if direction == 'horizontal':
            self.camera.AOIHBin = binning
            self.AOI_settings['horizontal']['AOI_binning'] = self.camera.AOIHBin
            self.AOI_settings['horizontal']['AOI_size'] = self.cemera.AOIWidth # AOI size could be overwritten by binning setting, so read out again
            self.AOI_settings['horizontal']['min_AOI_size'] = self.cemera.min_AOIWidth
            self.AOI_settings['horizontal']['max_AOI_size'] = self.cemera.max_AOIWidth
            self.AOI_settings['horizontal']['AOI_start'] = self.cemera.AOILeft # AOI start position could be overwritten by binning setting, so read out again
            self.AOI_settings['horizontal']['min_AOI_start'] = self.cemera.min_AOILeft
            self.AOI_settings['horizontal']['max_AOI_start'] = self.cemera.max_AOILeft
            if self.AOI_settings['horizontal']['AOI_centered']:
                AOI_start = int((self.AOI_settings['horizontal']['sensor_size'] - self.AOI_settings['horizontal']['AOI_size'] * self.AOI_settings['horizontal']['AOI_binning']) / 2) + 1
                AOI_start = np.clip(AOI_start, self.AOI_settings['horizontal']['min_AOI_start'], self.AOI_settings['horizontal']['max_AOI_start'])
                self.cemera.AOILeft = AOI_start
                self.AOI_settings['horizontal']['AOI_start'] = self.cemera.AOILeft
        else: 
            # direction = 'vertical'
            self.camera.AOIVBin = binning
            self.AOI_settings['vertical']['AOI_binning'] = self.camera.AOIVBin
            self.AOI_settings['vertical']['AOI_size'] = self.cemera.AOIHeight # AOI size could be overwritten by binning setting, so read out again
            self.AOI_settings['vertical']['min_AOI_size'] = self.cemera.min_AOIHeight
            self.AOI_settings['vertical']['max_AOI_size'] = self.cemera.max_AOIHeight
            self.AOI_settings['vertical']['AOI_start'] = self.cemera.AOITop # AOI start position could be overwritten by binning setting, so read out again
            self.AOI_settings['vertical']['min_AOI_start'] = self.cemera.min_AOITop
            self.AOI_settings['vertical']['max_AOI_start'] = self.cemera.max_AOITop
            # camera will automatically vertically center AOI, if self.camera.VerticallyCenterAOI == True

        if self.AOI_settings[direction]['AOI_binning'] != binning:
            logging.warning(f"Failed to set {direction} binning to {binning}. Current value is {self.AOI_settings[direction]['AOI_binning']}.")
            return (self.AOI_settings[direction]['AOI_binning'], False)
        else:
            return (self.AOI_settings[direction]['AOI_binning'], True)

    def set_AOI_size(self, direction: str, size: int) -> tuple:
        """
        Set AOI size in unit of binned pixels.
        
        returns: (current AOI size: int, success: bool)
        """

        assert direction in ['horizontal', 'vertical']
        assert type(size) == int

        if self.camera is None:
            return (self.AOI_settings[direction]['AOI_size'], None)
        
        new_size = np.clip(size, self.AOI_settings[direction]["AOI_size_min"], self.AOI_settings[direction]["AOI_size_max"])
        if new_size != size:
            logging.warning(f"Invalid {direction} AOI size {size}. Round it to {new_size} instead.")
            size = new_size

        if direction == 'horizontal':
            self.camera.AOIWidth = size
            self.AOI_settings['horizontal']['AOI_size'] = self.camera.AOIWidth
            self.AOI_settings['horizontal']['min_AOI_size'] = self.camera.min_AOIWidth
            self.AOI_settings['horizontal']['max_AOI_size'] = self.camera.max_AOIWidth
            self.AOI_settings['horizontal']['AOI_start'] = self.cemera.AOILeft # AOI start position could be overwritten by size setting, so read out again
            self.AOI_settings['horizontal']['min_AOI_start'] = self.cemera.min_AOILeft
            self.AOI_settings['horizontal']['max_AOI_start'] = self.cemera.max_AOILeft
            if self.AOI_settings['horizontal']['AOI_centered']:
                AOI_start = int((self.AOI_settings['horizontal']['sensor_size'] - self.AOI_settings['horizontal']['AOI_size'] * self.AOI_settings['horizontal']['AOI_binning']) / 2) + 1
                AOI_start = np.clip(AOI_start, self.AOI_settings['horizontal']['min_AOI_start'], self.AOI_settings['horizontal']['max_AOI_start'])
                self.cemera.AOILeft = AOI_start
                self.AOI_settings['horizontal']['AOI_start'] = self.cemera.AOILeft
        else:
            # direction = 'vertical'
            self.camera.AOIHeight = size
            self.AOI_settings['vertical']['AOI_size'] = self.camera.AOIHeight
            self.AOI_settings['vertical']['min_AOI_size'] = self.camera.min_AOIHeight
            self.AOI_settings['vertical']['max_AOI_size'] = self.camera.max_AOIHeight
            self.AOI_settings['vertical']['AOI_start'] = self.cemera.AOITop # AOI start position could be overwritten by size setting, so read out again
            self.AOI_settings['vertical']['min_AOI_start'] = self.cemera.min_AOITop
            self.AOI_settings['vertical']['max_AOI_start'] = self.cemera.max_AOITop
            # camera will automatically vertically center AOI, if self.camera.VerticallyCenterAOI == True

        if self.AOI_settings[direction]['AOI_size'] != size:
            logging.warning(f"Failed to set {direction} AOI size to {size}. Current value is {self.AOI_settings[direction]['AOI_size']}.")
            return (self.AOI_settings[direction]['AOI_size'], False)
        else:
            return (self.AOI_settings[direction]['AOI_size'], True)
        
    def set_AOI_start_index(self, direction: str, index: int) -> tuple:
        """
        Set AOI starting pixel (top-most pixel for vertical direciton and left-most pixel for horizontal direciton ) index in unit of unbinned pixels.
        Pixels are counted from 1, rather than 0.
        
        returns: (current AOI starting pixel index: int, success: bool)
        """

        assert direction in ['horizontal', 'vertical']
        assert type(index) == int

        if self.camera is None:
            return (self.AOI_settings[direction]['AOI_start'], None)
        
        if self.AOI_settings[direction]['AOI_centered']:
            logging.warning(f"Cannot set {direction} AOI starting pixel index when AOI is set to centered.")
            return (self.AOI_settings[direction]['AOI_start'], False)
        
        new_index = np.clip(index, self.AOI_settings[direction]["AOI_start_min"], self.AOI_settings[direction]["AOI_start_max"])
        if new_index != index:
            logging.warning(f"Invalid {direction} AOI starting pixel index {index}. Round it to {new_index} instead.")
            index = new_index

        if direction == 'horizontal':
            self.camera.AOILeft = index
            self.AOI_settings['horizontal']['AOI_start'] = self.camera.AOILeft
        else:
            # direction = 'vertical'
            self.camera.AOITop = index
            self.AOI_settings['vertical']['AOI_start'] = self.camera.AOITop

        if self.AOI_settings[direction]['AOI_start'] != index:
            logging.warning(f"Failed to set {direction} AOI starting pixel index to {index}. Current value is {self.AOI_settings[direction]['AOI_start']}.")
            return (self.AOI_settings[direction]['AOI_start'], False)
        else:
            return (self.AOI_settings[direction]['AOI_start'], True)
        
    def set_AOI_centered(self, direction: str, centered: bool) -> tuple:
        """
        Set AOI to be centered in the sensor or not.

        returns: (current AOI centered status: bool, AOI starting pixel index: int, success: bool)        
        """

        assert direction in ['horizontal', 'vertical']
        assert type(centered) == bool

        if self.camera is None:
            return (self.AOI_settings[direction]['AOI_centered'], self.AOI_settings[direction]['AOI_start'], None)
        

        if direction == 'horizontal':
            if centered:
                AOI_start = int((self.AOI_settings['horizontal']['sensor_size'] - self.AOI_settings['horizontal']['AOI_size'] * self.AOI_settings['horizontal']['AOI_binning']) / 2) + 1
                _, success = self.set_AOI_start_index('horizontal', AOI_start)
                self.AOI_settings['horizontal']['AOI_centered'] = success
            else:
                self.AOI_settings['horizontal']['AOI_centered'] = False
        else:
            # direction = 'vertical'
            self.camera.VerticallyCenterAOI = centered
            self.AOI_settings['vertical']['AOI_centered'] = self.camera.VerticallyCenterAOI
            self.AOI_settings['vertical']['AOI_start'] = self.cemera.AOITop
            self.AOI_settings['vertical']['min_AOI_start'] = self.cemera.min_AOITop
            self.AOI_settings['vertical']['max_AOI_start'] = self.cemera.max_AOITop

        if self.AOI_settings[direction]['AOI_centered'] != centered:
            logging.warning(f"Failed to set {direction} AOI centered to {centered}. 
                            Current value is {self.AOI_settings[direction]['AOI_centered']}. 
                            {direction} AOI starting pixel index is {self.AOI_settings[direction]['AOI_start']}.")
            return (self.AOI_settings[direction]['AOI_centered'], self.AOI_settings[direction]['AOI_start'], False)
        else:
            return (self.AOI_settings[direction]['AOI_centered'], self.AOI_settings[direction]['AOI_start'], True)


# logging.getLogger().setLevel("INFO")
# cam = AndorZL41Wave(None)

