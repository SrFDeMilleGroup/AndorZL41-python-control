from pyAndorSDK3 import AndorSDK3
import numpy as np
import logging, time

from .dummy_camera import DummyCamera

class AndorZL41Wave:
    def __init__(self, parent):
        self.parent = parent

        try:
            cam_index = 0
            raise NotImplementedError("AndorZL41Wave is not implemented yet.")
            self.sdk3 = AndorSDK3()
            logging.info(f"Using SDK version {self.sdk3.SoftwareVersion}.")
            logging.info("Found {:d} camera(s) in the system.".format(self.sdk3.DeviceCount))

            cam_index = 0
            logging.info("Connecting to camera of index = {:d} ...".format(cam_index))

            self.camera = self.sdk3.GetCamera(cam_index)
            logging.info("Connected to camera of index = {:d}.".format(cam_index))
            logging.info(f"Camera model: {self.camera.CameraModel}. Serial number: {self.camera.SerialNumber}. Interface: {self.camera.InterfaceType}.")
            logging.info(f"Firmware version: {self.camera.FirmwareVersion}.")

        except Exception as err:
            logging.error(f"Failed to connect to camera of index = {cam_index}.")
            logging.error(err)
            logging.error("Using a dummy camera instance instead.")
            self.camera = DummyCamera()

        self.init_cam()

    def init_cam(self):
        self.camera.AOILayout = 'Image' # default setting is also image, this step can be skipped
        self.camera.ShutterMode = 'Closed' # shutter closed until exposure is triggered

        # below are constants for Andor ZL41 Wave 5.5
        self.cam_sensor_size_horizontal = 2560
        self.cam_sensor_size_vertical = 2160
        self.cam_binning_horizontal_min = 1
        self.cam_binning_horizontal_max = 640
        self.cam_binning_vertical_min = 1
        self.cam_binning_vertical_max = 2160

    def close(self):
        self.camera.close()
        logging.info("Camera closed.")
            
    def set_AOI_binning(self, direction: str, binning: int, centered: bool) -> tuple:
        """
        Set binning in AOI.
        """

        assert direction in ['horizontal', 'vertical']
        assert type(binning) == int
        assert type(centered) == bool

        if direction == 'horizontal':
            new_binning = np.clip(binning, self.cam_binning_horizontal_min, self.cam_binning_horizontal_max)
            if new_binning != binning:
                logging.warning(f"Invalid {direction} binning value {binning}. Round it to {new_binning} instead.")
                binning = new_binning
            self.camera.AOIHBin = binning
            aoi_binning = self.camera.AOIHBin
            aoi_width = self.cemera.AOIWidth # AOI size could be overwritten by binning setting, so read out again
            aoi_width_min = self.cemera.min_AOIWidth
            aoi_width_max = self.cemera.max_AOIWidth
            aoi_left = self.cemera.AOILeft - 1 # AOI start position could be overwritten by binning setting, so read out again
            aoi_left_min = self.cemera.min_AOILeft - 1
            aoi_left_max = self.cemera.max_AOILeft - 1
            if centered:
                AOI_start = int((self.cam_sensor_size_horizontal - aoi_width * aoi_binning) / 2)
                AOI_start = np.clip(AOI_start, aoi_width_min, aoi_width_max)
                self.cemera.AOILeft = AOI_start + 1
                aoi_left = self.cemera.AOILeft - 1
                if AOI_start != aoi_left:
                    logging.error(f"Failed to center AOI horizontally. AOI left should be {AOI_start}, but current value is {aoi_left}.")

        else: 
            # direction = 'vertical'
            new_binning = np.clip(binning, self.cam_binning_vertical_min, self.cam_binning_vertical_max)
            if new_binning != binning:
                logging.warning(f"Invalid {direction} binning value {binning}. Round it to {new_binning} instead.")
                binning = new_binning
            self.camera.AOIVBin = binning
            aoi_binning = self.camera.AOIVBin
            aoi_height = self.cemera.AOIHeight # AOI size could be overwritten by binning setting, so read out again
            aoi_height_min = self.cemera.min_AOIHeight
            aoi_height_max = self.cemera.max_AOIHeight
            aoi_top = self.cemera.AOITop - 1 # AOI start position could be overwritten by binning setting, so read out again
            aoi_top_min = self.cemera.min_AOITop - 1
            aoi_top_max = self.cemera.max_AOITop - 1
            # camera will automatically vertically center AOI, if self.camera.VerticallyCenterAOI == True

        if aoi_binning != binning:
            logging.error(f"Failed to set {direction} binning to {binning}. Current value is {aoi_binning}.")

        if direction == 'horizontal':
            return aoi_binning, aoi_width, aoi_width_min, aoi_width_max, aoi_left, aoi_left_min, aoi_left_max, aoi_binning == binning
        else:
            # direction = 'vertical'
            return aoi_binning, aoi_height, aoi_height_min, aoi_height_max, aoi_top, aoi_top_min, aoi_top_max, aoi_binning == binning

    def set_AOI_size(self, direction: str, size: int, centered: bool) -> tuple:
        """
        Set AOI size in unit of binned pixels.
        """

        assert direction in ['horizontal', 'vertical']
        assert type(size) == int
        assert type(centered) == bool

        if direction == 'horizontal':
            new_size = np.clip(size, self.camera.min_AOIWidth, self.camera.max_AOIWidth)
            if new_size != size:
                logging.warning(f"Invalid {direction} AOI size {size}. Round it to {new_size} instead.")
                size = new_size
            self.camera.AOIWidth = size
            aoi_width = self.camera.AOIWidth
            aoi_width_min = self.camera.min_AOIWidth
            aoi_width_max = self.camera.max_AOIWidth
            aoi_left = self.cemera.AOILeft - 1 # AOI start position could be overwritten by size setting, so read out again
            aoi_left_min = self.cemera.min_AOILeft - 1
            aoi_left_max = self.cemera.max_AOILeft - 1
            if centered:
                AOI_start = int((self.cam_sensor_size_horizontal - aoi_width * self.camera.AOIHBin) / 2)
                AOI_start = np.clip(AOI_start, aoi_left_min, aoi_left_max)
                self.cemera.AOILeft = AOI_start + 1
                aoi_left = self.cemera.AOILeft - 1
                if AOI_start != aoi_left:
                    logging.error(f"Failed to center AOI horizontally. AOI left should be {AOI_start}, but current value is {aoi_left}.")

        else:
            # direction = 'vertical'
            new_size = np.clip(size, self.camera.min_AOIHeight, self.camera.max_AOIHeight)
            if new_size != size:
                logging.warning(f"Invalid {direction} AOI size {size}. Round it to {new_size} instead.")
                size = new_size
            self.camera.AOIHeight = size
            aoi_height = self.camera.AOIHeight
            aoi_height_min = self.camera.min_AOIHeight
            aoi_height_max = self.camera.max_AOIHeight
            aoi_top = self.cemera.AOITop - 1 # AOI start position could be overwritten by size setting, so read out again
            aoi_top_min = self.cemera.min_AOITop - 1
            aoi_top_max = self.cemera.max_AOITop - 1
            # camera will automatically vertically center AOI, if self.camera.VerticallyCenterAOI == True

        if direction == 'horizontal':
            if aoi_width != size:
                logging.error(f"Failed to set {direction} AOI size to {size}. Current value is {aoi_width}.")
            return aoi_width, aoi_width_min, aoi_width_max, aoi_left, aoi_left_min, aoi_left_max, aoi_width == size
        else:
            # direction = 'vertical'
            if aoi_height != size:
                logging.error(f"Failed to set {direction} AOI size to {size}. Current value is {aoi_height}.")
            return aoi_height, aoi_height_min, aoi_height_max, aoi_top, aoi_top_min, aoi_top_max, aoi_height == size
        
    def set_AOI_start_index(self, direction: str, index: int, centered: bool) -> int:
        """
        Set AOI starting pixel (top-most pixel for vertical direciton and left-most pixel for horizontal direciton ) index in unit of unbinned pixels.
        Pixels are counted from 0.

        Andor cameras count pixels from 1, so the input index should be 1 less than the actual index. This is compensated below.
        """

        assert direction in ['horizontal', 'vertical']
        assert type(index) == int
        assert centered == False # cannot set AOI start index when AOI is centered

        if direction == 'horizontal':
            new_index = np.clip(index, self.camera.min_AOILeft - 1, self.camera.max_AOILeft - 1)
            if new_index != index:
                logging.warning(f"Invalid {direction} AOI starting pixel index {index}. Round it to {new_index} instead.")
                index = new_index
            self.camera.AOILeft = index + 1
            aoi_left = self.camera.AOILeft - 1
        else:
            # direction = 'vertical'
            new_index = np.clip(index, self.camera.min_AOITop - 1, self.camera.max_AOITop - 1)
            if new_index != index:
                logging.warning(f"Invalid {direction} AOI starting pixel index {index}. Round it to {new_index} instead.")
                index = new_index
            self.camera.AOITop = index + 1
            aoi_top = self.camera.AOITop - 1

        if direction == 'horizontal':
            if aoi_left != index:
                logging.error(f"Failed to set {direction} AOI starting pixel index to {index}. Current value is {aoi_left}.")
            return aoi_left, aoi_left != index
        else:
            # direction = 'vertical'
            if aoi_top != index:
                logging.error(f"Failed to set {direction} AOI starting pixel index to {index}. Current value is {aoi_top}.")
            return aoi_top, aoi_top != index
        
    def set_AOI_centered(self, direction: str, centered: bool) -> tuple:
        """
        Set AOI to be centered in the sensor or not.      
        """

        assert direction in ['horizontal', 'vertical']
        assert type(centered) == bool
        
        if direction == 'horizontal':
            if centered:
                AOI_start = int((self.cam_sensor_size_horizontal - self.camera.AOIWidth * self.camera.AOIHBin) / 2)
                aoi_left, success = self.set_AOI_start_index('horizontal', AOI_start, centered=False)
                actual_centered = True if success else False
            else:
                aoi_left = self.camera.AOILeft - 1
                success = True
                actual_centered = False
        else:
            # direction = 'vertical'
            self.camera.VerticallyCenterAOI = centered
            actual_centered = self.camera.VerticallyCenterAOI
            aoi_top = self.cemera.AOITop - 1
            success = actual_centered == centered

        if direction == 'horizontal':
            if not success:
                logging.warning(f"Failed to set {direction} AOI centered to {centered}." + \
                                f"{direction} AOI starting pixel index is {aoi_left}.")
            return aoi_left, actual_centered, success
        else:
            # direction = 'vertical'
            if not success:
                logging.warning(f"Failed to set {direction} AOI centered to {centered}." + \
                                f"{direction} AOI starting pixel index is {aoi_top}.")
            return aoi_top, actual_centered, success
        
    def set_shutter_mode(self, mode: str) -> tuple:
        """
        Set shutter mode.
        """

        assert mode in ['Rolling', 'Global']
        
        self.camera.ElectronicShutteringMode = mode
        mode_actual = self.camera.ShutterMode

        if mode_actual != mode:
            logging.error(f"Failed to set shutter mode to {mode}. Current value is {mode_actual}.")
            return (mode_actual, False)
        else:  
            return (mode_actual, True)
        
    def set_trigger_mode(self, mode: str) -> tuple:
        """
        Set trigger mode.
        """

        assert mode in ['Internal', 'Software', 'External', 'External Start', 'External Exposure']
        
        self.camera.TriggerMode = mode
        mode_actual = self.camera.TriggerMode

        if mode_actual != mode:
            logging.error(f"Failed to set trigger mode to {mode}. Current value is {mode_actual}.")
            return (mode_actual, False)
        else:
            return (mode_actual, True)
        
    def set_exposure_overlap(self, overlap: bool) -> tuple:
        """
        Set exposure overlap.
        """

        assert type(overlap) == bool
        
        self.camera.Overlap = overlap
        overlap_actual = bool(self.camera.Overlap)

        if overlap_actual != overlap:
            logging.error(f"Failed to set exposure overlap to {overlap}. Current value is {overlap_actual}.")
            return (overlap_actual, False)
        else:
            return (overlap_actual, True)
        
    def set_exposure_time(self, time: float) -> float:
        """
        Set exposure time in unit of ms.
        """

        assert type(time) == float
        
        new_time = np.clip(time, self.camera.min_ExposureTime * 1e3, self.camera.max_ExposureTime * 1e3) # convert from s to ms
        if new_time != time:
            logging.warning(f"Invalid exposure time {time} ms. Round it to {new_time} ms instead.")
            time = new_time
        
        self.camera.ExposureTime = time / 1e3 # convert from ms to s
        time_actual = self.camera.ExposureTime * 1e3 # convert from s to ms

        if abs(time_actual - time) > 5e-6:
            logging.error(f"Failed to set exposure time to {time}. Current value is {time_actual}.")
            return time_actual, False
        else:
            return time_actual, True
        
    def set_pixel_readout_rate(self, rate: str) -> tuple:
        """
        Set pixel readout rate.
        """

        assert rate in ['100 MHz', '280 MHz']
        
        self.camera.PixelReadoutRate = rate
        rate_actual = self.camera.PixelReadoutRate

        if rate_actual != rate:
            logging.error(f"Failed to set pixel readout rate to {rate}. Current value is {rate_actual}.")
            return (rate_actual, False)
        else:
            return (rate_actual, True)
        
    def read_exposre_time_range(self) -> tuple:
        """
        Read out minimum exposure time in unit of ms.
        """


        return (self.camera.min_ExposureTime * 1e3, self.camera.max_ExposureTime * 1e3)

    def read_readout_time(self, type: str) -> float:
        """
        Read out readout time in unit of ms or us.
        """

        assert type in ['frame', 'row']
        
        if type == 'frame':
            return self.camera.ReadoutTime * 1e3 # convert from s to ms
        else:
            # type == 'row'
            return self.camera.RowReadTime * 1e6 # convert from s to us        

    def set_pre_amp_gain(self, gain: str) -> tuple:
        """
        Set pre-amplifier gain.
        """

        assert gain in ['16-bit (low noise & high well capacity)', '12-bit (high well capacity)', '12-bit (low noise)']
        
        self.camera.SimplePreAmpGainControl = gain
        gain_actual = self.camera.SimplePreAmpGainControl

        if gain_actual != gain:
            logging.error(f"Failed to set pre-amplifier gain to {gain}. Current value is {gain_actual}.")
            return (gain_actual, self.camera.PixelEncoding, False) # pre amp gain automatically changes pixel encoding
        else:
            return (gain_actual, self.camera.PixelEncoding, True)
        
    def set_pixel_encoding(self, encoding: str) -> tuple:
        """
        Set pixel encoding.
        """

        assert encoding in ['Mono12', 'Mono12Packed', 'Mono16', 'Mono32']
        
        self.camera.PixelEncoding = encoding
        encoding_actual = self.camera.PixelEncoding

        if encoding_actual != encoding:
            logging.error(f"Failed to set pixel encoding to {encoding}. Current value is {encoding_actual}.")
            return (encoding_actual, False)
        else:
            return (encoding_actual, True)
        
    def read_image_size(self) -> float:
        """Read out image size in unit of kilobytes."""

        return self.camera.ImageSizeBytes / 1e3
    
    def read_interface_transfer_rate(self) -> float:
        """Read out interface (Camera link or USB) transfer rate in unit of frames per second."""
 
        return self.camera.InterfaceTransferRate
    
    def read_image_baseline(self) -> int:
        """Read out the baseline of each pixel reading."""

        return self.camera.Baseline
    
    def enable_noise_filter(self, filter_type: str,  enable: bool) -> tuple:
        """
        Turn on or off camera noise filter.
        """

        assert filter_type in ['spurious', 'blemish']
        assert type(enable) == bool
        
        if filter_type == 'spurious':
            self.camera.SpuriousNoiseFilter = enable
            current_value = bool(self.camera.SpuriousNoiseFilter)
            if current_value != enable:
                logging.error(f"Failed to set spurious noise filter to {enable}. Current value is {current_value}.")
                return (current_value, False)
            else:
                return (current_value, True)
            
        elif filter_type == 'blemish':
            self.camera.StaticBlemishCorrection = enable
            current_value = bool(self.camera.StaticBlemishCorrection)
            if current_value != enable:
                logging.error(f"Failed to set blemish noise filter to {enable}. Current value is {current_value}.")
                return (current_value, False)
            else:
                return (current_value, True)

    def set_auxiliary_output(self, aux_index: int, aux_output:str) -> tuple:
        """
        Set auxiliary output signal.
        """

        assert aux_index in [1, 2]
        if aux_index == 1:
            assert aux_output in ['FireRow1', 'FireRowN', 'FireAll', 'FireAny']
        else:
            assert aux_output in ['ExternalShutterControl', 'FrameClock', 'RowClock', 'ExposedRowClock']
        
        if aux_index == 1:
            self.camera.AuxiliaryOutSource = aux_output
            aux_output_actual = self.camera.AuxiliaryOutSource
        else:
            # aux_index == 2
            self.camera.AuxOutSourceTwo = aux_output
            aux_output_actual = self.camera.AuxOutSourceTwo

        if aux_output_actual != aux_output:
            logging.error(f"Failed to set auxiliary output signal {aux_index} to {aux_output}. Current value is {aux_output_actual}.")
            return (aux_output_actual, False)
        else:
            return (aux_output_actual, True)

    def enable_cooler(self, cooler_type: str, enable: bool) -> tuple:
        """
        Turn on or off camera sensor cooling.
        """

        assert cooler_type in ['sensor', 'fan']
        assert type(enable) == bool
        
        if cooler_type == 'sensor':
            # Sensor cooler refers to the TE cooler on sCMOS sensor.
            self.camera.SensorCooling = enable
            current_value = bool(self.camera.SensorCooling)

            if current_value != enable:
                logging.error(f"Failed to set sensor cooler to {enable}. Current value is {current_value}.")
                return (current_value, False)
            else:
                return (current_value, True)
            
        elif cooler_type == 'fan':
            self.camera.FanSpeed = 'On' if enable else 'Off'
            current_value = self.camera.FanSpeed == 'On'

            if current_value != enable:
                logging.error(f"Failed to set fan to {enable}. Current value is {current_value}.")
                return (current_value, False)
            else:
                return (current_value, True)

    def read_cooling_status(self) -> tuple:
        """Get camera cooling status, including fan status, sensor cooler status, sensor cooling status, and sensor temperature."""

        fan_status = self.camera.FanSpeed == 'On' # "on" or "Off"
        sensor_cooler_status = self.camera.SensorCooling == 1 # 1 or 0, Sensor cooler refers to the TE cooler on sCMOS sensor.
        sensor_cooling_status = self.camera.TemperatureStatus # options: Cooler off, Stabilised, Cooling, Drift, Not Stabilised, Fault
        sensor_temp = self.camera.SensorTemperature

        return fan_status, sensor_cooler_status, sensor_cooling_status, sensor_temp

# logging.getLogger().setLevel("INFO")
# cam = AndorZL41Wave(None)

