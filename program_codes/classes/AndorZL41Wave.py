from pyAndorSDK3 import AndorSDK3
import numpy as np
import logging, multiprocessing

from .dummy_camera import DummyCamera


def worker():
    # try open sdk3 once
    # in the case where the camera PCIe card is not installed, the program will be stuck here.
    sdk3 = AndorSDK3()

class AndorZL41Wave:
    def __init__(self, parent):
        self.parent = parent

        try:
            cam_index = 0
            timeout = 30 # in seconds
            
            # follow https://stackoverflow.com/a/14924210
            # and https://stackoverflow.com/a/37736655
            # to set a timeout for opening Andor SDK3.
            # But for some readon we can't just use the sdk3 object opened in the worker process, but have to create a new one later.
            # If the camera PCIe card is not installed, then opening Andor SDK3 will take forever.
            p = multiprocessing.Process(target=worker, args=())
            p.start()
            p.join(timeout)
            if p.is_alive():
                p.terminate() # stop the process because it times out
                p.join()
                raise Exception(f"Timeout for opening ANdor SDK3 (timeout = {timeout} seconds). Please check if the camera or the PCIe Camera Link card are installed.")
                        
            self.sdk3 = AndorSDK3() # If the above exception doesn't raise, it means the SDK3 can be successfully opened.
            logging.info(f"Using SDK version {self.sdk3.SoftwareVersion}.")
            logging.info("Found {:d} camera(s) in the system.".format(self.sdk3.DeviceCount))

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
        self.AccumulateCount = 1 # number of images to be summed to get each image

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

        assert direction in ['horizontal', 'vertical'], f"Invalid AOI binning direction {direction}. Direction can only be horizontal or vertical."
        assert type(binning) == int, f"Invalid AOI binning value type {type(binning)}. Binning should be an integer."
        assert type(centered) == bool, f"Invalid AOI binning centered value type {type(centered)}. Centered should be a boolean."

        if direction == 'horizontal':
            new_binning = np.clip(binning, self.cam_binning_horizontal_min, self.cam_binning_horizontal_max)
            if new_binning != binning:
                logging.warning(f"Invalid {direction} binning value {binning}. Round it to {new_binning} instead.")
                binning = new_binning
            self.camera.AOIHBin = binning
            aoi_binning = self.camera.AOIHBin
            aoi_width = self.camera.AOIWidth # AOI size could be overwritten by binning setting, so read out again
            aoi_width_min = self.camera.min_AOIWidth
            aoi_width_max = self.camera.max_AOIWidth
            aoi_left = self.camera.AOILeft - 1 # AOI start position could be overwritten by binning setting, so read out again
            aoi_left_min = self.camera.min_AOILeft - 1
            aoi_left_max = self.camera.max_AOILeft - 1
            if centered:
                AOI_start = int((self.cam_sensor_size_horizontal - aoi_width * aoi_binning) / 2)
                AOI_start = np.clip(AOI_start, aoi_left_min, aoi_left_max)
                self.camera.AOILeft = AOI_start + 1
                aoi_left = self.camera.AOILeft - 1
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
            aoi_height = self.camera.AOIHeight # AOI size could be overwritten by binning setting, so read out again
            aoi_height_min = self.camera.min_AOIHeight
            aoi_height_max = self.camera.max_AOIHeight
            aoi_top = self.camera.AOITop - 1 # AOI start position could be overwritten by binning setting, so read out again
            aoi_top_min = self.camera.min_AOITop - 1
            aoi_top_max = self.camera.max_AOITop - 1
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

        assert direction in ['horizontal', 'vertical'], f"Invalid AOI size direction {direction}. Direction can only be horizontal or vertical."
        assert type(size) == int, f"Invalid AOI size value type {type(size)}. Size should be an integer."
        assert type(centered) == bool, f"Invalid AOI size centered value type {type(centered)}. Centered should be a boolean."

        if direction == 'horizontal':
            new_size = np.clip(size, self.camera.min_AOIWidth, self.camera.max_AOIWidth)
            if new_size != size:
                logging.warning(f"Invalid {direction} AOI size {size}. Round it to {new_size} instead.")
                size = new_size
            self.camera.AOIWidth = size
            aoi_width = self.camera.AOIWidth
            aoi_width_min = self.camera.min_AOIWidth
            aoi_width_max = self.camera.max_AOIWidth
            aoi_left = self.camera.AOILeft - 1 # AOI start position could be overwritten by size setting, so read out again
            aoi_left_min = self.camera.min_AOILeft - 1
            aoi_left_max = self.camera.max_AOILeft - 1
            if centered:
                AOI_start = int((self.cam_sensor_size_horizontal - aoi_width * self.camera.AOIHBin) / 2)
                AOI_start = np.clip(AOI_start, aoi_left_min, aoi_left_max)
                self.camera.AOILeft = AOI_start + 1
                aoi_left = self.camera.AOILeft - 1
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
            aoi_top = self.camera.AOITop - 1 # AOI start position could be overwritten by size setting, so read out again
            aoi_top_min = self.camera.min_AOITop - 1
            aoi_top_max = self.camera.max_AOITop - 1
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

        assert direction in ['horizontal', 'vertical'], f"Invalid AOI start index direction {direction}. Direction can only be horizontal or vertical."
        assert type(index) == int, f"Invalid AOI start index value type {type(index)}. Index should be an integer."
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
            return aoi_left, aoi_left == index
        else:
            # direction = 'vertical'
            if aoi_top != index:
                logging.error(f"Failed to set {direction} AOI starting pixel index to {index}. Current value is {aoi_top}.")
            return aoi_top, aoi_top == index
        
    def set_AOI_centered(self, direction: str, centered: bool) -> tuple:
        """
        Set AOI to be centered in the sensor or not.      
        """

        assert direction in ['horizontal', 'vertical'], f"Invalid AOI centered direction {direction}. Direction can only be horizontal or vertical."
        assert type(centered) == bool, f"Invalid AOI centered value type {type(centered)}. Centered should be a boolean."
        
        if direction == 'horizontal':
            if centered:
                AOI_start = int((self.cam_sensor_size_horizontal - self.camera.AOIWidth * self.camera.AOIHBin) / 2)
                aoi_left, success = self.set_AOI_start_index('horizontal', AOI_start, centered=False)
                actual_centered = success
            else:
                aoi_left = self.camera.AOILeft - 1
                success = True
                actual_centered = False
        else:
            # direction = 'vertical'
            self.camera.VerticallyCenterAOI = centered
            actual_centered = self.camera.VerticallyCenterAOI
            aoi_top = self.camera.AOITop - 1
            success = actual_centered == centered

        if direction == 'horizontal':
            if not success:
                logging.error(f"Failed to set {direction} AOI centered to {centered}." + \
                                f"{direction} AOI starting pixel index is {aoi_left}.")
            return aoi_left, actual_centered, success
        else:
            # direction = 'vertical'
            if not success:
                logging.error(f"Failed to set {direction} AOI centered to {centered}." + \
                                f"{direction} AOI starting pixel index is {aoi_top}.")
            return aoi_top, actual_centered, success
        
    def set_shutter_mode(self, mode: str) -> tuple:
        """
        Set shutter mode.
        """

        assert mode in ['Rolling', 'Global'], f"Invalid shutter mode {mode}. Mode can only be Rolling or Global."
        
        self.camera.ElectronicShutteringMode = mode
        mode_actual = self.camera.ElectronicShutteringMode

        if mode_actual != mode:
            logging.error(f"Failed to set shutter mode to {mode}. Current value is {mode_actual}.")
            return (mode_actual, False)
        else:  
            return (mode_actual, True)
        
    def set_trigger_mode(self, mode: str) -> tuple:
        """
        Set trigger mode.
        """

        assert mode in ['Internal', 'Software', 'External', 'External Start', 'External Exposure'], f"Invalid trigger mode {mode}. Mode can only be Internal, Software, External, External Start, or External Exposure."
        
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

        assert type(overlap) == bool, f"Invalid exposure overlap value type {type(overlap)}. Overlap should be a boolean."

        exist = self.read_overlap_writable()
        if not exist:
            # in this case, overlap is fixed to be false
            overlap_actual = False
        else:
            self.camera.Overlap = overlap
            overlap_actual = bool(self.camera.Overlap)

        if overlap_actual != overlap:
            logging.error(f"Failed to set exposure overlap to {overlap}. Current value is {overlap_actual}.")
            return (exist, overlap_actual, False)
        else:
            return (exist, overlap_actual, True)
        
    def read_overlap_writable(self) -> bool:
        """
        Read if exposure overlap is writable.
        """

        shutter_mode = self.camera.ElectronicShutteringMode
        assert shutter_mode in ["Rolling", "Global"], f"Invalid shutter mode {shutter_mode}. Mode can only be Rolling or Global."

        trigger_mode = self.camera.TriggerMode
        assert trigger_mode in ['Internal', 'Software', 'External', 'External Start', 'External Exposure'], f"Invalid trigger mode {trigger_mode}. Mode can only be Internal, Software, External, External Start, or External Exposure."

        if shutter_mode == "Rolling" and trigger_mode in ["External", "Software"]:
            # in this case, overlap is fixed to be false
            return False
        else:
            return True
        
    def set_exposure_time(self, time: float) -> float:
        """
        Set exposure time in unit of ms.
        """

        assert type(time) in [float, int], f"Invalid exposure time value type {type(time)}. Time should be a float or integer."
        
        new_time = np.clip(time, self.camera.min_ExposureTime * 1e3, self.camera.max_ExposureTime * 1e3) # convert from s to ms
        if new_time != time:
            logging.warning(f"Invalid exposure time {time} ms. Round it to {new_time} ms instead.")
            time = new_time
        
        self.camera.ExposureTime = time / 1e3 # convert from ms to s
        time_actual = self.camera.ExposureTime * 1e3 # convert from s to ms

        if abs(time_actual - time) > self.read_readout_time('row'): # in ms
            logging.error(f"Failed to set exposure time to {time} ms. Current value is {time_actual} ms.")
            return time_actual, False
        elif abs(time_actual - time) > 1e-3: # in ms
            return time_actual, False
        else:
            return time_actual, True
        
    def set_pixel_readout_rate(self, rate: str) -> tuple:
        """
        Set pixel readout rate.
        """

        assert rate in ['100 MHz', '280 MHz'], f"Invalid pixel readout rate {rate}. Rate can only be 100 MHz or 280 MHz."
        
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
    
    def read_long_exposure_time(self) -> tuple:
        """
        Read out long exposure time in unit of ms.
        """

        shutter_mode = self.camera.ElectronicShutteringMode
        assert shutter_mode in ["Rolling", "Global"], f"Invalid shutter mode {shutter_mode}. Mode can only be Rolling or Global."

        trigger_mode = self.camera.TriggerMode
        assert trigger_mode in ['Internal', 'Software', 'External', 'External Start', 'External Exposure'], f"Invalid trigger mode {trigger_mode}. Mode can only be Internal, Software, External, External Start, or External Exposure."

        if shutter_mode == "Rolling":
            # trigger_mode in ["Internal", "External", "Software", "External Exposure", "Exposure Start"]
            # In Internal or External Start trigger mode, although the manual distinguishes between short and long exposure, 
            # the camera does not make any practical difference, and LongExposureTransition returns 0. 
            # "exist" variable indicates if the camera distinguishes between short and long exposure.
            exist = False
        else:
            # Global Shutter mode
            if trigger_mode in ["Internal", "External Start", "External", "Software"]:
                # In this case, if Overlap is True, the camera will effectively only run on long exposure mode
                # In External trigger mode, although the manual doesn't explicitly say, the camera still distinguishes between short (expo delayed by ~ 1 frame readout time) and long exposure.
                exist = False if self.camera.Overlap else True
            else:
                # trigger_mode == "External Exposure"
                exist = False

        l = self.camera.LongExposureTransition * 1e3 # convert from s to ms

        return exist, l
    
    def read_trigger_delay(self, long_expo) -> tuple:
        """
        Read out external trigger delay in unit of ms.
        """

        trigger_mode = self.camera.TriggerMode
        assert trigger_mode in ['Internal', 'Software', 'External', 'External Start', 'External Exposure'], f"Invalid trigger mode {trigger_mode}. Mode can only be Internal, Software, External, External Start, or External Exposure."

        if trigger_mode not in ["External", "External Exposure"]:
            return None, None
        
        shutter_mode = self.camera.ElectronicShutteringMode
        assert shutter_mode in ["Rolling", "Global"], f"Invalid shutter mode {shutter_mode}. Mode can only be Rolling or Global."

        t = self.read_readout_time('row') # in ms
        if shutter_mode == "Rolling":
            # for both External and External Exposure trigger mode
            # return delay_min, delay_max
            return 0, t
        else:
            # Global Shutter mode
            if (not long_expo) and trigger_mode == "External" and (not self.camera.Overlap):
                # Because in this mode, expsure time can be shorter than frame reaout time.
                # In this case, the trigger delay should be set to frame reaout time.
                f = self.read_readout_time('frame')
                return f + t, f + t * 2
            elif trigger_mode == "External Exposure" and self.camera.Overlap:
                return 0, t       
            else:
                return t, 2 * t

    def read_readout_time(self, type: str) -> float:
        """
        Read out readout time in unit of ms.
        """

        assert type in ['frame', 'row'], f"Invalid readout time type {type}. Type can only be frame or row."
        
        if type == 'frame':
            return self.camera.ReadoutTime * 1e3 # convert from s to ms
        else:
            # type == 'row'
            return self.camera.RowReadTime * 1e3 # convert from s to ms        

    def set_pre_amp_gain(self, gain: str) -> tuple:
        """
        Set pre-amplifier gain.
        """

        assert gain in ['16-bit (low noise & high well capacity)', '12-bit (high well capacity)', '12-bit (low noise)'], f"Invalid pre-amplifier gain {gain}. Gain can only be 16-bit (low noise & high well capacity), 12-bit (high well capacity), or 12-bit (low noise)."
        
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

        assert encoding in ['Mono12', 'Mono12Packed', 'Mono16', 'Mono32'], f"Invalid pixel encoding {encoding}. Encoding can only be Mono12, Mono12Packed, Mono16, or Mono32."
        
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
 
        return self.camera.MaxInterfaceTransferRate
    
    def read_image_baseline(self) -> int:
        """Read out the baseline of each pixel reading."""

        return self.camera.Baseline
    
    def enable_noise_filter(self, filter_type: str,  enable: bool) -> tuple:
        """
        Turn on or off camera noise filter.
        """

        assert filter_type in ['spurious', 'blemish'], f"Invalid noise filter type {filter_type}. Type can only be spurious or blemish."
        assert type(enable) == bool, f"Invalid noise filter enable value type {type(enable)}. Enable should be a boolean."
        
        if filter_type == 'spurious':
            self.camera.SpuriousNoiseFilter = enable
            current_value = bool(self.camera.SpuriousNoiseFilter)
            if current_value != enable:
                logging.error(f"Failed to set spurious noise filter to {enable}. Current value is {current_value}.")
                return (current_value, False)
            else:
                return (current_value, True)
            
        else:
            #  filter_type == 'blemish'
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

        assert aux_index in [1, 2], f"Invalid auxiliary output index {aux_index}. Index can only be 1 or 2."

        if aux_index == 1:
            assert aux_output in ['FireRow1', 'FireRowN', 'FireAll', 'FireAny'], f"Invalid auxiliary output signal {aux_output}. Signal can only be FireRow1, FireRowN, FireAll, or FireAny."
        else:
            assert aux_output in ['ExternalShutterControl', 'FrameClock', 'RowClock', 'ExposedRowClock'], f"Invalid auxiliary output signal {aux_output}. Signal can only be ExternalShutterControl, FrameClock, RowClock, or ExposedRowClock."
        
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

        assert cooler_type in ['sensor', 'fan'], f"Invalid cooler type {cooler_type}. Type can only be sensor or fan."
        assert type(enable) == bool, f"Invalid cooler enable value type {type(enable)}. Enable should be a boolean."
        
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
    
    def read_camera_param(self):
        """
        Generate a list of camera parameters that's read directly from the camera. 
        It can be used to compare if program settings are successfully applied to the camera.
        """
        
        cam_param = {}

        cam_param["SDK3 version"] = self.sdk3.SoftwareVersion
        cam_param["camera model"] = self.camera.CameraModel
        cam_param["serial number"] = self.camera.SerialNumber
        cam_param["interface type"] = self.camera.InterfaceType
        cam_param["firmware version"] = self.camera.FirmwareVersion

        cam_param["AOI layout"] = self.camera.AOILayout
        cam_param["shutter mode"] = self.camera.ShutterMode

        cam_param["AOI H binning"] = self.camera.AOIHBin
        cam_param["AOI width"] = self.camera.AOIWidth
        cam_param["AOI left (counting from 1)"] = self.camera.AOILeft
        cam_param["AOI V binning"] = self.camera.AOIVBin
        cam_param["AOI height"] = self.camera.AOIHeight
        cam_param["AOI top (counting from 1)"] = self.camera.AOITop
        cam_param["AOI V centered"] = self.camera.VerticallyCenterAOI

        cam_param["electronic shutter mode"] = self.camera.ElectronicShutteringMode
        cam_param["trigger mode"] = self.camera.TriggerMode
        cam_param["exposure overlap"] = self.camera.Overlap
        cam_param["exposure time (s)"] = self.camera.ExposureTime
        cam_param["long exposure transition (s)"] = self.camera.LongExposureTransition
        cam_param["pixel readout rate"] = self.camera.PixelReadoutRate
        cam_param["frame readout time (s)"] = self.camera.ReadoutTime
        cam_param["row readout time (s)"] = self.camera.RowReadTime

        cam_param["pre-amplifier gain"] = self.camera.SimplePreAmpGainControl
        cam_param["pixel encoding"] = self.camera.PixelEncoding
        cam_param["image size (Bytes)"] = self.camera.ImageSizeBytes
        cam_param["interface transfer rate (FPS)"] = self.camera.MaxInterfaceTransferRate
        cam_param["image baseline"] = self.camera.Baseline
        cam_param["spurious noise filter"] = self.camera.SpuriousNoiseFilter
        cam_param["blemish noise filter"] = self.camera.StaticBlemishCorrection

        cam_param["auxiliary output 1"] = self.camera.AuxiliaryOutSource
        cam_param["auxiliary output 2"] = self.camera.AuxOutSourceTwo

        cam_param["sensor cooler"] = self.camera.SensorCooling
        cam_param["fan cooling"] = self.camera.FanSpeed
        cam_param["sensor temperature (deg C)"] = self.camera.SensorTemperature
        cam_param["temperature status"] = self.camera.TemperatureStatus

        return cam_param
    
    def software_trigger(self):
        """
        Trigger camera exposure by software.
        """

        self.camera.SoftwareTrigger()

    def start_acquisition(self, cycle_mode: str, frame_count: int = 0, buffer_size: int = 25, save_metadata: bool = False):
        """
        Configure and start camera acquisition.
        In continuous mode, frame_count is ignored.
        In fixed mode, frame_count is necessary.
        """

        assert cycle_mode in ['Fixed', 'Continuous'], f"Invalid cycle mode {cycle_mode}. Mode can only be Fixed, Continuous."
        self.camera.CycleMode = cycle_mode
        actual_cycle_mode = self.camera.CycleMode
        if actual_cycle_mode != cycle_mode:
            logging.error(f"Failed to set cycle mode to {cycle_mode}. Current value is {actual_cycle_mode}.")

        if actual_cycle_mode == 'Fixed':
            assert type(frame_count) == int, f"Invalid frame count value type {type(frame_count)}. Frame count should be an integer."
            self.camera.FrameCount = np.clip(frame_count, self.camera.min_FrameCount, self.camera.max_FrameCount)
            actual_frame_count = self.camera.FrameCount
            if actual_frame_count != frame_count:
                logging.error(f"Failed to set frame count to {frame_count}. Current value is {actual_frame_count}.")

        assert type(buffer_size) == int, f"Invalid buffer size value type {type(buffer_size)}. Buffer size should be an integer."
        imgsize = self.camera.ImageSizeBytes
        for _ in range(0, buffer_size):
            buf = np.empty((imgsize,), dtype='B')
            self.camera.queue(buf, imgsize)

        assert type(save_metadata) == bool, f"Invalid save metadata value type {type(save_metadata)}. Save metadata should be a boolean."
        if save_metadata:
            self.camera.MetadataEnable = True
            self.camera.MetadataTimeStamp = True
        else:
            self.camera.MetadataEnable = False
            self.camera.MetadataTimeStamp = False
        actual_metadate_enable = self.camera.MetadataEnable
        actual_metadate_timestamp = self.camera.MetadataTimeStamp
        if actual_metadate_enable != save_metadata:
            logging.error(f"Failed to set metadata enable to {save_metadata}. Current value is {actual_metadate_enable}.")
        if actual_metadate_timestamp != save_metadata:
            logging.error(f"Failed to set metadata timestamp to {save_metadata}. Current value is {actual_metadate_timestamp}.")

        self.camera.AcquisitionStart()

    def read_buffer(self, circular_buffer: bool = True, image_size: int = 0, timeout: int = 10000):
        """
        Read out 1 image from the buffer.
        If circular_buffer is False, image_size (self.camera.ImageSizeBytes) argument is ignored.
        timeout is in milliseconds.
        """

        assert type(circular_buffer) == bool, f"Invalid circular buffer value type {type(circular_buffer)}. Circular buffer should be a boolean."
        assert type(timeout) == int, f"Invalid timeout value type {type(timeout)}. Timeout should be an integer."

        acq = self.camera.wait_buffer(timeout)

        if circular_buffer:
            assert type(image_size) == int, f"Invalid image size value type {type(image_size)}. Image size should be an integer."
            self.camera.queue(acq._np_data, image_size)

        return acq.image

    def stop_acquisition(self):
        """
        Stop camera acquisition.
        """

        self.camera.AcquisitionStop()
        self.camera.flush()


# logging.getLogger().setLevel("INFO")
# cam = AndorZL41Wave(None)
