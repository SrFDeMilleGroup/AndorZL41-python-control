from PyQt5.QtCore import QTimer

# To make programming easier,
# the values of class data attributes are only changed when they are read out,
# but not when the correlated data attributes are changed.
# For example, changing SimplePreAmpGainControl could change PixelEncoding,
# but the value of PixelEncoding is only checked and changed when it is read out.
# The same also applies to AOI, exposure time and readout time.

# So when writing a setter function, simply do a sanity check on the input value, and set the value of the data attribute.
# But when writing a getter function, if the value of this attribute can be changed by other attributes,
# then the value of this attribute needs to be re-calculated.
class DummyCamera:
    def __init__(self):
        self.AOILayout = 'Image' # default setting is also image, this step can be skipped
        self.ShutterMode = 'Closed' # shutter closed until exposure is triggered

        self._sensor_width = 2560 # in unbinned pixels
        self._sensor_height = 2160 # in unbinned pixels
        self._AOIWidth_unbinned = 2560 # AOI in unbinned pixels
        self._AOIHeight_unbinned = 2160 # AOI in unbinned pixels
        self._AOIHBin = 1
        self._AOIVBin = 1
        self._AOIWidth = 2560
        self._AOIHeight = 2160
        self._AOILeft = 1
        self._AOITop = 1

        self.VerticallyCenterAOI = False

        self.ElectronicShutterMode = 'Rolling' # Rolling or Global
        self.TriggerMode = 'Internal' # Internal, External, External Start, External Exposure, Software
        self.Overlap = False
        self.PixelReadoutRate = '280 MHz' # 280 MHz or 100 MHz
        self.ExposureTime = 0.1 # in seconds

        self.SimplePreAmpGainControl = '16-bit (low noise & high well capacity)' # 16-bit (low noise & high well capacity), 12-bit (low noise) or 12-bit (high well capacity)
        self.PixelEncoding = 'Mono16' # Mono16, Mono12Packed, Mono12, Mono32

        self.SpuriousNoiseFilter = True
        self.StaticBlemishCorrection = True

        self.AuxiliaryOutSource = 'FireAll'
        self.AuxOutSourceTwo = 'ExternalShutterControl'

        self.SensorCooling = 0 # 0 or 1
        self.FanSpeed = 'Off' # "On" or "Off"
        self._TemperatureStatus = 'Cooler off' # Cooler off, Stabilised, Cooling, Drift, Not Stabilised, Fault
        self._SensorTemperature = 25

        self.temp_timer = QTimer()
        self.temp_timer.timeout.connect(self.update_temperature)
        self.temp_timer.start(1000) # in ms, time interval

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.temp_timer.stop()

    @property
    def AOILayout(self):
        return self._AOILayout
    
    @AOILayout.setter
    def AOILayout(self, layout):
        assert layout in ["Image", "Multitrack"]
        self._AOILayout = layout

    @property
    def ShutterMode(self):
        return self._ShutterMode
    
    @ShutterMode.setter
    def ShutterMode(self, mode):
        assert mode in ["Open", "Closed", "Auto"]
        self._ShutterMode = mode

    @property
    def AOIHBin(self):
        return self._AOIHBin
    
    @AOIHBin.setter
    def AOIHBin(self, bin):
        assert bin >= self.min_AOIHBin
        assert bin <= self.max_AOIHBin

        width_binned = int(self._AOIWidth_unbinned / bin)
        width_binned = max(self.min_AOIWidth, width_binned)
        self._AOIWidth_unbinned = width_binned * bin

        self._AOIHBin = bin

    @property
    def min_AOIHBin(self):
        self._min_AOIHBin = 1
        return self._min_AOIHBin
    
    @property
    def max_AOIHBin(self):
        self._max_AOIHBin = 640
        return self._max_AOIHBin

    @property
    def AOIVBin(self):
        return self._AOIVBin
    
    @AOIVBin.setter
    def AOIVBin(self, bin):
        assert bin >= self.min_AOIVBin
        assert bin <= self.max_AOIVBin

        height_binned = int(self._AOIHeight_unbinned / bin)
        height_binned = max(self.min_AOIHeight, height_binned)
        self._AOIHeight_unbinned = height_binned * bin

        self._AOIVBin = bin
    
    @property
    def min_AOIVBin(self):
        self._min_AOIVBin = 1
        return self._min_AOIVBin
    
    @property
    def max_AOIVBin(self):
        self._max_AOIVBin = self._sensor_height
        return self._max_AOIVBin
    
    @property
    def AOIWidth(self):
        # width can be changed by AOIHBin or AOIWidth setter method
        # in both method, the value of _AOIWidth_unbinned is changed
        # so there's no need to re-calculate _AOIWidth_unbinned here
        self._AOIWidth = int(self._AOIWidth_unbinned / self.AOIHBin)
        return self._AOIWidth
    
    @AOIWidth.setter
    def AOIWidth(self, width):
        assert width >= self.min_AOIWidth
        assert width <= self.max_AOIWidth

        self._AOIWidth_unbinned = width * self.AOIHBin
        self._AOIWidth = width

    @property
    def min_AOIWidth(self):
        self._min_AOIWidth = 4 # in 4 binned pixels, independent of AOIHBin
        return self._min_AOIWidth
    
    @property
    def max_AOIWidth(self):
        self._max_AOIWidth = int(self._sensor_width / self.AOIHBin) # Since max_AOIWidth is 640, this value if >= 4, which is min_AOIWidth
        return self._max_AOIWidth
    
    @property
    def AOIHeight(self):
        # height can be changed by AOIVBin or AOIHeight setter method
        # in both method, the value of _AOIHeight_unbinned is changed
        # so there's no need to re-calculate _AOIHeight_unbinned here
        self._AOIHeight = int(self._AOIHeight_unbinned / self.AOIVBin)
        return self._AOIHeight
    
    @AOIHeight.setter
    def AOIHeight(self, height):
        assert height >= self.min_AOIHeight
        assert height <= self.max_AOIHeight

        self._AOIHeight_unbinned = height * self.AOIVBin
        self._AOIHeight = height

    @property
    def min_AOIHeight(self):
        # min height is 8 unbinned pixels
        self._min_AOIHeight = int(8 / self.AOIVBin) + 1 # int() rounds down
        return self._min_AOIHeight
    
    @property
    def max_AOIHeight(self):
        self._max_AOIHeight = int(self._sensor_height / self.AOIVBin) 
        return self._max_AOIHeight
    
    @property
    def AOILeft(self):
        # AOILeft can be changed by AOIWidth, AOIHBin, or AOILeft setter method
        self._AOILeft = min(self._AOILeft, self._sensor_width - self.AOIWidth * self.AOIHBin + 1)
        return self._AOILeft
    
    @AOILeft.setter
    def AOILeft(self, left):
        assert left >= self.min_AOILeft
        assert left <= self.max_AOILeft

        self._AOILeft = left

    @property
    def min_AOILeft(self):
        self._min_AOILeft = 1
        return self._min_AOILeft
    
    @property
    def max_AOILeft(self):
        self._max_AOILeft = self._sensor_width - self.AOIWidth * self.AOIHBin + 1 # in unit of unbinned pixels
        return self._max_AOILeft
    
    @property
    def AOITop(self):
        if self.VerticallyCenterAOI:
            self._AOITop = int((self._sensor_height - self.AOIHeight * self.AOIVBin) / 2) + 1
        else:
            # AOITop can be changed by AOIHeight, AOIVBin, or AOITop setter method
            self._AOITop = min(self._AOITop, self._sensor_height - self.AOIHeight * self.AOIVBin + 1)
        
        return self._AOITop
    
    @AOITop.setter
    def AOITop(self, top):
        if self.VerticallyCenterAOI:
            raise Exception("AOI is vertically centered, AOITop is no more writable.")
        
        assert top >= self.min_AOITop
        assert top <= self.max_AOITop

        self._AOITop = top

    @property
    def min_AOITop(self):
        self._min_AOITop = 1
        return self._min_AOITop
    
    @property
    def max_AOITop(self):
        self._max_AOITop = self._sensor_height - self.AOIHeight * self.AOIVBin + 1 # in unit of unbinned pixels
        return self._max_AOITop
    
    @property
    def VerticallyCenterAOI(self):
        return self._VerticallyCenterAOI
    
    @VerticallyCenterAOI.setter
    def VerticallyCenterAOI(self, center):
        assert center in [True, False]
        self._VerticallyCenterAOI = center

    @property
    def ElectronicShutterMode(self):
        return self._ElectronicShutterMode
    
    @ElectronicShutterMode.setter
    def ElectronicShutterMode(self, mode):
        assert mode in ["Rolling", "Global"]
        self._ElectronicShutterMode = mode

    @property
    def TriggerMode(self):
        return self._TriggerMode
    
    @TriggerMode.setter
    def TriggerMode(self, mode):
        assert mode in ["Internal", "External", "External Start", "External Exposure", "Software"]
        self._TriggerMode = mode

    @property
    def Overlap(self):
        return self._Overlap
    
    @Overlap.setter
    def Overlap(self, overlap):
        assert overlap in [True, False]
        self._Overlap = overlap

    @property
    def PixelReadoutRate(self):
        return self._PixelReadoutRate
    
    @PixelReadoutRate.setter
    def PixelReadoutRate(self, rate):
        assert rate in ["280 MHz", "100 MHz"]
        self._PixelReadoutRate = rate

    @property
    def ExposureTime(self):
        self._ExposureTime = max(self.min_ExposureTime, self._ExposureTime)
        self._ExposureTime = min(self.max_ExposureTime, self._ExposureTime)

        return self._ExposureTime
    
    @ExposureTime.setter
    def ExposureTime(self, time):
        assert time >= self.min_ExposureTime
        assert time <= self.max_ExposureTime

        self._ExposureTime = time

    @property
    def min_ExposureTime(self):
        shutter = self.ElectronicShutterMode
        trigger = self.TriggerMode
        if shutter == "Global" and trigger == "External":
            self._min_ExposureTime = self.RowReadTime
        else:
            self._min_ExposureTime = self.ReadoutTime

        return self._min_ExposureTime

    @property
    def RowReadTime(self):
        sensor_width = self._sensor_width
        if self.PixelReadoutRate == "280 MHz":
            self._RowReadTime = sensor_width / 280e6 # in seconds
        else:
            # self.PixelReadoutRate == "100 MHz"
            self._RowReadTime = sensor_width / 100e6

        return self._RowReadTime
    
    @property
    def ReadoutTime(self):
        # pixels are counted from 1 to 2160 (rather than from 0 to 2159)
        # self.AOITop is the index of the top row of the AOI
        # bottom is the index of the bottom row of the AOI
        sensor_height = self._sensor_height
        top = self.AOITop
        height = self.AOIHeight * self.AOIVBin
        bottom = top + height - 1
        if bottom <= int(sensor_height / 2) or top >= int(sensor_height / 2) + 1:
            # the AOI is entirely on the top or bottom half of the sensor
            self._ReadoutTime = self.RowReadTime * height
        else:
            # top and bottom halves of the sensor are read out individually
            # readout time is limited by the half with the larger AOI
            rows = max(bottom - int(sensor_height / 2), int(sensor_height / 2) - top + 1)
            self._ReadoutTime = self.RowReadTime * rows

        return self._ReadoutTime

    @property
    def max_ExposureTime(self):
        self._max_ExposureTime = 30 # in seconds
        return self._max_ExposureTime

    @property
    def SimplePreAmpGainControl(self):
        return self._SimplePreAmpGainControl
    
    @SimplePreAmpGainControl.setter
    def SimplePreAmpGainControl(self, gain):
        assert gain in ["16-bit (low noise & high well capacity)", "12-bit (low noise)", "12-bit (high well capacity)"]
        self._SimplePreAmpGainControl = gain

    @property
    def PixelEncoding(self):
        gain = self.SimplePreAmpGainControl
        if gain == "16-bit (low noise & high well capacity)" and self._PixelEncoding in ["Mono12Packed", "Mono12"]:
            self._PixelEncoding = "Mono16"
        elif gain in ["12-bit (low noise)", "12-bit (high well capacity)"] and self._PixelEncoding == "Mono16":
            self._PixelEncoding = "Mono12"
        # else: either pixel encoding matches gain, or encoding is Mono32, no changes need to be made

        return self._PixelEncoding
    
    @PixelEncoding.setter
    def PixelEncoding(self, encoding):
        assert encoding in ["Mono16", "Mono12Packed", "Mono12", "Mono32"]
        self._PixelEncoding = encoding

    @property
    def ImageSizeBytes(self):
        encoding = self.PixelEncoding
        if encoding == "Mono12" or "Mono16":
            b = 2
        elif encoding == "Mono12Packed":
            b = 1.5
        else:
            # encoding == "Mono32"
            b = 4

        self._ImageSizeBytes = round(self.AOIWidth * self.AOIHeight * b) # in bytes
        return self._ImageSizeBytes
    
    @property
    def MaxInterfaceTransferRate(self):
        speed = 0.82944e9 # in bytes per second
        self._MaxInterfaceTransferRate = speed / self.ImageSizeBytes
        return self._InterfaceTransferRate

    @property
    def Baseline(self):
        self._Baseline = 100
        return self._Baseline

    @property
    def SpuriousNoiseFilter(self):
        return self._SpuriousNoiseFilter
    
    @SpuriousNoiseFilter.setter
    def SpuriousNoiseFilter(self, filter):
        assert filter in [True, False]
        self._SpuriousNoiseFilter = filter

    @property
    def StaticBlemishCorrection(self):
        return self._StaticBlemishCorrection
    
    @StaticBlemishCorrection.setter
    def StaticBlemishCorrection(self, correction):
        assert correction in [True, False]
        self._StaticBlemishCorrection = correction

    @property
    def AuxiliaryOutSource(self):
        return self._AuxiliaryOutSource

    @AuxiliaryOutSource.setter
    def AuxiliaryOutSource(self, source):
        assert source in ['FireRow1', 'FireRowN', 'FireAll', 'FireAny']
        self._AuxiliaryOutSource = source

    @property
    def AuxOutSourceTwo(self):
        return self._AuxOutSourceTwo
    
    @AuxOutSourceTwo.setter
    def AuxOutSourceTwo(self, source):
        assert source in ['ExternalShutterControl', 'FrameClock', 'RowClock', 'ExposedRowClock']
        self._AuxOutSourceTwo = source

    @property
    def SensorCooling(self):
        return self._SensorCooling
    
    @SensorCooling.setter
    def SensorCooling(self, cooling):
        assert cooling in [0, 1]
        self._SensorCooling = cooling

    @property
    def FanSpeed(self):
        return self._FanSpeed
    
    @FanSpeed.setter
    def FanSpeed(self, speed):
        assert speed in ["Off", "On"]
        self._FanSpeed = speed

    @property
    def TemperatureStatus(self):
        return self._TemperatureStatus
    
    @property
    def SensorTemperature(self):
        return self._SensorTemperature
    
    def update_temperature(self):
        min_temp = 0
        max_temp = 25
        if self.SensorCooling == 1:
            self._SensorTemperature -= 1
            self._TemperatureStatus = 'Cooling'
            if self._SensorTemperature <= min_temp:
                self._SensorTemperature = min_temp
                self._TemperatureStatus = 'Stabilised'
        else:
            self._SensorTemperature += 1
            self._TemperatureStatus = 'Cooler off'
            if self._SensorTemperature >= max_temp:
                self._SensorTemperature = max_temp