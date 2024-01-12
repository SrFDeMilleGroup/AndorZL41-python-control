class DummyCamera:
    def __init__(self):
        self.AOILayout = 'Image' # default setting is also image, this step can be skipped
        self.ShutterMode = 'Closed' # shutter closed until exposure is triggered

        self.SensorCooling = 0 # 0 or 1
        self.FanSpeed = 'Off' # "On" or "Off"
        self.TemperatureStatus = 'Cooler off' # Cooler off, Stabilised, Cooling, Drift, Not Stabilised, Fault
        self.SensorTemperature = 20

        self.SpuriousNoiseFilter = True
        self.StaticBlemishCorrection = True

        self.AOIHBin = 1
        self.min_AOIHBin = 1
        self.max_AOIHBin = 640
        self.AOIVBin = 1
        self.min_AOIVBin = 1
        self.max_AOIVBin = 2160
        self.AOIWidth = 2560
        self.min_AOIWidth = 1
        self.max_AOIWidth = 2560
        self.AOIHeight = 2160
        self.min_AOIHeight = 1
        self.max_AOIHeight = 2160
        self.AOILeft = 1
        self.min_AOILeft = 1
        self.max_AOILeft = 2560
        self.AOITop = 1
        self.min_AOITop = 1
        self.max_AOITop = 2160
        self.VerticallyCenterAOI = False

        self.ElectronicShutterMode = 'Rolling' # Rolling or Global
        self.TriggerMode = 'Internal' # Internal, External, External Start, External Exposure, Software
        self.Overlap = False
        self.PixelReadoutRate = '280 MHz' # 280 MHz or 100 MHz
        self.ExposureTime = 0.001 # in seconds
        self.min_ExposureTime = 1e-5 # in seocnds
        self.max_ExposureTime = 30 # in seconds
        self.ReadoutTime = 0.010 # in seconds, frame readout time
        self.RowReadTime = 1e-5 # in seconds, row readout time

        self.SimplePreAmpGainControl = '16-bit (low noise & high well capacity)' # 16-bit (low noise & high well capacity), 12-bit (low noise) or 12-bit (high well capacity)
        self.PixelEncoding = 'Mono16' # Mono16, Mono12Packed, Mono12, Mono12, Mono32
        self.Baseline = 100 # image baseline
        self.ImageSizeBytes = 11059200
        self.InterfaceTransferRate = 75 # frames per second

    def close(self):
        pass