from collections import deque
import configparser, h5py, time, logging, os
import PyQt5.QtWidgets as qt
import PyQt5.QtCore
import numpy as np

from ..widgets import Scrollarea, NewSpinBox, NewComboBox, NewDoubleSpinBox
from .AndorZL41Wave import AndorZL41Wave
from .TCP_thread import TCPThread


# the class that places elements in UI and handles data processing
class ControlGUI(Scrollarea):
    def __init__(self, parent):
        super().__init__(parent, label="", type="vbox")
        self.setMaximumWidth(400)
        self.frame.setContentsMargins(0,0,0,0)

        self.cam = AndorZL41Wave(self)

        # boolean variable, turned on when the camera starts to take images
        self.active = False

        # control mode, can be "record" or "scan" in current implementation
        self.control_mode = None

        # boolean variable, turned on when the TCP thread is started
        self.tcp_active = False

        # save signal count
        self.signal_count_deque = deque([], maxlen=20)

        # places GUI elements
        self.place_recording_control()
        self.place_image_control()
        self.place_cam_control()
        self.place_tcp_control()
        self.place_save_load_control()

        # don't start tcp thread here, 
        # it will be started when the program load latest setting (using load_settings(latest=true))
        
        # self.tcp_start()

    # place recording gui elements
    def place_recording_control(self):
        record_box = qt.QGroupBox("Recording")
        record_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        record_box.setMaximumHeight(270)
        record_frame = qt.QGridLayout()
        record_box.setLayout(record_frame)
        self.frame.addWidget(record_box)

        self.record_pb = qt.QPushButton("Record")
        self.record_pb.clicked[bool].connect(lambda val, mode="record": self.parent.start(mode))
        record_frame.addWidget(self.record_pb, 0, 0)
        self.record_pb.setEnabled(False)

        self.scan_pb = qt.QPushButton("Scan")
        self.scan_pb.clicked[bool].connect(lambda val, mode="scan": self.parent.start(mode))
        record_frame.addWidget(self.scan_pb, 0, 1)
        self.scan_pb.setEnabled(False)

        self.stop_pb = qt.QPushButton("Stop")
        self.stop_pb.clicked[bool].connect(lambda val: self.parent.stop())
        record_frame.addWidget(self.stop_pb, 0, 2)
        self.stop_pb.setEnabled(False)

        record_frame.addWidget(qt.QLabel("Measurement:"), 1, 0, 1, 1)
        self.meas_rblist = []
        meas_bg = qt.QButtonGroup(self.parent)
        op = ["fluorescence", "absorption"]
        for j, i in enumerate(op):
            meas_rb = qt.QRadioButton(i)
            meas_rb.setFixedHeight(30)
            meas_rb.setChecked(True if i == self.parent.config["record_control"]["meas_mode"] else False)
            meas_rb.toggled[bool].connect(lambda val, rb=meas_rb: self.update_config("record_control", "meas_mode", (rb.text(), val)))
            self.meas_rblist.append(meas_rb)
            meas_bg.addButton(meas_rb)
            record_frame.addWidget(meas_rb, 1, 1+j, 1, 1)

        # display signal count in real time
        record_frame.addWidget(qt.QLabel("Signal count:"), 2, 0, 1, 1)
        self.signal_count = qt.QLabel()
        self.signal_count.setText("0")
        self.signal_count.setStyleSheet("QLabel{background-color: gray; font: 20pt}")
        self.signal_count.setToolTip("Singal after bkg subtraction or OD")
        record_frame.addWidget(self.signal_count, 2, 1, 1, 2)

        # display mean of signal count in real time in "record" mode
        record_frame.addWidget(qt.QLabel("Singal mean:"), 3, 0, 1, 1)
        self.signal_count_mean = qt.QLabel()
        self.signal_count_mean.setText("0")
        self.signal_count_mean.setStyleSheet("QLabel{background-color: gray; font: 20pt}")
        self.signal_count_mean.setToolTip("Singal after bkg subtraction or OD")
        record_frame.addWidget(self.signal_count_mean, 3, 1, 1, 2)

        # display standard deviation of signal count in real time in "record" mode
        record_frame.addWidget(qt.QLabel("Signal error:"), 4, 0, 1, 1)
        self.signal_count_std = qt.QLabel()
        self.signal_count_std.setText("0")
        self.signal_count_std.setStyleSheet("QLabel{background-color: gray; font: 20pt}")
        self.signal_count_std.setToolTip("Singal after bkg subtraction or OD")
        record_frame.addWidget(self.signal_count_std, 4, 1, 1, 2)

    # place image control gui elements
    def place_image_control(self):
        img_ctrl_box = qt.QGroupBox("Image Control")
        img_ctrl_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        img_ctrl_frame = qt.QFormLayout()
        img_ctrl_box.setLayout(img_ctrl_frame)
        self.frame.addWidget(img_ctrl_box)

        # a spinbox to set number of images to take in next run
        self.num_image_sb = NewSpinBox(range=(1, self.parent.config.getint("image_control", "num_image_max")), suffix=None)
        self.num_image_sb.setValue(self.parent.config.getint("image_control", "num_image"))
        self.num_image_sb.valueChanged[int].connect(lambda val: self.update_config("image_control", "num_image", val))
        img_ctrl_frame.addRow("Num of image to take:", self.num_image_sb)

        # spinboxes to set image region of interest in x
        self.software_roi_xmin_sb = NewSpinBox(range=(0, 
                                                       self.parent.config.getint("image_control", "software_roi_xmax")), 
                                                suffix=None)
        self.software_roi_xmin_sb.setValue(self.parent.config.getint("image_control", "software_roi_xmin"))
        self.software_roi_xmin_sb.setToolTip("Software ROI xmin")
        self.software_roi_xmax_sb = NewSpinBox(range=(self.parent.config.getint("image_control", "software_roi_xmin"), 
                                                        self.parent.config.getint("camera_control", "hardware_roi_width") - 1), 
                                                suffix=None)
        self.software_roi_xmax_sb.setValue(self.parent.config.getint("image_control", "software_roi_xmax"))
        self.software_roi_xmax_sb.setToolTip("Software ROI xmax")
        self.software_roi_xmin_sb.valueChanged[int].connect(lambda val, roi_type='xmin': self.set_software_roi(roi_type, val))
        self.software_roi_xmax_sb.valueChanged[int].connect(lambda val, roi_type='xmax': self.set_software_roi(roi_type, val))

        x_range_box = qt.QWidget()
        x_range_layout = qt.QHBoxLayout()
        x_range_layout.setContentsMargins(0,0,0,0)
        x_range_box.setLayout(x_range_layout)
        x_range_layout.addWidget(self.software_roi_xmin_sb)
        x_range_layout.addWidget(self.software_roi_xmax_sb)
        img_ctrl_frame.addRow("Software ROI (x):", x_range_box)

        # spinboxes to set image region of interest in y
        self.software_roi_ymin_sb = NewSpinBox(range=(0, 
                                                        self.parent.config.getint("image_control", "software_roi_ymax")), 
                                                suffix=None)
        self.software_roi_ymin_sb.setValue(self.parent.config.getint("image_control", "software_roi_ymin"))
        self.software_roi_ymin_sb.setToolTip("Software ROI ymin")
        self.software_roi_ymax_sb = NewSpinBox(range=(self.parent.config.getint("image_control", "software_roi_ymax"), 
                                                     self.parent.config.getint("camera_control", "hardware_roi_height") - 1), 
                                              suffix=None)
        self.software_roi_ymax_sb.setValue(self.parent.config.getint("image_control", "software_roi_ymax"))
        self.software_roi_ymax_sb.setToolTip("Software ROI ymax")
        self.software_roi_ymin_sb.valueChanged[int].connect(lambda val, roi_type='ymin': self.set_software_roi(roi_type, val))
        self.software_roi_ymax_sb.valueChanged[int].connect(lambda val, roi_type='ymax': self.set_software_roi(roi_type, val))

        y_range_box = qt.QWidget()
        y_range_layout = qt.QHBoxLayout()
        y_range_layout.setContentsMargins(0,0,0,0)
        y_range_box.setLayout(y_range_layout)
        y_range_layout.addWidget(self.software_roi_ymin_sb)
        y_range_layout.addWidget(self.software_roi_ymax_sb)
        img_ctrl_frame.addRow("Software ROI (y):", y_range_box)

        # display number of images that have been taken
        self.num_image_la = qt.QLabel()
        self.num_image_la.setText("0")
        self.num_image_la.setStyleSheet("background-color: gray;")
        img_ctrl_frame.addRow("Num of recorded images:", self.num_image_la)

        # set hdf group name and whether to save image to a hdf file
        self.run_name_le = qt.QLineEdit()
        self.run_name_le.setText(self.parent.config.get("image_control", "run_name"))
        self.run_name_le.editingFinished.connect(lambda le=self.run_name_le: self.update_config("image_control", "run_name", le.text()))
        self.run_name_le.setToolTip("HDF group name/run name")
        self.img_save_chb = qt.QCheckBox()
        self.img_save_chb.setTristate(False)
        self.img_save_chb.setChecked(self.parent.config.getboolean("image_control", "image_auto_save"))
        self.img_save_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.img_save_chb.clicked[bool].connect(lambda val: self.update_config("image_control", "image_auto_save", val))
        img_save_box = qt.QWidget()
        img_save_layout = qt.QHBoxLayout()
        img_save_layout.setContentsMargins(0,0,0,0)
        img_save_box.setLayout(img_save_layout)
        img_save_layout.addWidget(self.run_name_le)
        img_save_layout.addWidget(self.img_save_chb)
        img_ctrl_frame.addRow("Image auto save:", img_save_box)

        img_ctrl_frame.addRow("------------------", qt.QWidget())

        # set whether to apply gaussian filter
        self.gaussian_filter_chb = qt.QCheckBox()
        self.gaussian_filter_chb.setTristate(False)
        self.gaussian_filter_chb.setChecked(self.parent.config.getboolean("image_control", "gaussian_filter"))
        self.gaussian_filter_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.gaussian_filter_chb.clicked[bool].connect(lambda val: self.update_config("image_control", "gaussian_filter", val))
        img_ctrl_frame.addRow("Gaussian filter:", self.gaussian_filter_chb)

        # spinboxes to set gaussian filter sigma
        self.gaussian_filter_sigma_dsb = NewDoubleSpinBox(range=(0.01, 10000), decimals=2, suffix=None)
        self.gaussian_filter_sigma_dsb.setValue(self.parent.config.getfloat("image_control", "gaussian_filter_sigma"))
        self.gaussian_filter_sigma_dsb.valueChanged[float].connect(lambda val: self.update_config("image_control", "gaussian_filter_sigma", val))
        img_ctrl_frame.addRow("Gaussian filter sigma:", self.gaussian_filter_sigma_dsb)

        img_ctrl_frame.addRow("------------------", qt.QWidget())

        # set whether to do gaussian fit in real time
        self.gaussian_fit_chb = qt.QCheckBox()
        self.gaussian_fit_chb.setTristate(False)
        self.gaussian_fit_chb.setChecked(self.parent.config.getboolean("image_control", "gaussian_fit"))
        self.gaussian_fit_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.gaussian_fit_chb.clicked[bool].connect(lambda val: self.update_config("image_control", "gaussian_fit", val))
        l = max(self.parent.config.getint("image_control", "gaussian_fit_cpu_limit"), self.parent.config.getint("image_control", "gaussian_fit_gpu_limit"))
        self.gaussian_fit_chb.setToolTip(f"Can only be enabled when image size less than {l} pixels.")
        img_ctrl_frame.addRow("2D gaussian fit:", self.gaussian_fit_chb)

        # display 2D gaussian fit results
        self.gaussian_fit_x_mean_la = qt.QLabel()
        self.gaussian_fit_x_mean_la.setMaximumWidth(90)
        self.gaussian_fit_x_mean_la.setText("0")
        self.gaussian_fit_x_mean_la.setStyleSheet("QWidget{background-color: gray;}")
        self.gaussian_fit_x_mean_la.setToolTip("x mean")
        self.gaussian_fit_x_std_la = qt.QLabel()
        self.gaussian_fit_x_std_la.setMaximumWidth(90)
        self.gaussian_fit_x_std_la.setText("0")
        self.gaussian_fit_x_std_la.setStyleSheet("QWidget{background-color: gray;}")
        self.gaussian_fit_x_std_la.setToolTip("x standard deviation")
        gauss_x_box = qt.QWidget()
        gauss_x_layout = qt.QHBoxLayout()
        gauss_x_layout.setContentsMargins(0,0,0,0)
        gauss_x_box.setLayout(gauss_x_layout)
        gauss_x_layout.addWidget(self.gaussian_fit_x_mean_la)
        gauss_x_layout.addWidget(self.gaussian_fit_x_std_la)
        img_ctrl_frame.addRow("2D gaussian fit (x):", gauss_x_box)

        self.gaussian_fit_y_mean_la = qt.QLabel()
        self.gaussian_fit_y_mean_la.setMaximumWidth(90)
        self.gaussian_fit_y_mean_la.setText("0")
        self.gaussian_fit_y_mean_la.setStyleSheet("QWidget{background-color: gray;}")
        self.gaussian_fit_y_mean_la.setToolTip("y mean")
        self.gaussian_fit_y_std_la = qt.QLabel()
        self.gaussian_fit_y_std_la.setMaximumWidth(90)
        self.gaussian_fit_y_std_la.setText("0")
        self.gaussian_fit_y_std_la.setStyleSheet("QWidget{background-color: gray;}")
        self.gaussian_fit_y_std_la.setToolTip("y standard deviation")
        gauss_y_box = qt.QWidget()
        gauss_y_layout = qt.QHBoxLayout()
        gauss_y_layout.setContentsMargins(0,0,0,0)
        gauss_y_box.setLayout(gauss_y_layout)
        gauss_y_layout.addWidget(self.gaussian_fit_y_mean_la)
        gauss_y_layout.addWidget(self.gaussian_fit_y_std_la)
        img_ctrl_frame.addRow("2D gaussian fit (y):", gauss_y_box)

        self.gaussian_fit_amp = qt.QLabel()
        self.gaussian_fit_amp.setText("0")
        self.gaussian_fit_amp.setStyleSheet("QWidget{background-color: gray;}")
        img_ctrl_frame.addRow("2D gaussian fit (amp.):", self.gaussian_fit_amp)

        self.gaussian_fit_offset = qt.QLabel()
        self.gaussian_fit_offset.setText("0")
        self.gaussian_fit_offset.setStyleSheet("QWidget{background-color: gray;}")
        img_ctrl_frame.addRow("2D gaussian fit (offset):", self.gaussian_fit_offset)

    # place camera control gui elements
    def place_cam_control(self):
        self.cam_ctrl_box = qt.QGroupBox("Camera Control")
        self.cam_ctrl_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        cam_ctrl_frame = qt.QFormLayout()
        self.cam_ctrl_box.setLayout(cam_ctrl_frame)
        self.frame.addWidget(self.cam_ctrl_box)

        # set ROI
        self.hardware_bin_h_sb = NewSpinBox(range=(1, 640), suffix=None)
        self.hardware_bin_h_sb.setValue(self.parent.config.getint("camera_control", "hardware_bin_h"))
        self.hardware_bin_h_sb.setToolTip("Hardware binning horizontal")
        self.hardware_bin_v_sb = NewSpinBox(range=(1, 2160), suffix=None)
        self.hardware_bin_v_sb.setValue(self.parent.config.getint("camera_control", "hardware_bin_v"))
        self.hardware_bin_v_sb.setToolTip("Hardware binning vertical")
        self.hardware_bin_h_sb.valueChanged[int].connect(lambda val: self.set_hardware_roi_size(roi_type="binning", direction="horizontal", val=val))
        self.hardware_bin_v_sb.valueChanged[int].connect(lambda val: self.set_hardware_roi_size(roi_type="binning", direction="vertical", val=val))
        bin_box = qt.QWidget()
        bin_box.setMaximumWidth(200)
        bin_layout = qt.QHBoxLayout()
        bin_layout.setContentsMargins(0,0,0,0)
        bin_box.setLayout(bin_layout)
        bin_layout.addWidget(self.hardware_bin_h_sb)
        bin_layout.addWidget(self.hardware_bin_v_sb)
        cam_ctrl_frame.addRow("HW bin H & V:", bin_box)

        self.hardware_roi_h_centered_chb = qt.QCheckBox()
        self.hardware_roi_h_centered_chb.setTristate(False)
        self.hardware_roi_h_centered_chb.setChecked(self.parent.config.getboolean("camera_control", "hardware_roi_h_centered"))
        self.hardware_roi_h_centered_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.hardware_roi_h_centered_chb.clicked[bool].connect(lambda val: self.set_hardware_roi_centered(direction="horizontal", val=val))
        self.hardware_roi_h_centered_chb.setToolTip("Hardware ROI centered horizontally")
        self.hardware_roi_v_centered_chb = qt.QCheckBox()
        self.hardware_roi_v_centered_chb.setTristate(False)
        self.hardware_roi_v_centered_chb.setChecked(self.parent.config.getboolean("camera_control", "hardware_roi_v_centered"))
        self.hardware_roi_v_centered_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.hardware_roi_v_centered_chb.clicked[bool].connect(lambda val: self.set_hardware_roi_centered(direction="vertical", val=val))
        self.hardware_roi_v_centered_chb.setToolTip("Hardware ROI centered vertically")
        roi_center_box = qt.QWidget()
        roi_center_box.setMaximumWidth(200)
        roi_center_layout = qt.QHBoxLayout()
        roi_center_layout.setContentsMargins(0,0,0,0)
        roi_center_box.setLayout(roi_center_layout)
        roi_center_layout.addWidget(self.hardware_roi_h_centered_chb)
        roi_center_layout.addWidget(self.hardware_roi_v_centered_chb)
        cam_ctrl_frame.addRow("HW ROI centered H & V:", roi_center_box)

        self.hardware_roi_left_sb = NewSpinBox(range=(0, 2559), suffix=None)
        self.hardware_roi_left_sb.setValue(self.parent.config.getint("camera_control", "hardware_roi_left"))
        self.hardware_roi_left_sb.valueChanged[int].connect(lambda val: self.set_hardware_roi_start(direction="horizontal", val=val))
        self.hardware_roi_left_sb.setToolTip("Hardware ROI left")
        self.hardware_roi_top_sb = NewSpinBox(range=(0, 2159), suffix=None)
        self.hardware_roi_top_sb.setValue(self.parent.config.getint("camera_control", "hardware_roi_top"))
        self.hardware_roi_top_sb.valueChanged[int].connect(lambda val: self.set_hardware_roi_start(direction="vertical", val=val))
        self.hardware_roi_top_sb.setToolTip("Hardware ROI top")
        roi_start_box = qt.QWidget()
        roi_start_box.setMaximumWidth(200)
        roi_start_layout = qt.QHBoxLayout()
        roi_start_layout.setContentsMargins(0,0,0,0)
        roi_start_box.setLayout(roi_start_layout)
        roi_start_layout.addWidget(self.hardware_roi_left_sb)
        roi_start_layout.addWidget(self.hardware_roi_top_sb)
        cam_ctrl_frame.addRow("HW ROI start H & V:", roi_start_box)

        self.hardware_roi_width_sb = NewSpinBox(range=(1, 2560), suffix=None)
        self.hardware_roi_width_sb.setValue(self.parent.config.getint("camera_control", "hardware_roi_width"))
        self.hardware_roi_width_sb.valueChanged[int].connect(lambda val: self.set_hardware_roi_size(roi_type="size", direction="horizontal", val=val))
        self.hardware_roi_width_sb.setToolTip("Width in binned pixels")
        self.hardware_roi_width_unbinned_la = qt.QLabel("2560")
        self.hardware_roi_width_unbinned_la.setStyleSheet("QLabel{background-color: gray;}")
        self.hardware_roi_width_unbinned_la.setToolTip("Width in unbinned pixels")
        roi_width_box = qt.QWidget()
        roi_width_box.setMaximumWidth(200)
        roi_width_layout = qt.QHBoxLayout()
        roi_width_layout.setContentsMargins(0,0,0,0)
        roi_width_box.setLayout(roi_width_layout)
        roi_width_layout.addWidget(self.hardware_roi_width_sb)
        roi_width_layout.addWidget(self.hardware_roi_width_unbinned_la)
        cam_ctrl_frame.addRow("HW ROI width:", roi_width_box)

        self.hardware_roi_height_sb = NewSpinBox(range=(1, 2160), suffix=None)
        self.hardware_roi_height_sb.setValue(self.parent.config.getint("camera_control", "hardware_roi_height"))
        self.hardware_roi_height_sb.valueChanged[int].connect(lambda val: self.set_hardware_roi_size(roi_type="size",direction="vertical" , val=val))
        self.hardware_roi_height_sb.setToolTip("Height in binned pixels")
        self.hardware_roi_height_unbinned_la = qt.QLabel("2160")
        self.hardware_roi_height_unbinned_la.setStyleSheet("QLabel{background-color: gray;}")
        self.hardware_roi_height_unbinned_la.setToolTip("Height in unbinned pixels")
        roi_height_box = qt.QWidget()
        roi_height_box.setMaximumWidth(200)
        roi_height_layout = qt.QHBoxLayout()
        roi_height_layout.setContentsMargins(0,0,0,0)
        roi_height_box.setLayout(roi_height_layout)
        roi_height_layout.addWidget(self.hardware_roi_height_sb)
        roi_height_layout.addWidget(self.hardware_roi_height_unbinned_la)
        cam_ctrl_frame.addRow("HW ROI height:", roi_height_box)

        cam_ctrl_frame.addRow("------------------", qt.QWidget())

        # set shutter, trigger, and exposure
        self.shutter_mode_cb = NewComboBox(item_list=["Rolling", "Global"], current_item=self.parent.config.get("camera_control", "shutter_mode"))
        self.shutter_mode_cb.setMaximumWidth(200)
        self.shutter_mode_cb.setMaximumHeight(20)
        self.shutter_mode_cb.currentTextChanged[str].connect(lambda val: self.set_shutter_mode(val))
        cam_ctrl_frame.addRow("Shutter mode:", self.shutter_mode_cb)

        self.trigger_mode_cb = NewComboBox(item_list=["Internal", "Software", "External", "External Exposure", "External Start"], 
                                           current_item=self.parent.config.get("camera_control", "trigger_mode"))
        self.trigger_mode_cb.setMaximumWidth(200)
        self.trigger_mode_cb.setMaximumHeight(20)
        self.trigger_mode_cb.currentTextChanged[str].connect(lambda val: self.set_trigger_mode(val))
        cam_ctrl_frame.addRow("Trigger mode:", self.trigger_mode_cb)

        self.expo_overlap_chb = qt.QCheckBox()
        self.expo_overlap_chb.setTristate(False)
        self.expo_overlap_chb.setChecked(self.parent.config.getboolean("camera_control", "exposure_overlap"))
        self.expo_overlap_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.expo_overlap_chb.clicked[bool].connect(lambda val: self.set_expo_overlap(val))
        cam_ctrl_frame.addRow("Exposure overlap:", self.expo_overlap_chb)

        self.short_exposure_chb = qt.QCheckBox()
        self.short_exposure_chb.setTristate(False)  
        self.short_exposure_chb.setChecked(self.parent.config.getboolean("camera_control", "short_exposure"))
        self.short_exposure_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.short_exposure_chb.clicked[bool].connect(lambda val: self.set_short_exposure(val))
        self.short_exposure_chb.setToolTip("If checked, in global shutter and external trigger mode, " + \
                                           "exposure time can be set to less than frame raedout time. " + \
                                            "The camera will auto delay the exposure by frame readout time.")
        cam_ctrl_frame.addRow("Short exposure:", self.short_exposure_chb)

        self.expo_time_dsb = NewDoubleSpinBox(range=(0.005, 10000), decimals=3, suffix=None)
        self.expo_time_dsb.setValue(self.parent.config.getfloat("camera_control", "exposure_time"))
        self.expo_time_dsb.valueChanged[float].connect(lambda val: self.set_expo_time(val))
        cam_ctrl_frame.addRow("Exposure time (ms):", self.expo_time_dsb)

        self.pixel_readout_rate_cb = NewComboBox(item_list=["100 MHz", "280 MHz"], current_item=self.parent.config.get("camera_control", "pixel_readout_rate"))
        self.pixel_readout_rate_cb.setMaximumWidth(200)
        self.pixel_readout_rate_cb.setMaximumHeight(20)
        self.pixel_readout_rate_cb.currentTextChanged[str].connect(lambda val: self.set_pixel_readout_rate(val))
        cam_ctrl_frame.addRow("Pixel readout rate:", self.pixel_readout_rate_cb)

        self.row_readout_time_la = qt.QLabel("0")
        self.row_readout_time_la.setStyleSheet("QLabel{background-color: gray;}")
        cam_ctrl_frame.addRow("Row readout time (us):", self.row_readout_time_la)

        self.frame_readout_time_la = qt.QLabel("0")
        self.frame_readout_time_la.setStyleSheet("QLabel{background-color: gray;}")
        cam_ctrl_frame.addRow("Frame readout time (ms):", self.frame_readout_time_la)

        cam_ctrl_frame.addRow("------------------", qt.QWidget())

        # set pixel encoding
        self.preamp_gain_cb = NewComboBox(item_list=["12-bit (low noise)", "12-bit (high well capacity)", "16-bit (low noise & high well capacity)"], 
                                          current_item=self.parent.config.get("camera_control", "preamp_gain"))
        self.preamp_gain_cb.setMaximumWidth(200)
        self.preamp_gain_cb.setMaximumHeight(20)
        self.preamp_gain_cb.currentTextChanged[str].connect(lambda val: self.set_preamp_gain(val))
        cam_ctrl_frame.addRow("PreAmp gain:", self.preamp_gain_cb)

        self.pixel_encoding_cb = NewComboBox(item_list=["Mono12", "Mono12Packed","Mono16", "Mono32"], 
                                             current_item=self.parent.config.get("camera_control", "pixel_encoding"))
        self.pixel_encoding_cb.setMaximumWidth(200)
        self.pixel_encoding_cb.setMaximumHeight(20)
        self.pixel_encoding_cb.currentTextChanged[str].connect(lambda val: self.set_pixel_encoding(val))
        cam_ctrl_frame.addRow("Pixel encoding:", self.pixel_encoding_cb)

        self.image_size_bytes_la = qt.QLabel("0")
        self.image_size_bytes_la.setStyleSheet("QLabel{background-color: gray;}")
        cam_ctrl_frame.addRow("Image size (kb):", self.image_size_bytes_la)

        self.max_interface_rate_la = qt.QLabel("0")
        self.max_interface_rate_la.setStyleSheet("QLabel{background-color: gray;}")
        cam_ctrl_frame.addRow("Max interface rate (fps):", self.max_interface_rate_la)

        self.image_baseline_la = qt.QLabel("0")
        self.image_baseline_la.setStyleSheet("QLabel{background-color: gray;}")
        cam_ctrl_frame.addRow("Image baseline:", self.image_baseline_la)

        self.spurious_noise_filter_chb = qt.QCheckBox()
        self.spurious_noise_filter_chb.setTristate(False)
        self.spurious_noise_filter_chb.setChecked(self.parent.config.getboolean("camera_control", "spurious_noise_filter"))
        self.spurious_noise_filter_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.spurious_noise_filter_chb.clicked[bool].connect(lambda val: self.set_noise_filter("spurious", val))
        cam_ctrl_frame.addRow("Spurious noise filter:", self.spurious_noise_filter_chb)

        self.blemish_correction_chb = qt.QCheckBox()
        self.blemish_correction_chb.setTristate(False)
        self.blemish_correction_chb.setChecked(self.parent.config.getboolean("camera_control", "blemish_correction"))
        self.blemish_correction_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.blemish_correction_chb.clicked[bool].connect(lambda val: self.set_noise_filter("blemish", val))
        cam_ctrl_frame.addRow("Blemish correction:", self.blemish_correction_chb)

        cam_ctrl_frame.addRow("------------------", qt.QWidget())

        # set auxiliary TTL output
        self.aux_out_1_cb = NewComboBox(item_list=["FireRow1", "FireRowN", "FireAll", "FireAny"], 
                                        current_item=self.parent.config.get("camera_control", "auxiliary_output_1"))
        self.aux_out_1_cb.setMaximumWidth(200)
        self.aux_out_1_cb.setMaximumHeight(20)
        self.aux_out_1_cb.currentTextChanged[str].connect(lambda val: self.set_aux_out(1, val))
        cam_ctrl_frame.addRow("Auxiliary output 1:", self.aux_out_1_cb)

        self.aux_out_2_cb = NewComboBox(item_list=["ExternalShutterControl", "FrameClock", "RowClock", "ExposedRowClock"], 
                                        current_item=self.parent.config.get("camera_control", "auxiliary_output_2"))
        self.aux_out_2_cb.setMaximumWidth(200)
        self.aux_out_2_cb.setMaximumHeight(20)
        self.aux_out_2_cb.currentTextChanged[str].connect(lambda val: self.set_aux_out(2, val))
        cam_ctrl_frame.addRow("Auxiliary output 2:", self.aux_out_2_cb)

        cam_ctrl_frame.addRow("------------------", qt.QWidget())

        # set cooling
        self.fan_cooling_chb = qt.QCheckBox()
        self.fan_cooling_chb.setTristate(False)
        self.fan_cooling_chb.setChecked(self.parent.config.getboolean("camera_control", "fan_cooling"))
        self.fan_cooling_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.fan_cooling_chb.clicked[bool].connect(lambda val: self.set_cam_cooling("fan", val))
        cam_ctrl_frame.addRow("Fan cooling:", self.fan_cooling_chb)

        self.sensor_cooling_chb = qt.QCheckBox()
        self.sensor_cooling_chb.setTristate(False)
        self.sensor_cooling_chb.setChecked(self.parent.config.getboolean("camera_control", "sensor_cooling"))
        self.sensor_cooling_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.sensor_cooling_chb.clicked[bool].connect(lambda val: self.set_cam_cooling("sensor", val))
        cam_ctrl_frame.addRow("Sensor cooling:", self.sensor_cooling_chb)

        self.cooling_status_la = qt.QLabel("0")
        self.cooling_status_la.setStyleSheet("QLabel{background-color: gray;}")
        cam_ctrl_frame.addRow("Cooling status:", self.cooling_status_la)

        self.sensor_temp_la = qt.QLabel("0")
        self.sensor_temp_la.setStyleSheet("QLabel{background-color: gray;}")
        cam_ctrl_frame.addRow("Sensor temp (C):", self.sensor_temp_la)

        self.cam_reconnect_pb = qt.QPushButton("Reconnect Camera")
        self.cam_reconnect_pb.clicked[bool].connect(lambda val: self.cam_reconnect())
        cam_ctrl_frame.addRow("Reconnect camera:", self.cam_reconnect_pb)

    # place gui elements related to TCP connection
    def place_tcp_control(self):
        tcp_ctrl_box = qt.QGroupBox("TCP Control")
        tcp_ctrl_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        tcp_ctrl_frame = qt.QFormLayout()
        tcp_ctrl_box.setLayout(tcp_ctrl_frame)
        self.frame.addWidget(tcp_ctrl_box)

        server_host = self.parent.config["tcp_control"]["host_addr"]
        server_port = self.parent.config["tcp_control"]["port"]
        self.server_addr_la = qt.QLabel(server_host+" ("+server_port+")")
        self.server_addr_la.setStyleSheet("QLabel{background-color: gray;}")
        self.server_addr_la.setToolTip("server = this PC")
        tcp_ctrl_frame.addRow("Server/This PC address:", self.server_addr_la)

        self.client_addr_la = qt.QLabel("No connection")
        self.client_addr_la.setStyleSheet("QLabel{background-color: gray;}")
        tcp_ctrl_frame.addRow("Last client address:", self.client_addr_la)

        self.last_write_la = qt.QLabel("No connection")
        self.last_write_la.setStyleSheet("QLabel{background-color: gray;}")
        tcp_ctrl_frame.addRow("Last connection time:", self.last_write_la)

        self.restart_tcp_pb = qt.QPushButton("Restart Connection")
        self.restart_tcp_pb.clicked[bool].connect(lambda val: self.restart_tcp())
        tcp_ctrl_frame.addRow("Restart:", self.restart_tcp_pb)

    # place save/load program configuration gui elements
    def place_save_load_control(self):
        self.save_load_box = qt.QGroupBox("Save/Load Settings")
        self.save_load_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        save_load_frame = qt.QFormLayout()
        self.save_load_box.setLayout(save_load_frame)
        self.frame.addWidget(self.save_load_box)

        self.filename_le = qt.QLineEdit()
        default_file_name = self.parent.config["save_load_control"]["filename_to_save"]
        self.filename_le.setText(default_file_name)
        self.filename_le.editingFinished.connect(lambda le = self.filename_le: self.update_config("save_load_control", "filename_to_save", le.text()))
        save_load_frame.addRow("File name to save:", self.filename_le)

        self.date_time_chb = qt.QCheckBox()
        self.date_time_chb.setTristate(False)
        self.date_time_chb.setChecked(self.parent.config["save_load_control"].getboolean("append_datetime"))
        self.date_time_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.date_time_chb.clicked[bool].connect(lambda val: self.update_config("save_load_control", "append_datetime", val))
        save_load_frame.addRow("Auto append time:", self.date_time_chb)

        self.save_settings_pb = qt.QPushButton("save settings")
        self.save_settings_pb.setMaximumWidth(200)
        self.save_settings_pb.clicked[bool].connect(lambda val, latest=False: self.parent.save_settings(latest=latest))
        save_load_frame.addRow("Save settings:", self.save_settings_pb)

        self.load_settings_pb = qt.QPushButton("load settings")
        self.load_settings_pb.setMaximumWidth(200)
        self.load_settings_pb.clicked[bool].connect(lambda val, latest=False: self.parent.load_settings(latest=latest))
        save_load_frame.addRow("Load settings:", self.load_settings_pb)

    def enable_widgets(self, arg):
        # enable/disable controls
        # self.stop_pb.setEnabled(not arg)
        # self.record_pb.setEnabled(arg)
        # self.scan_pb.setEnabled(arg)
        for rb in self.meas_rblist:
            rb.setEnabled(arg)

        self.num_image_sb.setEnabled(arg)
        # self.gaussian_fit_chb.setEnabled(arg)
        self.img_save_chb.setEnabled(arg)
        self.run_name_le.setEnabled(arg)
        self.cam_ctrl_box.setEnabled(arg)
        self.save_load_box.setEnabled(arg)

        # enable/disable in image ROI selection
        # self.software_roi_xmin_sb.setEnabled(arg)
        # self.software_roi_xmax_sb.setEnabled(arg)
        # self.software_roi_ymin_sb.setEnabled(arg)
        # self.software_roi_ymax_sb.setEnabled(arg)
        # for key, roi in self.parent.image_win.img_roi_dict.items():
        #     roi.setEnabled(arg)
        # self.parent.image_win.x_plot_lr.setMovable(arg)
        # self.parent.image_win.y_plot_lr.setMovable(arg)

        # force GUI to respond now
        self.parent.app.processEvents()

    def update_config(self, section, config_type, val):
        if config_type == "meas_mode":
            text, checked = val
            if checked:
                self.parent.config[section][config_type] = text
        else:
            self.parent.config[section][config_type] = str(val)

    def set_software_roi(self, roi_type, val):
        if roi_type == "xmin":
            self.software_roi_xmax_sb.setMinimum(val)
        elif roi_type == "xmax":
            self.software_roi_xmin_sb.setMaximum(val)
        elif roi_type == "ymin":
            self.software_roi_ymax_sb.setMinimum(val)
        elif roi_type == "ymax":
            self.software_roi_ymin_sb.setMaximum(val)

        self.parent.config["image_control"][f"software_roi_{roi_type}"] = str(val)

        self.parent.update_software_roi_in_image(roi_type, 
                                                xmin=self.parent.config.getint("image_control", "software_roi_xmin"),
                                                xmax=self.parent.config.getint("image_control", "software_roi_xmax"),
                                                ymin=self.parent.config.getint("image_control", "software_roi_ymin"),
                                                ymax=self.parent.config.getint("image_control", "software_roi_ymax"))

        self.check_gaussian_fit_limit()
        
    def check_gaussian_fit_limit(self):
        # set in image ROI selection boxes position/size
        x_range = self.parent.config.getint("image_control", "software_roi_xmax") - self.parent.config.getint("image_control", "software_roi_xmin")
        y_range = self.parent.config.getint("image_control", "software_roi_ymax") - self.parent.config.getint("image_control", "software_roi_ymin")

        # disable 2D gaussian fit if ROI is too larges
        l = max(self.parent.config.getint("image_control", "gaussian_fit_cpu_limit"), self.parent.config.getint("image_control", "gaussian_fit_gpu_limit"))
        if x_range * y_range > l:
            if self.gaussian_fit_chb.isEnabled():
                self.gaussian_fit_chb.setChecked(False)
                self.gaussian_fit_chb.setEnabled(False)
                self.parent.config["image_control"]["gaussian_fit"] = "False"
        else:
            if not self.gaussian_fit_chb.isEnabled():
                self.gaussian_fit_chb.setEnabled(True)

    def set_hardware_roi_size(self, roi_type, direction, val):

        if roi_type not in ["binning", "size"]:
            logging.error(f"(ControlGUI.set_hardware_roi_size): roi_type {roi_type} not supported.")
            return

        if direction not in ["horizontal", "vertical"]:
            logging.error(f"(ControlGUI.set_hardware_roi_size): ROI direction {direction} not supported.")
            return

        if direction == "horizontal":
            if roi_type == "binning":
                aoi_binning, aoi_width, aoi_width_min, aoi_width_max, aoi_left, aoi_left_min, aoi_left_max, success = self.camera.set_AOI_binning(direction, val, 
                                                                                                                        centered=self.parent.config.getboolean("camera_control", "hardware_roi_h_centered"))
                
                if not success:
                    self.hardware_bin_h_sb.blockSignals(True)
                    self.hardware_bin_h_sb.setValue(aoi_binning)
                    self.hardware_bin_h_sb.blockSignals(False)
                
                self.parent.config["camera_control"][f"hardware_bin_h"] = str(aoi_binning)

                self.hardware_roi_width_sb.blockSignals(True)
                self.hardware_roi_width_sb.setMaximum(aoi_width_max)
                self.hardware_roi_width_sb.setMinimum(aoi_width_min)
                self.hardware_roi_width_sb.setValue(aoi_width)
                self.hardware_roi_width_sb.blockSignals(False)

            else:
                # roi_type == "size"
                aoi_width, aoi_width_min, aoi_width_max, aoi_left, aoi_left_min, aoi_left_max, success = self.camera.set_AOI_size(direction, val, 
                                                                                                        centered=self.parent.config.getboolean("camera_control", "hardware_roi_h_centered"))

                if not success:
                    self.hardware_roi_width_sb.blockSignals(True)
                    self.hardware_roi_width_sb.setValue(aoi_width)
                    self.hardware_roi_width_sb.blockSignals(False)

            self.parent.config["camera_control"]["hardware_roi_width"] = str(aoi_width)
            self.hardware_roi_width_unbinned_la.setText(str(aoi_width * self.parent.config.getint("camera_control", "hardware_bin_h")))

            self.hardware_roi_left_sb.blockSignals(True)
            self.hardware_roi_left_sb.setMaximum(aoi_left_max)
            self.hardware_roi_left_sb.setMinimum(aoi_left_min)
            self.hardware_roi_left_sb.setValue(aoi_left)
            self.hardware_roi_left_sb.blockSignals(False)
            self.parent.config["camera_control"]["hardware_roi_left"] = str(aoi_left)

            self.software_roi_xmax_sb.blockSignals(True)
            self.software_roi_xmin_sb.blockSignals(True)
            self.software_roi_xmax_sb.setMaximum(aoi_width - 1)
            self.software_roi_xmin_sb.setMaximum(self.software_roi_xmax_sb.value())
            self.software_roi_xmax_sb.setMinimum(self.software_roi_xmin_sb.value())
            self.software_roi_xmin_sb.blockSignals(False)
            self.software_roi_xmax_sb.blockSignals(False)
            self.parent.config["image_control"]["software_roi_xmax"] = str(self.software_roi_xmax_sb.value())
            self.parent.config["image_control"]["software_roi_xmin"] = str(self.software_roi_xmin_sb.value())

            self.parent.update_roi_bound_in_image(bound_type="x",
                                                  xbound=aoi_width,
                                                  ybound=self.parent.config.getint("camera_control", "hardware_roi_height"))
            
            self.parent.update_software_roi_in_image("xmin", # cal this function once, it will update both xmin and xmax
                                                      xmin=self.parent.config.getint("image_control", "software_roi_xmin"),
                                                      xmax=self.parent.config.getint("image_control", "software_roi_xmax"),
                                                      ymin=self.parent.config.getint("image_control", "software_roi_ymin"),
                                                      ymax=self.parent.config.getint("image_control", "software_roi_ymax"))

        else:
            # direction == "vertical"
            if roi_type == "binning":
                aoi_binning, aoi_height, aoi_height_min, aoi_height_max, aoi_top, aoi_top_min, aoi_top_max, success = self.camera.set_AOI_binning(direction, val, 
                                                                                                            centered=self.parent.config.getboolean("camera_control", "hardware_roi_v_centered"))
                
                if not success:
                    self.hardware_bin_v_sb.blockSignals(True)
                    self.hardware_bin_v_sb.setValue(aoi_binning)
                    self.hardware_bin_v_sb.blockSignals(False)
                
                self.parent.config["camera_control"][f"hardware_bin_v"] = str(aoi_binning)

                self.hardware_roi_height_sb.blockSignals(True)
                self.hardware_roi_height_sb.setMaximum(aoi_height_max)
                self.hardware_roi_height_sb.setMinimum(aoi_height_min)
                self.hardware_roi_height_sb.setValue(aoi_height)
                self.hardware_roi_height_sb.blockSignals(False)

            else: 
                # roi_type == "size"
                aoi_height, aoi_height_min, aoi_height_max, aoi_top, aoi_top_min, aoi_top_max, success = self.camera.set_AOI_size(direction, val, 
                                                                                                        centered=self.parent.config.getboolean("camera_control", "hardware_roi_v_centered"))
            
                if not success:
                    self.hardware_roi_height_sb.blockSignals(True)
                    self.hardware_roi_height_sb.setValue(aoi_height)
                    self.hardware_roi_height_sb.blockSignals(False)

            self.parent.config["camera_control"][f"hardware_roi_height"] = str(aoi_height)
            self.hardware_roi_height_unbinned_la.setText(str(aoi_height * self.parent.config.getint("camera_control", "hardware_bin_v")))

            self.hardware_roi_top_sb.blockSignals(True)
            self.hardware_roi_top_sb.setMaximum(aoi_top_max)
            self.hardware_roi_top_sb.setMinimum(aoi_top_min)
            self.hardware_roi_top_sb.setValue(aoi_top)
            self.hardware_roi_top_sb.blockSignals(False)
            self.parent.config["camera_control"]["hardware_roi_top"] = str(aoi_top)

            self.software_roi_ymax_sb.blockSignals(True)
            self.software_roi_ymin_sb.blockSignals(True)
            self.software_roi_ymax_sb.setMaximum(aoi_height - 1)
            self.software_roi_ymin_sb.setMaximum(self.software_roi_ymax_sb.value())
            self.software_roi_ymax_sb.setMinimum(self.software_roi_ymin_sb.value())
            self.software_roi_ymin_sb.blockSignals(False)
            self.software_roi_ymax_sb.blockSignals(False)
            self.parent.config["image_control"]["software_roi_ymax"] = str(self.software_roi_ymax_sb.value())
            self.parent.config["image_control"]["software_roi_ymin"] = str(self.software_roi_ymin_sb.value())

            self.parent.update_roi_bound_in_image(bound_type="y",
                                                  xbound=self.parent.config.getint("camera_control", "hardware_roi_width"),
                                                  ybound=aoi_height)
            
            self.parent.update_software_roi_in_image("ymin", # cal this function once, it will update both ymin and ymax
                                                      xmin=self.parent.config.getint("image_control", "software_roi_xmin"),
                                                      xmax=self.parent.config.getint("image_control", "software_roi_xmax"),
                                                      ymin=self.parent.config.getint("image_control", "software_roi_ymin"),
                                                      ymax=self.parent.config.getint("image_control", "software_roi_ymax"))


        self.check_gaussian_fit_limit()
        self.update_expo_limit_and_rates(update_expo_time=True, update_frame_readout_time=True, update_row_readout_time=False, 
                                       update_image_size=True, update_max_interface_rate=True)

    def set_hardware_roi_centered(self, direction, val):

        if direction == "horizontal":
            aoi_left, actual_centered, success = self.camera.set_AOI_centered(direction, val)
            if not success:
                self.hardware_roi_h_centered_chb.blockSignals(True)
                self.hardware_roi_h_centered_chb.setChecked(actual_centered)
                self.hardware_roi_h_centered_chb.blockSignals(False)
            self.parent.config["camera_control"]["hardware_roi_h_centered"] = str(actual_centered)
            self.hardware_roi_left_sb.setEnabled(not actual_centered)
            self.hardware_roi_left_sb.blockSignals(True)
            self.hardware_roi_left_sb.setValue(aoi_left)
            self.hardware_roi_left_sb.blockSignals(False)
            self.parent.config["camera_control"]["hardware_roi_left"] = str(aoi_left)
        else:
            # direction == "vertical"
            aoi_top, actual_centered, success = self.camera.set_AOI_centered(direction, val)
            if not success:
                self.hardware_roi_v_centered_chb.blockSignals(True)
                self.hardware_roi_v_centered_chb.setChecked(actual_centered)
                self.hardware_roi_v_centered_chb.blockSignals(False)
            self.parent.config["camera_control"]["hardware_roi_v_centered"] = str(actual_centered)
            self.hardware_roi_top_sb.setEnabled(not actual_centered)
            self.hardware_roi_top_sb.blockSignals(True)
            self.hardware_roi_top_sb.setValue(aoi_top)
            self.hardware_roi_top_sb.blockSignals(False)
            self.parent.config["camera_control"]["hardware_roi_top"] = str(aoi_top)
        
        self.update_expo_limit_and_rates(update_expo_time=True, update_frame_readout_time=True, update_row_readout_time=False,
                                       update_image_size=False, update_max_interface_rate=False)

    def set_hardware_roi_start(self, direction, val):
        if direction == "horizontal":
            aoi_left, success = self.camera.set_AOI_start(direction, val, centered=self.parent.config.getboolean("camera_control", "hardware_roi_h_centered"))
            self.parent.config["camera_control"]["hardware_roi_left"] = str(aoi_left)
            if not success:
                self.hardware_roi_left_sb.blockSignals(True)
                self.hardware_roi_left_sb.setValue(aoi_left)
                self.hardware_roi_left_sb.blockSignals(False)
        else:
            # direction == "vertical"
            aoi_top, success = self.camera.set_AOI_start(direction, val, centered=self.parent.config.getboolean("camera_control", "hardware_roi_v_centered"))
            self.parent.config["camera_control"]["hardware_roi_top"] = str(aoi_top)
            if not success:
                self.hardware_roi_top_sb.blockSignals(True)
                self.hardware_roi_top_sb.setValue(aoi_top)
                self.hardware_roi_top_sb.blockSignals(False)
        
        self.update_expo_limit_and_rates(update_expo_time=True, update_frame_readout_time=True, update_row_readout_time=False,
                                       update_image_size=False, update_max_interface_rate=False)
        
    def set_shutter_mode(self, val):
        actual_mode, success = self.cam.set_shutter_mode(val)
        self.parent.config["camera_control"]["shutter_mode"] = actual_mode
        if not success:
            self.shutter_mode_cb.blockSignals(True)
            self.shutter_mode_cb.setCurrentText(actual_mode)
            self.shutter_mode_cb.blockSignals(False)

        self.update_expo_limit_and_rates(update_expo_time=True, update_frame_readout_time=True, update_row_readout_time=False,
                                       update_image_size=False, update_max_interface_rate=False)

    def set_trigger_mode(self, val):
        actual_mode, success = self.cam.set_trigger_mode(val)
        self.parent.config["camera_control"]["trigger_mode"] = actual_mode
        if not success:
            self.trigger_mode_cb.blockSignals(True)
            self.trigger_mode_cb.setCurrentText(actual_mode)
            self.trigger_mode_cb.blockSignals(False)

        self.update_expo_limit_and_rates(update_expo_time=True, update_frame_readout_time=True, update_row_readout_time=False,
                                        update_image_size=False, update_max_interface_rate=False)

    def set_expo_overlap(self, val):
        actual_val, success = self.cam.set_exposure_overlap(val)
        self.parent.config["camera_control"]["exposure_overlap"] = str(actual_val)
        if not success:
            self.expo_overlap_chb.blockSignals(True)
            self.expo_overlap_chb.setChecked(actual_val)
            self.expo_overlap_chb.blockSignals(False)

        self.update_expo_limit_and_rates(update_expo_time=True, update_frame_readout_time=True, update_row_readout_time=False,
                                        update_image_size=False, update_max_interface_rate=False)

    def set_short_exposure(self, val):
        self.parent.config["camera_control"]["short_exposure"] = str(val)

        self.update_expo_limit_and_rates(update_expo_time=True, update_frame_readout_time=True, update_row_readout_time=False,
                                        update_image_size=False, update_max_interface_rate=False)

    def set_expo_time(self, time):
        t, success = self.cam.set_exposure_time(time)
        self.parent.config["camera_control"]["exposure_time"] = str(t)

        if not success:
            self.expo_time_dsb.blockSignals(True)
            self.expo_time_dsb.setValue(t)
            self.expo_time_dsb.blockSignals(False)

    def update_expo_limit_and_rates(self, update_expo_time, 
                                        update_frame_readout_time, 
                                        update_row_readout_time, 
                                        update_image_size, 
                                        update_interface_rate):
        
        if update_expo_time or update_frame_readout_time:
            frame_readout_time = self.camera.read_readout_time("frame")

        if update_expo_time or update_row_readout_time:
            row_readout_time = self.camera.read_readout_time("row")

        if update_expo_time:
            if self.parent.config.get("camera_control", "shutter_mode") == "Global" and \
            self.parent.config.get("camera_control", "trigger_mode") == "External" and \
            self.parent.config.getboolean("camera_control", "exposure_overlap") == False and \
            self.parent.config.getboolean("camera_control", "short_exposure") == False:
                t = frame_readout_time + row_readout_time / 1e3 * 5 # convert to ms
                self.expo_time_dsb.setMinimum(t)
            else:
                min_expo_time, max_expo_time = self.camera.read_exposre_time_range()
                self.expo_time_dsb.setMinimum(min_expo_time)

        if update_frame_readout_time:
            self.frame_readout_time_la.setText(":.3f".format(frame_readout_time))

        if update_row_readout_time:
            self.row_readout_time_la.setText(":.2f".format(row_readout_time))

        if update_image_size:
            self.image_size_bytes_la.setText(":.2f".format(self.camera.read_image_size()))

        if update_interface_rate:
            self.max_interface_rate_la.setText(":.2f".format(self.camera.read_interface_transfer_rate()))

    def set_pixel_readout_rate(self, val):
        actual_rate, success = self.cam.set_pixel_readout_rate(val)
        self.parent.config["camera_control"]["pixel_readout_rate"] = actual_rate

        if not success:
            self.pixel_readout_rate_cb.blockSignals(True)
            self.pixel_readout_rate_cb.setCurrentText(actual_rate)
            self.pixel_readout_rate_cb.blockSignals(False)

        self.update_expo_limit_and_rates(update_expo_time=True, update_frame_readout_time=True, update_row_readout_time=True,
                                        update_image_size=False, update_max_interface_rate=False)

    def set_preamp_gain(self, val):
        actual_gain, pixel_encoding, success = self.cam.set_preamp_gain(val)
        self.parent.config["camera_control"]["preamp_gain"] = actual_gain
        if not success:
            self.preamp_gain_cb.blockSignals(True)
            self.preamp_gain_cb.setCurrentText(actual_gain)
            self.preamp_gain_cb.blockSignals(False)

        if pixel_encoding != self.parent.config["camera_control"]["pixel_encoding"]:
            self.pixel_encoding_cb.blockSignals(True)
            self.pixel_encoding_cb.setCurrentText(pixel_encoding) # pre amp gain can change pixel encoding
            self.pixel_encoding_cb.blockSignals(False)

    def set_pixel_encoding(self, val):
        actual_encoding, success = self.cam.set_pixel_encoding(val)
        self.parent.config["camera_control"]["pixel_encoding"] = actual_encoding
        if not success:
            self.pixel_encoding_cb.blockSignals(True)
            self.pixel_encoding_cb.setCurrentText(actual_encoding)
            self.pixel_encoding_cb.blockSignals(False)

        self.update_expo_limit_and_rates(update_expo_time=False, update_frame_readout_time=False, update_row_readout_time=False,
                                        update_image_size=True, update_max_interface_rate=True)

    def noise_filter(self, filter_type, val):
        actual_val, success = self.cam.enable_noise_filter(filter_type, val)

        if filter_type == "spurious":
            self.parent.config["camera_control"]["spurious_noise_filter"] = str(actual_val)
            if not success:
                self.spurious_noise_filter_chb.blockSignals(True)
                self.spurious_noise_filter_chb.setChecked(actual_val)
                self.spurious_noise_filter_chb.blockSignals(False)
        else:
            # filter_type == "blemish"
            self.parent.config["camera_control"]["blemish_correction"] = str(actual_val)
            if not success:
                self.blemish_correction_chb.blockSignals(True)
                self.blemish_correction_chb.setChecked(actual_val)
                self.blemish_correction_chb.blockSignals(False)

    def set_aux_out(self, out_num, val):
        actual_out, success = self.cam.set_auxiliary_output(out_num, val)
        self.parent.config["camera_control"][f"auxiliary_output_{out_num}"] = actual_out

        if not success:
            if out_num == 1:
                self.aux_out_1_cb.blockSignals(True)
                self.aux_out_1_cb.setCurrentText(actual_out)
                self.aux_out_1_cb.blockSignals(False)
            else:
                # out_num == 2
                self.aux_out_2_cb.blockSignals(True)
                self.aux_out_2_cb.setCurrentText(actual_out)
                self.aux_out_2_cb.blockSignals(False)

    def set_cam_cooling(self, cooling_type, val):
        actual_val, success = self.cam.enable_cooler(cooling_type, val)
        self.parent.config["camera_control"][f"{cooling_type}_cooling"] = str(actual_val)
        if not success:
            if cooling_type == "sensor":
                self.sensor_cooling_chb.blockSignals(True)
                self.sensor_cooling_chb.setChecked(actual_val)
                self.sensor_cooling_chb.blockSignals(False)
            else:
                # cooling_type == "fan"
                self.fan_cooling_chb.blockSignals(True)
                self.fan_cooling_chb.setChecked(actual_val)
                self.fan_cooling_chb.blockSignals(False)

    def tcp_start(self):
        self.tcp_active = True
        self.tcp_thread = TCPThread(self, host_addr=self.parent.config["tcp_control"]["host_addr"], port=self.parent.config["tcp_control"]["port"])
        self.tcp_thread.update_signal.connect(self.tcp_widgets_update)
        self.tcp_thread.start_signal.connect(self.start)
        self.tcp_thread.stop_signal.connect(self.stop)
        self.tcp_thread.start()

    def tcp_stop(self):
        self.tcp_active = False
        try:
            self.tcp_thread.wait() # wait until closed
        except AttributeError as err:
            pass

    def restart_tcp(self):
        self.tcp_stop()
        self.tcp_start()

    @PyQt5.QtCore.pyqtSlot(dict)
    def tcp_widgets_update(self, dict):
        t = dict.get("last write")
        if t:
            self.last_write_la.setText(t)

        addr = dict.get("client addr")
        if addr:
            self.client_addr_la.setText(dict["client addr"][0]+" ("+str(dict["client addr"][1])+")")

    def load_settings(self, latest=False):
        pass
        
        # read camera 
        
        # for i in self.meas_rblist:
        #     if i.text() == self.parent.config["record_control"]["meas_mode"]:
        #         i.setChecked(True)
        #         break

        # self.num_image_sb.setValue(self.parent.onfig["image_control"].getint("num_image"))
        # # the spinbox emits 'valueChanged' signal, and its connected function will be called
        # self.software_roi_xmin_sb.setValue(config["image_control"].getint("xmin"))
        # self.software_roi_xmax_sb.setValue(config["image_control"].getint("xmax"))
        # self.software_roi_ymin_sb.setValue(config["image_control"].getint("ymin"))
        # self.software_roi_ymax_sb.setValue(config["image_control"].getint("ymax"))
        # # make sure image range is updated BEFORE gaussian_fit_chb
        # self.gaussian_fit_chb.setChecked(config["image_control"].getboolean("2d_gaussian_fit"))
        # # the combobox emits 'stateChanged' signal, and its connected function will be called
        # self.img_save_chb.setChecked(config["image_control"].getboolean("image_auto_save"))
        # self.run_name_le.setText(config["image_control"].get("run_name"))

        # self.gaussian_filter_chb.setChecked(config["image_control"].getboolean("gaussian_filter"))
        # self.gaussian_filter_sigma_dsb.setValue(config["image_control"].getfloat("gaussian_filter_sigma"))

        # for name in self.parent.image_win.imgs_name:
        #     self.parent.image_win.auto_scale_chb_dict[name].setChecked(config["image_control"].getboolean(f"auto_scale_state_{name}"))
        # self.parent.image_win.ave_img_auto_scale_chb.setChecked(config["image_control"].getboolean("auto_scale_state_Average_image"))

        # self.sensor_format_cb.setCurrentText(config["camera_control"]["sensor_format"])
        # # the combobox emits 'currentTextChanged' signal, and its connected function will be called

        # self.clock_rate_cb.setCurrentText(config["camera_control"]["clock_rate"])
        # self.conv_factor_cb.setCurrentText(config["camera_control"]["conversion_factor"])
        # for i in self.trig_mode_rblist:
        #     if i.text() == config["camera_control"]["trigger_mode"]:
        #         i.setChecked(True)
        #         break

        # # make sure expo_unit_cb changes before time, because it changes expo_dsb range
        # self.expo_unit_cb.setCurrentText(config["camera_control"]["exposure_unit"])
        # self.expo_dsb.setValue(config["camera_control"].getfloat("exposure_time"))

        # self.bin_hori_cb.setCurrentText(config["camera_control"].get("binning_horizontal"))
        # self.bin_vert_cb.setCurrentText(config["camera_control"].get("binning_vertical"))

        # self.tcp_stop()
        # self.parent.defaults["tcp_connection"] = config["tcp_control"]
        # server_host = self.parent.defaults["tcp_connection"]["host_addr"]
        # server_port = self.parent.defaults["tcp_connection"]["port"]
        # self.server_addr_la.setText(server_host+" ("+server_port+")")
        # self.tcp_start()

    def program_close(self):
        self.tcp_stop()
        self.cam.close()