import sys, time
import logging, traceback
import numpy as np
import PyQt5
import PyQt5.QtWidgets as qt
import pyqtgraph as pg
import qdarkstyle # see https://github.com/ColinDuquesnoy/QDarkStyleSheet

from program_codes.widgets import Scrollarea, NewSpinBox, NewDoubleSpinBox, imageWidget
from program_codes.classes import AndorZL41Wave

class AcquisitionThread(PyQt5.QtCore.QThread):
    update_signal = PyQt5.QtCore.pyqtSignal(dict)
    finished = PyQt5.QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def run(self):
        dev = self.parent.control_gui.camera
        image_size = int(dev.read_image_size() * 1000) # convert to number of bytes
        dev.start_acquisition(cycle_mode="Continuous")
        while self.parent.active:
            try:
                dev.software_trigger()
                image = dev.read_buffer(circular_buffer=True, image_size=image_size, timeout=10000)
                self.update_signal.emit({"image": image})
                time.sleep(0.03)
            except Exception as err:
                logging.error(f"(AcquisitionThread.run): {err}")
                break

        dev.stop_acquisition()
        self.finished.emit()


class ControlGUI(Scrollarea):
    def __init__(self, parent):
        super().__init__(parent, label="", type="vbox")
        self.setMaximumWidth(330)
        self.frame.setContentsMargins(0,0,0,0)

        self.camera = AndorZL41Wave(self)
        self.cam_init()

        # places GUI elements
        self.place_recording_control()
        self.place_cam_control()

    def cam_init(self):
        self.expo_time = 10 # in ms

        self.camera.set_AOI_binning("horizontal", 1, centered=False)
        self.camera.set_AOI_binning("vertical", 1, centered=False)
        self.camera.set_AOI_centered("horizontal", False)
        self.camera.set_AOI_centered("vertical", False)
        self.camera.set_AOI_start_index("horizontal", 0, centered=False)
        self.camera.set_AOI_start_index("vertical", 0, centered=False)
        self.camera.set_AOI_size("horizontal", 2560, centered=False)
        self.camera.set_AOI_size("vertical", 2160, centered=False)

        self.camera.set_shutter_mode("Global")
        self.camera.set_trigger_mode("Software")
        self.camera.set_exposure_time(self.expo_time)
        self.camera.set_exposure_overlap(False)
        self.camera.set_pixel_readout_rate("280 MHz")

        self.camera.set_pre_amp_gain("16-bit (low noise & high well capacity)")
        self.camera.set_pixel_encoding("Mono16")

        self.camera.enable_noise_filter(filter_type='spurious', enable=True)
        self.camera.enable_noise_filter(filter_type='blemish', enable=True)

        self.camera.enable_cooler(cooler_type="fan", enable=True)
        self.camera.enable_cooler(cooler_type="sensor", enable=True)

    def place_recording_control(self):
        record_box = qt.QGroupBox("Recording")
        record_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        record_box.setMaximumHeight(80)
        record_frame = qt.QFormLayout()
        record_box.setLayout(record_frame)
        self.frame.addWidget(record_box)

        self.record_pb = qt.QPushButton("Start")
        self.record_pb.clicked[bool].connect(lambda val, pb=self.record_pb: self.parent.record(action=pb.text()))
        record_frame.addRow("Live view:", self.record_pb)

    def place_cam_control(self):
        self.cam_ctrl_box = qt.QGroupBox("Camera Control")
        self.cam_ctrl_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        cam_ctrl_frame = qt.QFormLayout()
        self.cam_ctrl_box.setLayout(cam_ctrl_frame)
        self.frame.addWidget(self.cam_ctrl_box)

        # set ROI
        self.hardware_bin_h_sb = NewSpinBox(range=(1, 640), suffix=None) # default setting is 1
        self.hardware_bin_h_sb.setToolTip("Hardware binning horizontal")
        self.hardware_bin_v_sb = NewSpinBox(range=(1, 2160), suffix=None) # default setting is 1
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
        cam_ctrl_frame.addRow("HW binning H && V:", bin_box)

        self.hardware_roi_h_centered_chb = qt.QCheckBox() # default setting is False
        self.hardware_roi_h_centered_chb.setTristate(False)
        self.hardware_roi_h_centered_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.hardware_roi_h_centered_chb.clicked[bool].connect(lambda val: self.set_hardware_roi_centered(direction="horizontal", val=val))
        self.hardware_roi_h_centered_chb.setToolTip("Hardware ROI centered horizontally")
        self.hardware_roi_v_centered_chb = qt.QCheckBox() # default setting is False
        self.hardware_roi_v_centered_chb.setTristate(False)
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
        cam_ctrl_frame.addRow("HW ROI centered H && V:", roi_center_box)

        self.hardware_roi_left_sb = NewSpinBox(range=(0, 2559), suffix=None) # default setting is 0
        self.hardware_roi_left_sb.valueChanged[int].connect(lambda val: self.set_hardware_roi_start(direction="horizontal", val=val))
        self.hardware_roi_left_sb.setToolTip("Hardware ROI left")
        self.hardware_roi_top_sb = NewSpinBox(range=(0, 2159), suffix=None) # default setting is 0
        self.hardware_roi_top_sb.valueChanged[int].connect(lambda val: self.set_hardware_roi_start(direction="vertical", val=val))
        self.hardware_roi_top_sb.setToolTip("Hardware ROI top")
        roi_start_box = qt.QWidget()
        roi_start_box.setMaximumWidth(200)
        roi_start_layout = qt.QHBoxLayout()
        roi_start_layout.setContentsMargins(0,0,0,0)
        roi_start_box.setLayout(roi_start_layout)
        roi_start_layout.addWidget(self.hardware_roi_left_sb)
        roi_start_layout.addWidget(self.hardware_roi_top_sb)
        cam_ctrl_frame.addRow("HW ROI start H && V:", roi_start_box)

        self.hardware_roi_width_sb = NewSpinBox(range=(1, 2560), suffix=None)
        self.hardware_roi_width_sb.setValue(2560) # camera default setting
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
        self.hardware_roi_height_sb.setValue(2160) # camera default setting
        self.hardware_roi_height_sb.valueChanged[int].connect(lambda val: self.set_hardware_roi_size(roi_type="size", direction="vertical", val=val))
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

        self.expo_time_dsb = NewDoubleSpinBox(range=(0.005, 30000), decimals=3, suffix=None)
        self.expo_time_dsb.setValue(self.expo_time)
        self.expo_time_dsb.valueChanged[float].connect(lambda val: self.set_expo_time(val))
        cam_ctrl_frame.addRow("Exposure time (ms):", self.expo_time_dsb)

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
                                                                                                                        centered=self.hardware_roi_h_centered_chb.isChecked())
                
                if not success:
                    self.hardware_bin_h_sb.blockSignals(True)
                    self.hardware_bin_h_sb.setValue(aoi_binning)
                    self.hardware_bin_h_sb.blockSignals(False)

                self.hardware_roi_width_sb.blockSignals(True)
                self.hardware_roi_width_sb.setMaximum(aoi_width_max)
                self.hardware_roi_width_sb.setMinimum(aoi_width_min)
                self.hardware_roi_width_sb.setValue(aoi_width)
                self.hardware_roi_width_sb.blockSignals(False)

            else:
                # roi_type == "size"
                aoi_width, aoi_width_min, aoi_width_max, aoi_left, aoi_left_min, aoi_left_max, success = self.camera.set_AOI_size(direction, val, 
                                                                                                        centered=self.hardware_roi_h_centered_chb.isChecked())

                if not success:
                    self.hardware_roi_width_sb.blockSignals(True)
                    self.hardware_roi_width_sb.setValue(aoi_width)
                    self.hardware_roi_width_sb.blockSignals(False)

            self.hardware_roi_width_unbinned_la.setText(str(aoi_width * self.hardware_bin_h_sb.value()))

            self.hardware_roi_left_sb.blockSignals(True)
            self.hardware_roi_left_sb.setMaximum(aoi_left_max)
            self.hardware_roi_left_sb.setMinimum(aoi_left_min)
            self.hardware_roi_left_sb.setValue(aoi_left)
            self.hardware_roi_left_sb.blockSignals(False)

        else:
            # direction == "vertical"
            if roi_type == "binning":
                aoi_binning, aoi_height, aoi_height_min, aoi_height_max, aoi_top, aoi_top_min, aoi_top_max, success = self.camera.set_AOI_binning(direction, val, 
                                                                                                            centered=self.hardware_roi_v_centered_chb.isChecked())
                
                if not success:
                    self.hardware_bin_v_sb.blockSignals(True)
                    self.hardware_bin_v_sb.setValue(aoi_binning)
                    self.hardware_bin_v_sb.blockSignals(False)
                

                self.hardware_roi_height_sb.blockSignals(True)
                self.hardware_roi_height_sb.setMaximum(aoi_height_max)
                self.hardware_roi_height_sb.setMinimum(aoi_height_min)
                self.hardware_roi_height_sb.setValue(aoi_height)
                self.hardware_roi_height_sb.blockSignals(False)

            else: 
                # roi_type == "size"
                aoi_height, aoi_height_min, aoi_height_max, aoi_top, aoi_top_min, aoi_top_max, success = self.camera.set_AOI_size(direction, val, 
                                                                                                        centered=self.hardware_roi_v_centered_chb.isChecked())
            
                if not success:
                    self.hardware_roi_height_sb.blockSignals(True)
                    self.hardware_roi_height_sb.setValue(aoi_height)
                    self.hardware_roi_height_sb.blockSignals(False)

            self.hardware_roi_height_unbinned_la.setText(str(aoi_height * self.hardware_bin_v_sb.value()))

            self.hardware_roi_top_sb.blockSignals(True)
            self.hardware_roi_top_sb.setMaximum(aoi_top_max)
            self.hardware_roi_top_sb.setMinimum(aoi_top_min)
            self.hardware_roi_top_sb.setValue(aoi_top)
            self.hardware_roi_top_sb.blockSignals(False)

        self.update_expo_limits()

    def set_hardware_roi_centered(self, direction, val):

        if direction == "horizontal":
            aoi_left, actual_centered, success = self.camera.set_AOI_centered(direction, val)
            if not success:
                self.hardware_roi_h_centered_chb.blockSignals(True)
                self.hardware_roi_h_centered_chb.setChecked(actual_centered)
                self.hardware_roi_h_centered_chb.blockSignals(False)
            self.hardware_roi_left_sb.setEnabled(not actual_centered)
            self.hardware_roi_left_sb.blockSignals(True)
            self.hardware_roi_left_sb.setValue(aoi_left)
            self.hardware_roi_left_sb.blockSignals(False)
        else:
            # direction == "vertical"
            aoi_top, actual_centered, success = self.camera.set_AOI_centered(direction, val)
            if not success:
                self.hardware_roi_v_centered_chb.blockSignals(True)
                self.hardware_roi_v_centered_chb.setChecked(actual_centered)
                self.hardware_roi_v_centered_chb.blockSignals(False)
            self.hardware_roi_top_sb.setEnabled(not actual_centered)
            self.hardware_roi_top_sb.blockSignals(True)
            self.hardware_roi_top_sb.setValue(aoi_top)
            self.hardware_roi_top_sb.blockSignals(False)
        
        self.update_expo_limits()

    def set_hardware_roi_start(self, direction, val):
        if direction == "horizontal":
            aoi_left, success = self.camera.set_AOI_start_index(direction, val, centered=self.hardware_roi_h_centered_chb.isChecked())
            if not success:
                self.hardware_roi_left_sb.blockSignals(True)
                self.hardware_roi_left_sb.setValue(aoi_left)
                self.hardware_roi_left_sb.blockSignals(False)
        else:
            # direction == "vertical"
            aoi_top, success = self.camera.set_AOI_start_index(direction, val, centered=self.hardware_roi_v_centered_chb.isChecked())
            if not success:
                self.hardware_roi_top_sb.blockSignals(True)
                self.hardware_roi_top_sb.setValue(aoi_top)
                self.hardware_roi_top_sb.blockSignals(False)
        
        self.update_expo_limits()

    def set_expo_time(self, time):
        t, success = self.camera.set_exposure_time(time)

        if not success:
            self.expo_time_dsb.blockSignals(True)
            self.expo_time_dsb.setValue(t)
            self.expo_time_dsb.blockSignals(False)
        
    def update_expo_limits(self):
        min_expo_time, max_expo_time = self.camera.read_exposre_time_range()
        self.expo_time_dsb.setMinimum(min_expo_time)
        self.expo_time_dsb.setMaximum(max_expo_time)

    def enable_widgets(self, arg):
        self.cam_ctrl_box.setEnabled(arg)

    def program_close(self):
        self.camera.close()

# the class that places images and plots
class ImageGUI(Scrollarea):
    def __init__(self, parent):
        super().__init__(parent, label="Images", type="grid")

        self.frame.setContentsMargins(0,0,0,0)

        # place images and plots
        self.place_signal_images()

    def place_signal_images(self):
        self.image_widget = imageWidget(parent=self, name='', include_ROI=False, colorname="jet", dummy_data_xmax=2560, dummy_data_ymax=2160)
        self.frame.addWidget(self.image_widget.graphlayout, 0, 0)
        self.image_widget.auto_scale_chb.setChecked(True)

class AndorGUI(qt.QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.setWindowTitle('Andor ZL41 Wave 5.5 Live View')
        self.setStyleSheet("QWidget{font: 10pt;}")
        # self.setStyleSheet("QToolTip{background-color: black; color: white; font: 10pt;}")
        self.app = app
        logging.getLogger().setLevel("INFO")
        
        self.active = False

        # instantiate other classes
        self.control_gui = ControlGUI(self)
        self.image_gui = ImageGUI(self)

        self.splitter = qt.QSplitter()
        self.splitter.setOrientation(PyQt5.QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        self.splitter.addWidget(self.image_gui)
        self.splitter.addWidget(self.control_gui)

        self.resize(1100, 600)
        self.show()

    def record(self, action):
        assert action in ["Start", "Stop"], f"action {action} not supported."

        if action == "Start":
            if self.active:
                return
            self.active = True
            self.control_gui.record_pb.setText("Stop")
            self.control_gui.enable_widgets(False)
            self.acquisition_thread = AcquisitionThread(self)
            self.acquisition_thread.update_signal[dict].connect(self.update_images)
            self.acquisition_thread.finished.connect(lambda action="Stop": self.record(action=action))
            self.acquisition_thread.start()

        else:
            if not self.active:
                return

            self.active = False
            try:
                self.acquisition_thread.wait() # wait to exit
            except AttributeError:
                pass
            self.control_gui.record_pb.setText("Start")
            self.control_gui.enable_widgets(True)


    @PyQt5.QtCore.pyqtSlot(dict)
    def update_images(self, info_dict):
        self.image_gui.image_widget.image.setImage(info_dict["image"], autoLevels=self.image_gui.image_widget.auto_scale_chb.isChecked())

    def closeEvent(self, event):
        if not self.active:
            self.control_gui.program_close()
            super().closeEvent(event)

        else:
            # ask if continue to close
            ans = qt.QMessageBox.warning(self, 'Program warning',
                                'Warning: the program is running. Conitnue to close the program?',
                                qt.QMessageBox.Yes | qt.QMessageBox.No,
                                qt.QMessageBox.No)
            if ans == qt.QMessageBox.Yes:
                self.record(action="Stop") # stop the camera acquisition
                self.control_gui.program_close() # close the camera
                super().closeEvent(event)
            else:
                event.ignore()


if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window = AndorGUI(app)

    try:
        app.exec_()
        sys.exit(0)
    except SystemExit:
        print("\nApp is closing...")