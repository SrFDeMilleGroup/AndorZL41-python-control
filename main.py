import sys, os, time, configparser
import logging, traceback
from PyQt5.QtCore import QByteArray
import numpy as np
import PyQt5
import PyQt5.QtWidgets as qt
import pyqtgraph as pg
import qdarkstyle # see https://github.com/ColinDuquesnoy/QDarkStyleSheet
import h5py

from program_codes.classes import ControlGUI, ImageGUI, AcquisitionThread
from program_codes.gaussian_fit import gaussian_2d_fit
from program_codes.widgets import ImageWidget

# main class, parent of other classes
class AndorGUI(qt.QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.setWindowTitle('Andor ZL41 Wave 5.5')
        self.setStyleSheet("QWidget{font: 10pt;}")
        # self.setStyleSheet("QToolTip{background-color: black; color: white; font: 10pt;}")
        self.app = app
        logging.getLogger().setLevel("INFO")

        # read saved settings from a local .ini file
        self.config = configparser.ConfigParser()
        self.config.read('program_setting_latest.ini')

        # # instantiate other classes
        self.control_gui = ControlGUI(self)
        # self.image_gui = ImageGUI(self)

        # # load latest settings
        # # self.load_settings(latest=True)

        # self.splitter = qt.QSplitter()
        # self.splitter.setOrientation(PyQt5.QtCore.Qt.Horizontal)
        # self.setCentralWidget(self.splitter)
        # self.splitter.addWidget(self.image_gui)
        # self.splitter.addWidget(self.control_gui)

        self.resize(1600, 900)
        self.show()

    def start(self, mode="scan"):
        # self.control_mode = mode
        self.active = True

        # clear signal count QLabels
        self.signal_count.setText("0")
        self.signal_count_mean.setText("0")
        self.signal_count_std.setText("0")
        self.num_image.setText("0")

        # clear images
        img = np.zeros((self.parent.device.image_shape["xmax"], self.parent.device.image_shape["ymax"]))
        for key, image_show in self.parent.image_win.imgs_dict.items():
            image_show.setImage(img, autoLevels=self.parent.image_win.auto_scale_state_dict[key])
        self.parent.image_win.x_plot_curve.setData(np.sum(img, axis=1))
        self.parent.image_win.y_plot_curve.setData(np.sum(img, axis=0))
        self.parent.image_win.ave_img.setImage(img, autoLevels=self.parent.image_win.ave_img_auto_scale_chb.isChecked())

        # clear gaussian fit QLabels
        self.gaussian_fit_amp.setText("0")
        self.gaussian_fit_offset.setText("0")
        self.gaussian_fit_x_mean_la.setText("0")
        self.gaussian_fit_x_std_la.setText("0")
        self.gaussian_fit_y_mean_la.setText("0")
        self.gaussian_fit_y_std_la.setText("0")

        # initialize a hdf group if image saving is required
        if self.img_save:
            # file name of the hdf file we save image to
            self.hdf_filename = self.parent.defaults["image_save"]["file_name"] + "_" + time.strftime("%Y%m%d") + ".hdf"
            with h5py.File(self.hdf_filename, "a") as hdf_file:
                self.hdf_group_name = self.run_name_le.text()+"_"+time.strftime("%Y%m%d_%H%M%S")
                hdf_file.create_group(self.hdf_group_name)

        self.scan_config = configparser.ConfigParser()
        self.scan_config.optionxform = str
        self.scan_config.read(self.parent.defaults["scan_file_name"]["default"])
        num = (self.scan_config["general"].getint("image_number") + self.scan_config["general"].getint("bkg_image_number")) * self.scan_config["general"].getint("sample_number")
        self.num_image_sb.setValue(num)
        # self.num_img_to_take will be changed automatically

        self.scan_elem_name = self.scan_config["general"].get("scanned_devices_parameters")
        self.scan_elem_name = self.scan_elem_name.split(",")
        self.scan_elem_name = self.scan_elem_name[0].strip()
        if self.scan_elem_name:
            self.control_mode = "scan"
        else:
            self.control_mode = "record"

        if self.control_mode == "scan":
            self.signal_count_dict = {}
            self.parent.image_win.scan_plot_widget.setLabel("bottom", self.scan_elem_name)

            self.parent.image_win.ave_scan_tab.setCurrentIndex(1) # switch to scan plot tab

        # disable and gray out image/camera controls, in case of any accidental parameter change
        self.enable_widgets(False)

        # if self.meas_mode == "fluorescence":
        #     self.parent.image_win.img_tab.setCurrentIndex(2) # switch to fluorescence plot tab
        # elif self.meas_mode == "absorption":
        #     self.parent.image_win.img_tab.setCurrentIndex(3) # switch to absorption plot tab
        # else:
        #     logging.warning("Measurement mode not supported.")
        #     return

        # initialize a image taking thread
        self.rec = AcquisitionThread(self.parent)
        self.rec.signal.connect(self.img_ctrl_update)
        self.rec.finished.connect(self.stop)
        self.rec.start() # start this thread

        # Another way to do this is to use QTimer() to trigger image readout (timer interval can be 0),
        # but in that case, the while loop that waits for the image is running in the main thread,
        # and blocks the main thread.

    # force to stop image taking
    def stop(self):
        if self.active:
            self.active = False
            try:
                self.rec.wait() #  wait until thread closed
            except AttributeError:
                pass

            # don't reset control_mode to None, bcause img_ctrl_update function for the last image may be called after this function being called
            # self.control_mode = None

            self.enable_widgets(True)

    # function that will be called in every experimental cycle to update GUI display
    @PyQt5.QtCore.pyqtSlot(dict)
    def img_ctrl_update(self, img_dict):
        img_type = img_dict["image_type"] # "image" or "bkg"
        if img_type == "bkg":
            img = img_dict["image"]
            # update background image
            self.parent.image_win.imgs_dict["Background"].setImage(img, autoLevels=self.parent.image_win.auto_scale_state_dict["Background"])

            self.num_image.setText(str(img_dict["counter"]+1))

        elif img_type == "image":
            # update signal images
            img = img_dict["image"]
            self.parent.image_win.imgs_dict["Raw Signal"].setImage(img, autoLevels=self.parent.image_win.auto_scale_state_dict["Raw Signal"])

            # img = img_dict["image_post"]
            # if self.meas_mode == "fluorescence":
            #     self.parent.image_win.imgs_dict["Signal minus ave bkg"].setImage(img, autoLevels=self.parent.image_win.auto_scale_state_dict["Signal minus ave bkg"])
            # elif self.meas_mode == "absorption":
            #     self.parent.image_win.imgs_dict["Optical density"].setImage(img, autoLevels=self.parent.image_win.auto_scale_state_dict["Optical density"])
            # else:
            #     logging.warning("Measurement type not supported")
            #     return

            self.num_image.setText(str(img_dict["counter"]+1))

            if img_dict["bkg_counter"] > 0:
                img = img_dict["image_post"]
                self.parent.image_win.imgs_dict["Signal minus ave bkg"].setImage(img, autoLevels=self.parent.image_win.auto_scale_state_dict["Signal minus ave bkg"])
                self.parent.image_win.x_plot_curve.setData(np.sum(img, axis=1))
                self.parent.image_win.y_plot_curve.setData(np.sum(img, axis=0))

                img_roi = img_dict["image_post_roi"]
                self.parent.image_win.x_plot_roi_curve.setData(np.sum(img_roi, axis=1))
                self.parent.image_win.y_plot_roi_curve.setData(np.sum(img_roi, axis=0))

                sc = img_dict["signal_count"]
                self.signal_count.setText(np.format_float_scientific(sc, precision=4))
                self.signal_count_deque.append(sc)
                self.parent.image_win.sc_plot_curve.setData(np.array(self.signal_count_deque), symbol='o')

                if self.control_mode == "record":
                    self.parent.image_win.ave_img.setImage(img_dict["image_ave"], autoLevels=self.parent.image_win.ave_img_auto_scale_state)
                    self.signal_count_mean.setText(np.format_float_scientific(img_dict["signal_count_ave"], precision=4))
                    self.signal_count_std.setText(np.format_float_scientific(img_dict["signal_count_err"], precision=4))
                elif self.control_mode == "scan":
                    x = np.array([])
                    y = np.array([])
                    err = np.array([])
                    for i, (param, sc_list) in enumerate(img_dict["signal_count_scan"].items()):
                        x = np.append(x, float(param))
                        y = np.append(y, np.mean(sc_list))
                        err = np.append(err, np.std(sc_list)/np.sqrt(len(sc_list)))
                    # sort data in order of value of the scan parameter
                    order = x.argsort()
                    x = x[order]
                    y = y[order]
                    err = err[order]
                    # update "signal count vs scan parameter" plot
                    self.parent.image_win.scan_plot_curve.setData(x, y, symbol='o')
                    self.parent.image_win.scan_plot_errbar.setData(x=x, y=y, top=err, bottom=err, beam=(x[-1]-x[0])/len(x)*0.2, pen=pg.mkPen('w', width=1.2))


                if self.gaussian_fit:
                    # do 2D gaussian fit and update GUI displays
                    popt = gaussian_2d_fit(img_dict["image_post_roi"])
                    param = {}
                    param["x_mean"] = popt[1]
                    param["y_mean"] = popt[2]
                    param["x_width"] = popt[3]
                    param["y_width"] = popt[4]
                    param["amp"] = popt[0]
                    param["offset"] = popt[5]
                    self.gaussian_fit_amp.setText("{:.2f}".format(param["amp"]))
                    self.gaussian_fit_offset.setText("{:.2f}".format(param["offset"]))
                    self.gaussian_fit_x_mean_la.setText("{:.2f}".format(param["x_mean"]+self.roi["xmin"]))
                    self.gaussian_fit_x_std_la.setText("{:.2f}".format(param["x_width"]))
                    self.gaussian_fit_y_mean_la.setText("{:.2f}".format(param["y_mean"]+self.roi["ymin"]))
                    self.gaussian_fit_y_std_la.setText("{:.2f}".format(param["y_width"]))

                    xy = np.indices((self.roi["xmax"]-self.roi["xmin"], self.roi["ymax"]-self.roi["ymin"]))
                    # fit = gaussian(param["amp"], param["x_mean"], param["y_mean"], param["x_width"], param["y_width"], param["offset"])(*xy)

                    # self.parent.image_win.x_plot_roi_fit_curve.setData(np.sum(fit, axis=1), pen=pg.mkPen('r'))
                    # self.parent.image_win.y_plot_roi_fit_curve.setData(np.sum(fit, axis=0), pen=pg.mkPen('r'))
                else:
                    self.parent.image_win.x_plot_roi_fit_curve.setData(np.array([]))
                    self.parent.image_win.y_plot_roi_fit_curve.setData(np.array([]))

        if self.img_save:
            # save imagees to local hdf file
            # in "record" mode, all images are save in the same group
            # in "scan" mode, images of the same value of scan parameter are saved in the same group
            with h5py.File(self.hdf_filename, "r+") as hdf_file:
                root = hdf_file.require_group(self.hdf_group_name)
                if self.control_mode == "scan":
                    root.attrs["scanned parameter"] = self.scan_elem_name
                    root.attrs["number of images"] = self.num_img_to_take
                    root = root.require_group(self.scan_elem_name+"_"+img_dict["scan_param"])
                    root.attrs["scanned parameter"] = self.scan_elem_name
                    root.attrs["scanned param value"] = img_dict["scan_param"]
                dset = root.create_dataset(
                                name                 = "image" + "_{:06d}".format(img_dict["counter"]),
                                data                 = img_dict["image"],
                                shape                = img_dict["image"].shape,
                                dtype                = "f",
                                compression          = "gzip",
                                compression_opts     = 4
                            )
                # dset.attrs["signal count"] = img_dict["signal_count"]
                dset.attrs["measurement type"] = self.meas_mode
                dset.attrs["region of interest: xmin"] = self.roi["xmin"]
                dset.attrs["region of interest: xmax"] = self.roi["xmax"]
                dset.attrs["region of interest: ymin"] = self.roi["ymin"]
                dset.attrs["region of interest: ymax"] = self.roi["ymax"]

                # display as image in HDFView
                # https://support.hdfgroup.org/HDF5/doc/ADGuide/ImageSpec.html
                dset.attrs["CLASS"] = np.string_("IMAGE")
                dset.attrs["IMAGE_VERSION"] = np.string_("1.2")
                dset.attrs["IMAGE_SUBCLASS"] = np.string_("IMAGE_GRAYSCALE")
                dset.attrs["IMAGE_WHITE_IS_ZERO"] = 0

                if self.gaussian_fit and (img_type == "image"):
                    for key, val in param.items():
                        dset.attrs["2D gaussian fit"+key] = val

    def update_software_roi_in_image(self, roi_type, xmin, xmax, ymin, ymax):
        if roi_type not in ["xmin", "xmax", "ymin", "ymax"]:
            logging.error(f"(AndorGUI.update_software_roi_from_control) Invalid roi_type: {roi_type}")
            return

        for name, image_widget in self.image_gui.image_widgets.items():
            if roi_type in ["xmin", "ymin"]:
                image_widget.image_roi.blockSignals(True)
                image_widget.image_roi.setPos(pos=(xmin, ymin))
                image_widget.image_roi.setSize(size=(xmax-xmin, ymax-ymin))
                image_widget.image_roi.blockSignals(False)
            elif roi_type in ["xmax", "ymax"]:
                image_widget.image_roi.blockSignals(True)
                image_widget.image_roi.setSize(size=(xmax-xmin, ymax-ymin))
                image_widget.image_roi.blockSignals(False)

        if roi_type in ["xmin", "xmax"]:
            self.image_gui.x_plot_lr.setRegion([xmin, xmax])
        elif roi_type in ["ymin", "ymax"]:
            self.image_gui.y_plot_lr.setRegion([ymin, ymax])
            
    def update_software_roi_in_control(self, roi_type, xmin, xmax, ymin, ymax):
        if roi_type not in ["x", "y", "all"]:
            logging.error(f"(AndorGUI.update_software_roi_from_image) Invalid roi_type: {roi_type}")
            return

        if roi_type in ["x", "all"]:
            self.control_gui.software_roi_xmin_sb.blockSignals(True)
            self.control_gui.software_roi_xmax_sb.blockSignals(True)
            self.control_gui.software_roi_xmin_sb.setRange(0, xmax)
            self.control_gui.software_roi_xmax_sb.setRange(xmin, self.config.getint("camera_control", "hardware_aoi_width") - 1)
            self.control_gui.software_roi_xmin_sb.setValue(xmin)
            self.control_gui.software_roi_xmax_sb.setValue(xmax)
            self.control_gui.software_roi_xmin_sb.blockSignals(False)
            self.control_gui.software_roi_xmax_sb.blockSignals(False)

        if roi_type in ["y", "all"]:
            self.control_gui.software_roi_ymin_sb.blockSignals(True)
            self.control_gui.software_roi_ymax_sb.blockSignals(True)
            self.control_gui.software_roi_ymin_sb.setRange(0, ymax)
            self.control_gui.software_roi_ymax_sb.setRange(ymin, self.config.getint("camera_control", "hardware_aoi_height") - 1)
            self.control_gui.software_roi_ymin_sb.setValue(ymin)
            self.control_gui.software_roi_ymax_sb.setValue(ymax)
            self.control_gui.software_roi_ymin_sb.blockSignals(False)
            self.control_gui.software_roi_ymax_sb.blockSignals(False)

        self.control_gui.check_gaussian_fit_limit()

    def update_roi_bound_in_image(self, bound_type, xbound, ybound):
        if bound_type not in ["x", "y"]:
            logging.error(f"(AndorGUI.update_roi_bound_in_image) Invalid bound_type: {bound_type}")
            return
        
        for name, image_widget in self.image_gui.image_widgets.items():
            image_widget.image_roi.blockSignals(True)
            image_widget.image_roi.setBounds(pos=[0, 0], size=[xbound, ybound])
            image_widget.image_roi.blockSignals(False) 

        if bound_type == "x":
            self.image_gui.x_plot_lr.blockLineSignals(True)
            self.image_gui.x_plot_lr.setRegion([0, xbound])
            self.image_gui.x_plot_lr.blockLineSignals(False)
        elif bound_type == "y":
            self.image_gui.y_plot_lr.blockLineSignals(True)
            self.image_gui.y_plot_lr.setRegion([0, ybound])
            self.image_gui.y_plot_lr.blockLineSignals(False)

    def save_settings(self, latest=False):
        if latest:
            filename = "program_setting_latest.ini"
        else:
        # compile file name
            filename = self.config["save_load_control"]["filename_to_save"]
            if self.date_time_chb.isChecked():
                filename += "_"
                filename += time.strftime("%Y%m%d_%H%M%S")
            filename += ".ini"

            # open a file dialog to choose a file name to save
            filename, _ = qt.QFileDialog.getSaveFileName(self, "Save settings", "saved_settingss/"+filename, "INI File (*.ini);;All Files (*)")
            if not filename:
                return

        configfile = open(filename, "w")
        self.config.write(configfile)
        configfile.close()

    def load_settings(self, latest=False):
        if latest:
            self.config = configparser.ConfigParser()
            self.config.read("program_setting_latest.ini")
        else:
            # open a file dialog to choose a configuration file to load
            filename, _ = qt.QFileDialog.getOpenFileName(self, "Load settigns", "saved_settings/", "All Files (*);;INI File (*.ini)")
            if not filename:
                return

            self.config = configparser.ConfigParser()
            self.config.read(filename)

        self.control_gui.load_settings()
        self.image_gui.load_settings()

    def closeEvent(self, event):
        if not self.control_gui.active:
            self.save_settings(latest=True)
            super().closeEvent(event)

        else:
            # ask if continue to close
            ans = qt.QMessageBox.warning(self, 'Program warning',
                                'Warning: the program is running. Conitnue to close the program?',
                                qt.QMessageBox.Yes | qt.QMessageBox.No,
                                qt.QMessageBox.No)
            if ans == qt.QMessageBox.Yes:
                self.save_settings(latest=True)
                super().closeEvent(event)
            else:
                event.ignore()


if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window = AndorGUI(app)

    try:
        app.exec_()
        # make sure the camera is closed after the program exits
        # main_window.device.cam.close()
        sys.exit(0)
    except SystemExit:
        print("\nApp is closing...")
