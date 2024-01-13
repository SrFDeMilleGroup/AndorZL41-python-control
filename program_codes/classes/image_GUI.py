import logging
import PyQt5.QtWidgets as qt
import pyqtgraph as pg
import numpy as np

from ..widgets import Scrollarea, imageWidget


# the class that places images and plots
class ImageGUI(Scrollarea):
    def __init__(self, parent):
        super().__init__(parent, label="Images", type="grid")
        self.frame.setColumnStretch(0,7)
        self.frame.setColumnStretch(1,4)
        self.frame.setRowStretch(0,1)
        self.frame.setRowStretch(1,1)
        self.frame.setRowStretch(2,1)
        self.frame.setContentsMargins(0,0,0,0)
        self.signal_image_name_list = ["Background", "Raw Signal", "Signal minus ave bkg", "Optical density"]
        self.signal_image_widget_list = {}

        # place images and plots
        self.place_signal_images()
        self.place_axis_plots()

        self.ave_scan_tab = qt.QTabWidget()
        self.frame.addWidget(self.ave_scan_tab, 2, 0)
        self.place_ave_image()
        self.place_scan_plot()

        self.place_signal_count_plot()

    # place background and signal images
    def place_signal_images(self):
        self.image_tab = qt.QTabWidget()
        self.frame.addWidget(self.image_tab, 0, 0, 2, 1)
        for i, name in enumerate(self.signal_image_name_list):
            image_widget = imageWidget(parent=self, name=name, include_ROI=True, colorname="viridis", 
                                    dummy_data_xmax=self.parent.config.getint("camera_control", "hardware_roi_width"),
                                    dummy_data_ymax=self.parent.config.getint("camera_control", "hardware_roi_height"),
                                    )

            # add the widget to the front panel
            self.image_tab.addTab(image_widget.graphlayout, " " + name + " ")

            # config ROI
            image_widget.image_roi.setPos(pos=(self.parent.config.getint("image_control" ,"software_roi_xmin"), 
                                          self.parent.config.getint("image_control" ,"software_roi_ymin")))
            image_widget.image_roi.setSize(size=(self.parent.config.getint("image_control", "software_roi_xmax") - \
                                            self.parent.config.getint("image_control", "software_roi_xmin"),
                                            self.parent.config.getint("image_control", "software_roi_ymax") - \
                                            self.parent.config.getint("image_control", "software_roi_ymin")))
            image_widget.image_roi.sigRegionChanged.connect(lambda roi, name=name: self.update_image_roi(source_image_name=name))
            image_widget.image_roi.setBounds(pos=[0,0], size=[self.parent.config.getint("camera_control", "hardware_roi_width"), 
                                                              self.parent.config.getint("camera_control", "hardware_roi_height")])

            image_widget.auto_scale_chb.setChecked(self.parent.config.getboolean("image_auto_scale_control", name))
            image_widget.auto_scale_chb.clicked[bool].connect(lambda val, param=name: self.update_auto_scale(val, param))

            self.signal_image_widget_list[name] = image_widget
        
        self.starting_data = image_widget.dummy_data

        self.image_tab.setCurrentIndex(2) # make tab #2 (count from 0) to show as default

    # place plots of signal_count along one axis
    def place_axis_plots(self):
        tickstyle = {"showValues": False}

        self.curve_tab = qt.QTabWidget()
        self.frame.addWidget(self.curve_tab, 0, 1, 2, 1)

        # place plot of signal_count along x axis
        x_data = np.sum(self.starting_data, axis=1)
        graphlayout = pg.GraphicsLayoutWidget(parent=self, border=True)
        self.curve_tab.addTab(graphlayout, " Full Frame Signal ")
        x_plot = graphlayout.addPlot(title="Integrated signal v.s. X")
        x_plot.showGrid(True, True)
        x_plot.setLabel("top")
        # x_plot.getAxis("top").setTicks([])
        x_plot.getAxis("top").setStyle(**tickstyle)
        x_plot.setLabel("right")
        # x_plot.getAxis("right").setTicks([])
        x_plot.getAxis("right").setStyle(**tickstyle)
        self.x_plot_curve = x_plot.plot(x_data)

        # add ROI selection
        self.x_plot_lr = pg.LinearRegionItem([self.parent.config.getint("image_control", "software_roi_xmin"),
                                              self.parent.config.getint("image_control", "software_roi_xmax")], 
                                             swapMode="block")
        # no "snap" option for LinearRegion item?
        self.x_plot_lr.setBounds([0, self.parent.config.getint("camera_control", "hardware_roi_width")])
        x_plot.addItem(self.x_plot_lr)
        self.x_plot_lr.sigRegionChanged.connect(lambda roi, name="x_plot": self.update_image_roi(source_image_name=name))

        graphlayout.nextRow()

        # place plot of signal_count along y axis
        y_data = np.sum(self.starting_data, axis=0)
        y_plot = graphlayout.addPlot(title="Integrated signal v.s. Y")
        y_plot.showGrid(True, True)
        y_plot.setLabel("top")
        y_plot.getAxis("top").setStyle(**tickstyle)
        y_plot.setLabel("right")
        y_plot.getAxis("right").setStyle(**tickstyle)
        self.y_plot_curve = y_plot.plot(y_data)

        # add ROI selection
        self.y_plot_lr = pg.LinearRegionItem([self.parent.config.getint("image_control", "software_roi_ymin"),
                                              self.parent.config.getint("image_control", "software_roi_ymax")], 
                                             swapMode="block")
        self.y_plot_lr.setBounds([0, self.parent.config.getint("camera_control", "hardware_roi_height")])
        y_plot.addItem(self.y_plot_lr)
        self.y_plot_lr.sigRegionChanged.connect(lambda roi, name="y_plot": self.update_image_roi(source_image_name=name))

        graphlayout = pg.GraphicsLayoutWidget(parent=self, border=True)
        self.curve_tab.addTab(graphlayout, " Signal in ROI ")

        x_plot = graphlayout.addPlot(title="Integrated signal v.s. X")
        x_plot.showGrid(True, True)
        x_plot.setLabel("top")
        # x_plot.getAxis("top").setTicks([])
        x_plot.getAxis("top").setStyle(**tickstyle)
        x_plot.setLabel("right")
        # x_plot.getAxis("right").setTicks([])
        x_plot.getAxis("right").setStyle(**tickstyle)
        data_roi = self.starting_data[self.parent.config.getint("image_control", "software_roi_xmin") : self.parent.config.getint("image_control", "software_roi_xmax"),
                                      self.parent.config.getint("image_control", "software_roi_ymin") : self.parent.config.getint("image_control", "software_roi_ymax")]
        x_data = np.sum(data_roi, axis=1)
        self.x_plot_roi_curve = x_plot.plot(x_data)
        self.x_plot_roi_fit_curve = x_plot.plot(np.array([]))

        graphlayout.nextRow()

        # place plot of signal_count along y axis
        y_plot = graphlayout.addPlot(title="Integrated signal v.s. Y")
        y_plot.showGrid(True, True)
        y_plot.setLabel("top")
        y_plot.getAxis("top").setStyle(**tickstyle)
        y_plot.setLabel("right")
        y_plot.getAxis("right").setStyle(**tickstyle)
        y_data = np.sum(data_roi, axis=0)
        self.y_plot_roi_curve = y_plot.plot(y_data)
        self.y_plot_roi_fit_curve = y_plot.plot(np.array([]))

    # place averaged image
    def place_ave_image(self):
        name = "Average image"
        self.ave_image_widget = imageWidget(parent=self, name=name, include_ROI=False, colorname="viridis", 
                                dummy_data_xmax=self.parent.config.getint("image_control", "software_roi_xmax"),
                                dummy_data_ymax=self.parent.config.getint("image_control", "software_roi_ymax"),
                                )

        self.ave_scan_tab.addTab(self.ave_image_widget.graphlayout, " " + name + " ")
        self.ave_image_widget.auto_scale_chb.setChecked(self.parent.config.getboolean("image_auto_scale_control", "Average image"))
        self.ave_image_widget.auto_scale_chb.clicked[bool].connect(lambda val, param="Average image": self.update_auto_scale(val, param))

    # place scan plots
    def place_scan_plot(self):
        tickstyle = {"showValues": False}

        self.scan_plot_widget = pg.PlotWidget(title="Integrated signal v.s. Scan param.")
        self.scan_plot_widget.showGrid(True, True)
        self.scan_plot_widget.setLabel("top")
        self.scan_plot_widget.getAxis("top").setStyle(**tickstyle)
        self.scan_plot_widget.setLabel("right")
        self.scan_plot_widget.getAxis("right").setStyle(**tickstyle)
        fontstyle = {"color": "#919191", "font-size": "11pt"}
        self.scan_plot_widget.setLabel("bottom", "Scan parameter", **fontstyle)
        self.scan_plot_widget.getAxis("bottom").enableAutoSIPrefix(False)
        self.scan_plot_curve = self.scan_plot_widget.plot()

        # place error bar
        self.scan_plot_errbar = pg.ErrorBarItem()
        self.scan_plot_widget.addItem(self.scan_plot_errbar)

        self.ave_scan_tab.addTab(self.scan_plot_widget, " Scan Plot ")

    # place a plot showing running signal count
    def place_signal_count_plot(self):
        tickstyle = {"showValues": False}

        self.signal_count_plot_widget = pg.PlotWidget(title="Signal count")
        self.signal_count_plot_widget.showGrid(True, True)
        self.signal_count_plot_widget.setLabel("top")
        self.signal_count_plot_widget.getAxis("top").setStyle(**tickstyle)
        self.signal_count_plot_widget.setLabel("right")
        self.signal_count_plot_widget.getAxis("right").setStyle(**tickstyle)
        self.signal_count_plot_curve = self.signal_count_plot_widget.plot()

        self.frame.addWidget(self.signal_count_plot_widget, 2, 1)

    # set ROI in background/signal images
    def update_image_roi(self, source_image_name):

        if source_image_name not in self.signal_image_name_list + ["x_plot", "y_plot"]:
            logging.error(f"(ImageGUI.update_image_roi): Unsupported ROI source image name {source_image_name}.")
            return

        if source_image_name in self.signal_image_name_list:
            xmin, ymin = self.signal_image_widget_list[source_image_name].image_roi.pos()
            xsize, ysize = self.signal_image_widget_list[source_image_name].image_roi.size()
            xmax = xmin + xsize
            ymax = ymin + ysize

            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)

            self.parent.config["image_control"]["software_roi_xmin"] = str(xmin)
            self.parent.config["image_control"]["software_roi_xmax"] = str(xmax)
            self.parent.config["image_control"]["software_roi_ymin"] = str(ymin)
            self.parent.config["image_control"]["software_roi_ymax"] = str(ymax)

            self.x_plot_lr.blockSignals(True)
            self.x_plot_lr.setRegion([xmin, xmax])
            self.x_plot_lr.blockSignals(False)

            self.y_plot_lr.blockSignals(True)
            self.y_plot_lr.setRegion([ymin, ymax])
            self.y_plot_lr.blockSignals(False)

            roi_type = "all"

        elif source_image_name == "x_plot":
            xmin, xmax = self.x_plot_lr.getRegion()
            xmin = int(xmin)
            xmax = int(xmax)
            self.parent.config["image_control"]["software_roi_xmin"] = str(xmin)
            self.parent.config["image_control"]["software_roi_xmax"] = str(xmax)
            ymin = self.parent.config.getint("image_control", "software_roi_ymin")
            ymax = self.parent.config.getint("image_control", "software_roi_ymax")

            roi_type = "x"

        elif source_image_name == "y_plot":
            ymin, ymax = self.y_plot_lr.getRegion()
            ymin = int(ymin)
            ymax = int(ymax)
            self.parent.config["image_control"]["software_roi_ymin"] = str(ymin)
            self.parent.config["image_control"]["software_roi_ymax"] = str(ymax)
            xmin = self.parent.config.getint("image_control", "software_roi_xmin")
            xmax = self.parent.config.getint("image_control", "software_roi_xmax")

            roi_type = "y"

        for name, image_widget in self.signal_image_widget_list.items():
            if name != source_image_name:
                image_widget.image_roi.blockSignals(True)
                image_widget.image_roi.setPos(pos=(xmin, ymin))
                image_widget.image_roi.setSize(size=(xmax-xmin, ymax-ymin))
                image_widget.image_roi.blockSignals(False)

        self.parent.update_software_roi_in_control(roi_type, xmin, xmax, ymin, ymax)

    def update_auto_scale(self, val, param):
        # logging.info(str(val))
        if param == "Average image":
            self.parent.config["image_auto_scale_control"]["Average image"] = str(val)
        elif param in self.signal_image_name_list:
            self.parent.config["image_auto_scale_control"][param] = str(val)
        else:
            logging.error(f"(ImageGUI.update_auto_scale): Unsupported auto scale param {param}.")

    def load_settings(self):
        pass
