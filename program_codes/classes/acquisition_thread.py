import logging, traceback, time
import numpy as np
import PyQt5.QtCore
from scipy.ndimage import gaussian_filter


# the thread called when the program starts to interface with camera and take images
# this thread waits unitl a new image is available and read it out from the camera
class AcquisitionThread(PyQt5.QtCore.QThread):
    signal = PyQt5.QtCore.pyqtSignal(dict)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.counter_limit = self.parent.control.num_img_to_take
        self.counter = 0

        if self.parent.control.control_mode == "record":
            self.signal_count_list = []
            self.img_ave = np.zeros((self.parent.device.image_shape["xmax"], self.parent.device.image_shape["ymax"]))
        elif self.parent.control.control_mode == "scan":
            self.signal_count_dict = {}

        self.parent.device.cam.record(number_of_images=4, mode='ring buffer')
        # number_of_images is buffer size in ring buffer mode, and has to be at least 4

        self.scan_config = self.parent.control.scan_config
        self.ave_bkg = None
        self.bkg_counter = 0
        self.last_time = time.time()

    def run(self):
        while self.counter < self.counter_limit and self.parent.control.active:
            # type = self.image_type[self.counter%2] # odd-numbered image is background, even-numbered image is signal
            

            if self.parent.device.trigger_mode == "software":
                self.parent.device.cam.sdk.force_trigger() # software-ly trigger the camera
                time.sleep(0.5)

            while self.parent.control.active:
                # wait until a new image is available,
                # this step will block the thread, so it can;t be in the main thread
                if self.parent.device.cam.rec.get_status()['dwProcImgCount'] > self.counter:
                    break
                time.sleep(0.001)

            if self.parent.control.active:
                image, meta = self.parent.device.cam.image(image_number=0xFFFFFFFF) # readout the lastest image
                # image is in "unit16" data type, althought it only has 14 non-zero bits at most
                # convert the image data type to float, to avoid overflow
                image = np.flip(image.T, 1).astype("float")
                xstart = int(image.shape[0]/2 - self.parent.device.image_shape['xmax']/2)
                ystart = int(image.shape[1]/2 - self.parent.device.image_shape['ymax']/2)
                image = image[xstart : xstart+self.parent.device.image_shape['xmax'],
                                ystart : ystart+self.parent.device.image_shape['ymax']]

                image_type = self.scan_config[f"scan_value_{self.counter}"]["image_type"] # "image" or "bkg"
                self.img_dict = {"image_type": image_type, "image": image, "counter": self.counter, "bkg_counter": self.bkg_counter}
                if self.parent.control.control_mode == "record":
                    self.img_dict["scan_param"] = ""
                if self.parent.control.control_mode == "scan":
                    scan_param = self.scan_config[f"scan_value_{self.counter}"][self.parent.control.scan_elem_name]
                    self.img_dict["scan_param"] = scan_param

                if image_type == "bkg":
                    if self.bkg_counter > 0:
                        self.ave_bkg = (self.ave_bkg*self.bkg_counter + image)/(self.bkg_counter + 1)
                        self.ave_bkg = np.average(np.array([self.ave_bkg, image]), axis=0, 
                                                    weights=[self.bkg_counter/(self.bkg_counter+1), 1/(self.bkg_counter+1)])
                    else:
                        self.ave_bkg = image
                    self.bkg_counter += 1

                    self.signal.emit(self.img_dict)
                
                elif image_type == "image":
                    if self.bkg_counter > 0:
                        image_post = image - self.ave_bkg
                        if self.parent.control.gaussian_filter:
                            image_post = gaussian_filter(image_post, self.parent.control.gaussian_filter_sigma)

                        image_post_roi = image_post[self.parent.control.roi["xmin"] : self.parent.control.roi["xmax"],
                                                    self.parent.control.roi["ymin"] : self.parent.control.roi["ymax"]]
                        sc = np.sum(image_post_roi) # signal count
                    else:
                        image_post = None
                        image_post_roi = None
                        sc = None
                    self.img_dict["image_post"] = image_post
                    self.img_dict['image_post_roi'] = image_post_roi
                    self.img_dict["signal_count"] = sc
                
                    if self.parent.control.control_mode == "record":
                        if self.bkg_counter > 0:
                            # a list to save signal count of every single image
                            self.signal_count_list.append(sc)

                            img_counter = self.counter - self.bkg_counter
                            self.img_ave = np.average(np.array([self.img_ave, image_post]), axis=0, 
                                                    weights=[img_counter/(img_counter+1), 1/(img_counter+1)])

                            signal_count_ave = np.mean(self.signal_count_list)
                            signal_count_err = np.std(self.signal_count_list)/np.sqrt(len(self.signal_count_list))
                        
                        else:
                            signal_count_ave = None
                            signal_count_err = None
                        
                        self.img_dict["image_ave"] = self.img_ave
                        # signal count statistics, mean and error of mean = stand. dev. / sqrt(image number)
                        self.img_dict["signal_count_ave"] = signal_count_ave
                        self.img_dict["signal_count_err"] = signal_count_err
                        # self.img_dict["scan_param"] = ""

                    if self.parent.control.control_mode == "scan":
                        scan_param = self.scan_config[f"scan_value_{self.counter}"][self.parent.control.scan_elem_name]
                        if sc:
                            if scan_param in self.signal_count_dict:
                                self.signal_count_dict[scan_param] = np.append(self.signal_count_dict[scan_param], sc)
                            else:
                                self.signal_count_dict[scan_param] = np.array([sc])

                        self.img_dict["signal_count_scan"] = self.signal_count_dict
                        # self.img_dict["scan_param"] = scan_param                        

                    self.signal.emit(self.img_dict)

                else:
                    logging.warning("Image type not supported.")

                
                # If I call "update imge" function here to update images in main thread, it sometimes work but sometimes not.
                # It may be because PyQt is not thread safe. A signal-slot way seemed to be preferred,
                # e.g. https://stackoverflow.com/questions/54961905/real-time-plotting-using-pyqtgraph-and-threading

                logging.info(f"image {self.counter+1}: "+"{:.5f} s".format(time.time()-self.last_time))
                self.counter += 1

        # stop the camera after taking required number of images.
        self.parent.device.cam.stop()