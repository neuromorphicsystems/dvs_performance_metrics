import numpy as np
import dsi

from event_buffer import EventBuffer


def init_bgn_hist_cpp(filename_noise_pos, filename_noise_neg):
    """ Load the distribution of the noise for the cpp simulator
        Pick randomly one noise distribution for each pixel and initialize also randomly the phases of the
        background noise
        Args:
            filename_noise_pos: path of the positive noise's filename
            filename_noise_neg: path of the negative noise's filename
        """
    noise_pos = np.load(filename_noise_pos)
    noise_pos = np.reshape(noise_pos, (noise_pos.shape[0] * noise_pos.shape[1], noise_pos.shape[2]))
    if len(noise_pos) == 0:
        print(filename_noise_pos, " is not correct")
        return
    noise_neg = np.load(filename_noise_neg)
    noise_neg = np.reshape(noise_neg, (noise_neg.shape[0] * noise_neg.shape[1], noise_neg.shape[2]))
    if len(noise_neg) == 0:
        print(filename_noise_neg, " is not correct")
        return

    dsi.initNoise(noise_pos, noise_neg)


class DvsSensor:
    def __init__(self, name):
        self.dsi = dsi
        self.name = name

    def initCamera(self, size_y, size_x,
                   lat=100, jit = 10, ref = 100, tau = 300, th_neg = 0.3, th_pos = 0.3, th_noise = 0.01
                   , bgnp=0.1, bgnn=0.01):
        dsi.initSimu(size_y, size_x)
        dsi.initLatency(lat, jit, ref, tau)
        dsi.initContrast(th_pos, th_neg, th_noise)
        self.img = np.zeros((size_y, size_x), dtype=np.uint8)
        self.img[:, :] = 125
        self.img[55:155, 55:155] = 170
        dsi.initImg(self.img)

    def init_bgn_hist(self, filename_noise_pos, filename_noise_neg):
        """ Load measured distributions of the noise,
            Pick randomly one noise distribution for each pixel and Initialise also randomly the phases of the
            background noise
            Args:
                filename_noise_pos: path of the positive noise's filename
                filename_noise_neg: path of the negative noise's filename
            """
        init_bgn_hist_cpp(filename_noise_pos, filename_noise_neg)

    def init_image(self, img):
        """ Initialise the first flux values of the sensor
        Args:
            img: image whose greylevel corresponds to a radiometric value
            It is assumed the maximum radiometric value is 1e6
        """
        dsi.initImg(img.transpose())
    def update(self, im, dt):
        """ Update the sensor with a nef irradiance's frame
            Follow the ICNS model
            Args:
                dt: delay between the frame and the last one (us)
            Returns:
                EventBuffer of the created events
             """
        self.img = im.transpose()
        buf = dsi.updateImg(self.img, dt)
        ev_buf = EventBuffer(1)
        ev_buf.add_array(np.array(buf["ts"], dtype=np.uint64),
                         np.array(buf["y"], dtype=np.uint16),
                         np.array(buf["x"], dtype=np.uint16),
                         np.array(buf["p"], dtype=np.uint64),
                         100000000)
        return ev_buf

    def getInfo(self):
        s = dsi.getShape()
        print(s)
        s = dsi.getCurv()
        print(s)

