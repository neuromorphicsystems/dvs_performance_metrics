import loris
import numpy as np
import matplotlib.pyplot as plt 
import os
import fnmatch
from tqdm import tqdm
import scipy.io as sio
import dvs_warping_package
import sys
sys.path.append("EVENT_SIMULATOR/src")
from dat_files import load_dat_event

FILENAME = "3.0_500.0_350.0_0.6_0.5_0.02"
events_path = "OUTPUT/events"
label_path = "OUTPUT/labels"


# my_file = loris.read_file(f"{events_path}/ev_{FILENAME}.dat")
ts, x, y, p = load_dat_event(f"{events_path}/ev_{FILENAME}.dat")
labels = np.loadtxt(f"{label_path}/labels_{FILENAME}.txt")


vx_velocity = np.zeros((len(x), 1)) + 0.0 / 1e6
vy_velocity = np.zeros((len(y), 1)) + 0.0 / 1e6

cumulative_map_object_zero, seg_label_zero = dvs_warping_package.accumulate_cnt_rgb((1280,720),events,labels.flatten().astype(np.int32),(vx_velocity.T,vy_velocity.T))
warped_image_segmentation_rgb_zero    = dvs_warping_package.rgb_render_advanced(cumulative_map_object_zero, seg_label_zero)
warped_image_segmentation_rgb_zero.show()
print("....")
