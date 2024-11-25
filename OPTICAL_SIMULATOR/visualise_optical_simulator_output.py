import numpy as np
import scipy.io
import sys
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import dvs_warping_package


filename = "ev_epoch_0_3.0_500.0_350.0_0.6_0.5_0.02"
events = np.loadtxt(f"OUTPUT/events_and_labels/{filename}.txt", dtype=int)
width,height = (max(events[:,0])+1,max(events[:,1])+1)

events_data = np.dtype([('x', 'f4'), ('y', 'f4'), ('p', 'f4'), ('t', 'f4'), ('l', 'i4')])
events_array = np.zeros(len(events[:,3]), dtype=events_data)

events_array['t'] = events[:,3].flatten()
events_array['x'] = events[:,0].flatten()
events_array['y'] = events[:,1].flatten()
events_array['p'] = events[:,2].flatten()
events_array['l'] = events[:,-1].flatten()

t_max = events_array['t'][-1]
time_range = np.where((events_array['t']>=0, events_array['t']<=t_max))[1]
selected_events = events_array[time_range]


vx_velocity = np.zeros((len(selected_events["x"]), 1)) + 0.0 / 1e6
vy_velocity = np.zeros((len(selected_events["y"]), 1)) + 0.0 / 1e6
cumulative_map_object_zero, seg_label_zero = dvs_warping_package.accumulate_cnt_rgb((width,height),
                                                                                     selected_events,
                                                                                     selected_events['l'].flatten().astype(np.int32),
                                                                                     (vx_velocity.T,vy_velocity.T))
warped_image_segmentation_rgb_zero = dvs_warping_package.rgb_render_advanced(cumulative_map_object_zero, seg_label_zero)
warped_image_segmentation_rgb_zero.show()
print("/..")
