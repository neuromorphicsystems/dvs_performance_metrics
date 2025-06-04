import numpy as np
import scipy.io
import sys
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import dvs_warping_package

EPOCH = 2
target_radius = [3,4,5,6]
lat = 500.0
jit = 350.0
diff_on = 0.6
diff_off = 0.5
threshold_noise = 0.02

total_signal        = np.zeros((len(target_radius), EPOCH))
total_background    = np.zeros((len(target_radius), EPOCH))
total_noise         = np.zeros((len(target_radius), EPOCH))

if not os.path.exists(f"OUTPUT/data_processing"):
    os.makedirs(f"OUTPUT/data_processing", exist_ok=True) 

for ep in range(0,EPOCH):
    for tr_rad in tqdm(range(len(target_radius))):
        dvs_warping_package.print_message(f"epoch: {ep} - target_radius: {float(target_radius[tr_rad])} lat: {float(lat)} jit: {float(jit)} diff_on: {float(diff_on)} diff_off: {float(diff_off)} threshold_noise: {float(threshold_noise)}", color='yellow', style='bold')

        base_filename = f"ev_epoch_{ep}_{float(target_radius[tr_rad])}_{float(lat)}_{float(jit)}_{float(diff_on)}_{float(diff_off)}_{float(threshold_noise)}"
        events = np.loadtxt(f"OUTPUT/events_and_labels/{base_filename}.txt", dtype=int)

        width, height = (max(events[:,0])+1,max(events[:,1])+1)

        events_data = np.dtype([('x', 'f4'), ('y', 'f4'), ('p', 'f4'), ('t', 'f4'), ('l', 'i4')])
        events_array = np.zeros(len(events[:,3]), dtype=events_data)

        events_array['t'] = events[:,3].flatten()
        events_array['x'] = events[:,0].flatten()
        events_array['y'] = events[:,1].flatten()
        events_array['p'] = events[:,2].flatten()
        events_array['l'] = events[:,-1].flatten()
        
        vx_velocity = np.zeros((len(events_array["x"]), 1)) + 0.0 / 1e6
        vy_velocity = np.zeros((len(events_array["y"]), 1)) + 0.0 / 1e6
        
        sig = np.where(events_array["l"]==-1)[0]
        cumulative_map_object_zero, seg_label_zero = dvs_warping_package.accumulate_cnt_rgb((width,height),
                                                                                             events_array[sig],
                                                                                             events_array['l'][sig].flatten().astype(np.int32),
                                                                                             (vx_velocity[sig].T,vy_velocity[sig].T))
        warped_image_segmentation_rgb_zero = dvs_warping_package.rgb_render_advanced(cumulative_map_object_zero, seg_label_zero)
        warped_image_segmentation_rgb_zero.save(f"OUTPUT/data_processing/sig_{base_filename}.png")
        
        # calculate variance (diff(background - forground))
        total_signal[tr_rad,ep] = len(np.where(events_array["l"]==-1)[0])
        total_background[tr_rad,ep] = len(np.where(events_array["l"]==1)[0])
        total_noise[tr_rad,ep] = len(np.where(events_array["l"]==0)[0])
        

for i in range(len(target_radius)):
    signal_mean = np.mean(total_signal[i,:])
    signal_var = np.var(total_signal[i,:])
    
    background_mean = np.mean(total_background[i,:])
    background_var = np.var(total_background[i,:])

    noise_mean = np.mean(total_noise[i,:])
    noise_var = np.var(total_noise[i,:])
    
    dvs_warping_package.print_message(f"target_radius: {float(target_radius[i])} -> mean:  signal_mean: {signal_mean} background_mean: {background_mean} noise_mean: {noise_mean}", color='green', style='bold')
    dvs_warping_package.print_message(f"target_radius: {float(target_radius[i])} -> vari:  signal_var: {signal_var} background_var: {background_var} noise_var: {noise_var}", color='magenta', style='bold')



