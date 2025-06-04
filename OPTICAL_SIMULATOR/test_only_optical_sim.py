# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:29:52 2024

@author: 30067913
"""


import numpy as np
import matplotlib.pyplot as plt
from basic_tar_bg_simulation import frame_sim_functions,initialize_simulation_params,read_ini_file
import cv2
import sys
from PIL import Image, ImageEnhance, ImageOps
import loris
from tqdm import tqdm
import os
import dvs_warping_package
from scipy.io import savemat
import random

sys.path.append("EVENT_SIMULATOR/src")
from event_buffer import EventBuffer
from dvs_sensor import DvsSensor
from event_display import EventDisplay
from arbiter import SynchronousArbiter, BottleNeckArbiter, RowArbiter

# random.seed(10)

DO_PLOTS    = 1 # Are we ploting or are we not - turn off for server running
SAVE_FRAMES = 1 # Enable/Disable frame saving

EPOCH       = 1 # Number of run for the same experiment
blankFrames = 1 # Number of blank frames
# skipFrames  = 0 # Number of skipped frames

bgnp = 0.3 # ON event noise rate in events / pixel / s
bgnn = 0.3 # OFF event noise rate in events / pixel / s

# output_path = "./performance_metric/"
output_path = "OUTPUT"



def run_simulation():    
    ini_file = 'config/simulation_config_1.ini'
    InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams, scanned_params = read_ini_file(ini_file)
    
    if scanned_params:
        scanned_param_name = list(scanned_params.keys())
        scanned_param_values = list(scanned_params.values())
        range_of_scan = range(len(scanned_param_values[0]))
    else:
        scanned_param_name      = list(['.'])
        scanned_param_values    = list()
        range_of_scan = range(1)

    for param_value_index in range_of_scan:
        section, param = scanned_param_name[0].split('.')
        if section == 'InitParams':
            InitParams[param] = scanned_param_values[0][param_value_index]
        if section == 'SceneParams':
            SceneParams[param] = scanned_param_values[0][param_value_index]
        if section == 'OpticParams':
            OpticParams[param] = scanned_param_values[0][param_value_index]
        if section == 'TargetParams':
            TargetParams[param] = scanned_param_values[0][param_value_index]
        if section == 'BgParams':
            BgParams[param] = scanned_param_values[0][param_value_index]
        if section == 'SensorBiases':
            SensorBiases[param] = scanned_param_values[0][param_value_index]
        if section == 'SensorParams':
            SensorParams[param] = scanned_param_values[0][param_value_index]

        
        # Initialize simulation data function
        Dynamics, InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams = initialize_simulation_params(InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams)
        t = Dynamics['t']             # Start time (from initialized dynamics)
        t_end = InitParams['t_end']   # End time (from the INI file)

        initial_frame, Dynamics, initial_target_frame = frame_sim_functions(Dynamics, InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams)
        frame_size = [SensorParams['height'], SensorParams['width']]
        
        simulation_data  = []
        final_events     = []    

        im = initial_frame

        # Initialize DVS sensor
        dvs = DvsSensor("MySensor")
        dvs.initCamera(frame_size[1], 
                       frame_size[0],
                       lat=SensorParams['lat'],
                       jit=SensorParams['jit'],
                       ref=SensorBiases['refr'],
                       tau=SensorParams['tau_dark'],
                       th_pos=SensorBiases['diff_on'],
                       th_neg=SensorBiases['diff_off'],
                       th_noise=SensorParams['threshold_noise'],
                       bgnp=bgnp, bgnn=bgnn,
                       Idr=SensorParams['Idr'],
                       pp=SensorParams['pixel_pitch'],
                       qe=SensorParams['QE'],
                       ff=SensorParams['fill_factor'],
                       tsf=SensorParams['tau_sf'],
                       tdr=SensorParams['tau_dark'],
                       q=SensorParams['q'])
        
        dvs.init_bgn_hist("EVENT_SIMULATOR/data/noise_pos_161lux.npy", "EVENT_SIMULATOR/data/noise_neg_161lux.npy")

        
        dvs.init_image(im)
        
        # Create the event buffer and arbiter
        ev_full = EventBuffer(1)
    
        ea = SynchronousArbiter(0.1, SensorParams['time'], im.shape[0])
        
        # Create the display for events
        render_timesurface = 1
        ed = EventDisplay("Events", frame_size[1], frame_size[0], SensorParams['dt'], render_timesurface)
        dirs = [
                f"{output_path}/events_and_labels",f"{output_path}/only_signal_image",f"{output_path}/only_background_image",f"{output_path}/only_noise_image",f"{output_path}/mask_overlay",f"{output_path}/labeled_image"
        ]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)                

        for ep in range(EPOCH):
            
            dvs_warping_package.print_message(f"epoch: {ep} - target_radius: {TargetParams['target_radius']} lat: {SensorParams['lat']} lat: {SensorParams['lat']} diff_on: {SensorBiases['diff_on']} diff_off: {SensorBiases['diff_off']} threshold_noise: {SensorParams['threshold_noise']}", color='yellow', style='bold')
            
            # Simulation loop
            while t < t_end:
                simulation_data.append({'t': t})
                # Create blank black frames at the beginning
                if len(simulation_data) < blankFrames:
                    
                    ev, ev_signal, ev_noise, ground_truth = dvs.update(im, SensorParams['dt'])
                    target_frame_norm = np.zeros((SensorParams['height'], SensorParams['width']))
                    binary_image_mask = np.zeros((SensorParams['height'], SensorParams['width']), dtype=np.uint8)
                    ed.update(ev, SensorParams['dt']) # Display the events
                    ev_full.increase_ev(ev) # Add the events to the buffer for the full video
                else:
                    # Create the camera pixel frame and target mask with useful information
                    pixel_frame, Dynamics, target_frame_norm = frame_sim_functions(Dynamics,
                                                                                    InitParams,
                                                                                    SceneParams,
                                                                                    OpticParams,
                                                                                    TargetParams,
                                                                                    BgParams,
                                                                                    SensorBiases,
                                                                                    SensorParams)
                     # Skip the first few frames based on "skipFrames"
                    # if len(simulation_data) >= skipFrames:
                    binary_image_mask = dvs_warping_package.create_binary_mask(target_frame_norm)

                    binary_target_mask = binary_image_mask
                    t = Dynamics['t']

                    # Run event simulator using the current image frame
                    ev, ev_signal, ev_noise, ground_truth = dvs.update(pixel_frame, SensorParams['dt'])
                    
                    # Update EventDisplay with events
                    ed.update(ev, SensorParams['dt']) # Display the events
                    ev_full.increase_ev(ev) # Add the events to the buffer for the full video
                    
                    
                    
                    ev_signal = dvs_warping_package.ev_sorting(ev_signal)
                    ev_noise  = dvs_warping_package.ev_sorting(ev_noise)

                    # Per-event labeling based on the binary mask
                    l = dvs_warping_package.label_events(binary_target_mask, ev.x, ev.y)
                    
                    # the labels are structure like this:
                    # 0 for only noise
                    # 1 for background
                    # -1 for moving target
                    final_l = np.where(l == -1, -1, ground_truth)

                    event_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('p', 'f4'), ('t', 'f4'), ('l', 'i4')])
                    events_array = np.zeros(len(ev.x), dtype=event_dtype)
                    events_array['x'] = ev.x.flatten()
                    events_array['y'] = ev.y.flatten()
                    events_array['p'] = ev.p.flatten()
                    events_array['t'] = ev.ts.flatten()
                    events_array['l'] = final_l #l
                    
                    final_events.extend(events_array.tolist())

                    # Store the pixel displacement in the current data
                    current_data = {
                        'i_azimuth': Dynamics['i_azimuth'],
                        'i_elevation': Dynamics['i_elevation'],
                        't_azimuth': Dynamics['t_azimuth'],
                        't_elevation': Dynamics['t_elevation'],
                        'imaging_los_speed': Dynamics['imaging_los_speed'],
                        'binary_target_mask': binary_target_mask,
                        'dx/dt': OpticParams['focal_length']/SensorParams['pixel_pitch']*(Dynamics['i_azimuth']-Dynamics['t_azimuth']),
                        'dy/dt': OpticParams['focal_length']/SensorParams['pixel_pitch']*(Dynamics['i_elevation']-Dynamics['t_elevation']),
                        'x': ev.x,
                        'y': ev.y,
                        'p': ev.p,
                        'ts': ev.ts,
                        'l': final_l #l
                    }
                    simulation_data[-1].update(current_data)                
                    
                    if SAVE_FRAMES:
                        eventsT = np.zeros(len(ev.x), dtype=[('t', 'f8'),
                                                            ('x', 'f8'),
                                                            ('y', 'f8'),
                                                            ('p', 'f8')])
                        
                        eventsT['t'] = ev.ts.astype(np.float64)
                        eventsT['x'] = ev.x.astype(np.float64)
                        eventsT['y'] = ev.y.astype(np.float64)
                        eventsT['p'] = ev.p.astype(np.float64)
                        
                        vx_velocity = np.zeros((len(eventsT["x"]), 1)) + 0.0 / 1e6
                        vy_velocity = np.zeros((len(eventsT["y"]), 1)) + 0.0 / 1e6
                        
                        
                        ########## only for ev_signal, ev_noise
                        events_signal = np.zeros(len(ev_signal.x), dtype=[('t', 'f8'),
                                                            ('x', 'f8'),
                                                            ('y', 'f8'),
                                                            ('p', 'f8')])
                        
                        events_signal['t'] = ev_signal.ts.astype(np.float64)
                        events_signal['x'] = ev_signal.x.astype(np.float64)
                        events_signal['y'] = ev_signal.y.astype(np.float64)
                        events_signal['p'] = ev_signal.p.astype(np.float64)
                        
                        vx_velocity_signal = np.zeros((len(events_signal["x"]), 1)) + 0.0 / 1e6
                        vy_velocity_signal = np.zeros((len(events_signal["y"]), 1)) + 0.0 / 1e6
                        
                        
                        events_noise = np.zeros(len(ev_noise.x), dtype=[('t', 'f8'),
                                                            ('x', 'f8'),
                                                            ('y', 'f8'),
                                                            ('p', 'f8')])
                        
                        events_noise['t'] = ev_noise.ts.astype(np.float64)
                        events_noise['x'] = ev_noise.x.astype(np.float64)
                        events_noise['y'] = ev_noise.y.astype(np.float64)
                        events_noise['p'] = ev_noise.p.astype(np.float64)
                        
                        vx_velocity_noise = np.zeros((len(events_noise["x"]), 1)) + 0.0 / 1e6
                        vy_velocity_noise = np.zeros((len(events_noise["y"]), 1)) + 0.0 / 1e6
                        
                        cumulative_map_object = dvs_warping_package.accumulate((binary_target_mask.shape[1],
                                                                                binary_target_mask.shape[0]),
                                                                                eventsT,
                                                                                (0,0))
                        warped_image_segmentation_raw = dvs_warping_package.render(cumulative_map_object,
                                                                                colormap_name="magma",
                                                                                gamma=lambda image: image ** (1 / 3))
                        
                        combined_image, labeled_events = dvs_warping_package.overlay_and_label_events(warped_image_segmentation_raw,
                                                                                                    binary_target_mask,
                                                                                                    ev.x,
                                                                                                    ev.y,
                                                                                                    y_offset=0)

                        cumulative_map_object_zero, seg_label_zero = dvs_warping_package.accumulate_cnt_rgb((binary_target_mask.shape[1],
                                                                                                            binary_target_mask.shape[0]),
                                                                                                            eventsT,
                                                                                                            final_l.flatten().astype(np.int32),
                                                                                                            (vx_velocity.T,vy_velocity.T))
                        warped_image_segmentation_rgb_zero    = dvs_warping_package.rgb_render_advanced(cumulative_map_object_zero, seg_label_zero)
                        
                        
                        only_sig = np.where(final_l==-1)
                        cumulative_map_object_zero, seg_label_zero = dvs_warping_package.accumulate_cnt_rgb((binary_target_mask.shape[1],
                                                                                                            binary_target_mask.shape[0]),
                                                                                                            eventsT[only_sig],
                                                                                                            final_l[only_sig].flatten().astype(np.int32),
                                                                                                            (vx_velocity[only_sig].T,vy_velocity[only_sig].T))
                        warped_image_only_signal    = dvs_warping_package.rgb_render_advanced(cumulative_map_object_zero, seg_label_zero)
                        
                        
                        only_bckg = np.where(final_l==1)
                        cumulative_map_object_zero, seg_label_zero = dvs_warping_package.accumulate_cnt_rgb((binary_target_mask.shape[1],
                                                                                                            binary_target_mask.shape[0]),
                                                                                                            eventsT[only_bckg],
                                                                                                            final_l[only_bckg].flatten().astype(np.int32),
                                                                                                            (vx_velocity[only_bckg].T,vy_velocity[only_bckg].T))
                        warped_image_only_bckg    = dvs_warping_package.rgb_render_advanced(cumulative_map_object_zero, seg_label_zero)
                        
                        only_noise = np.where(final_l==0)
                        cumulative_map_object_zero, seg_label_zero = dvs_warping_package.accumulate_cnt_rgb((binary_target_mask.shape[1],
                                                                                                            binary_target_mask.shape[0]),
                                                                                                            eventsT[only_noise],
                                                                                                            final_l[only_noise].flatten().astype(np.int32),
                                                                                                            (vx_velocity[only_noise].T,vy_velocity[only_noise].T))
                        warped_image_only_noise    = dvs_warping_package.rgb_render_advanced(cumulative_map_object_zero, seg_label_zero)
                        
                        
                        
                        warped_image_only_signal.save(f"{output_path}/only_signal_image/TargetParams_{TargetParams['target_radius']}_only_sig_{t:.3f}.png")
                        warped_image_only_bckg.save(f"{output_path}/only_background_image/TargetParams_{TargetParams['target_radius']}_only_backg_{t:.3f}.png")
                        warped_image_only_noise.save(f"{output_path}/only_noise_image/TargetParams_{TargetParams['target_radius']}_only_noise_{t:.3f}.png")
                        combined_image.save(f"{output_path}/mask_overlay/TargetParams_{TargetParams['target_radius']}_combined_image_{t:.3f}.png")
                        warped_image_segmentation_rgb_zero.save(f"{output_path}/labeled_image/TargetParams_{TargetParams['target_radius']}_warped_image_segmentation_rgb_zero_{t:.3f}.png")                

                        # Update initial frames to fit the new frames
                        initial_frame = pixel_frame
                        initial_target_frame = target_frame_norm
            
            
            base_filename = f"epoch_{ep}_{TargetParams['target_radius']}_{SensorParams['lat']}_{SensorParams['jit']}_{SensorBiases['diff_on']}_{SensorBiases['diff_off']}_{SensorParams['threshold_noise']}"

            final_events_array = np.array(final_events)
            simulation_data.append({'all_events': final_events_array})  
            
            # dvs_warping_package.save_to_es(final_events_array, output_path, base_filename)
            np.savetxt(f"{output_path}/events_and_labels/ev_{base_filename}.txt", final_events_array, fmt='%d')
            
            print(f"Finish saving data for: {base_filename}")
            
            # ev_full.write(f"OUTPUT/events/ev_{base_filename}.dat")
            # savemat(f"OUTPUT/masks/simulation_data_target_radius_{TargetParams['target_radius']}.mat",
            #                  {"simulation_data": simulation_data})
        
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    run_simulation()