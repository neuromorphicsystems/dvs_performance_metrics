# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:29:52 2024

@author: 30067913
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image, ImageEnhance, ImageOps
import argparse

#import loris
from tqdm import tqdm
import os
import dvs_warping_package
from scipy.io import savemat

sys.path.append("EVENT_SIMULATOR/src")
sys.path.append("OPTICAL_SIMULATOR")
from basic_tar_bg_simulation import frame_sim_functions,initialize_simulation_params,read_ini_file
from event_buffer import EventBuffer
from dvs_sensor import DvsSensor
from event_display import EventDisplay
from arbiter import SynchronousArbiter, BottleNeckArbiter, RowArbiter

DO_PLOTS    = 1 # Are we ploting or are we not - turn off for server running
EPOCH       = 1 #how many time you wanna run the same experiment
bgnp = 0.3 # ON event noise rate in events / pixel / s
bgnn = 0.3 # OFF event noise rate in events / pixel / s

def run_simulation(config_file_name):    
    ini_file = f"config/{config_file_name}.ini"
    InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams, scanned_params = read_ini_file(ini_file)
    os.makedirs(f"OUTPUT/{config_file_name}", exist_ok=True)

    if DO_PLOTS:
        plt.ion()    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Two subplots, one for pixel_frame, one for EventDisplay

    # simulation_data = []
    
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

        # Create the event buffer and arbiter
        ev_full = EventBuffer(1)
        
        # Initialize simulation data function
        frame_size = [SensorParams['height'], SensorParams['width']]
        Dynamics, InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams = initialize_simulation_params(InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams)
        
        
        # first frame is outside the loop
        initial_frame, Dynamics, initial_target_frame = frame_sim_functions(Dynamics, InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams)
        initial_binary_mask = dvs_warping_package.create_binary_mask(initial_target_frame)
        maskMemorySize = int(round(SensorParams['tau_sf']/InitParams['dt']/4))
        binary_target_mask_memory = np.zeros([np.size(initial_binary_mask,0),np.size(initial_binary_mask,1),maskMemorySize])
        binary_target_mask_memory[:,:,0] = initial_binary_mask

        # Store the target initial data
        current_data = {
            't': Dynamics['t'],
            'i_azimuth': Dynamics['i_azimuth'],
            'i_elevation': Dynamics['i_elevation'],
            't_azimuth': Dynamics['t_azimuth'],
            't_elevation': Dynamics['t_elevation'],
            'imaging_los_speed': Dynamics['imaging_los_speed'],
            'binary_target_mask': initial_binary_mask,
            'pixel_offset_x': OpticParams['focal_length']/SensorParams['pixel_pitch']*(Dynamics['i_azimuth']-Dynamics['t_azimuth']),
            'pixel_offset_y': OpticParams['focal_length']/SensorParams['pixel_pitch']*(Dynamics['i_elevation']-Dynamics['t_elevation']),
            # 'x': ev.x,
            # 'y': ev.y,
            # 'p': ev.p,
            # 'ts': ev.ts,
            # 'l': l
        }
        simulation_data = []
        simulation_data.append(current_data)

        all_labels = []
        
        # Create the initial frame and setup simulation plots
        if DO_PLOTS:
            if param_value_index == 0:
                imgg1 = ax1.imshow(initial_frame,
                                   cmap='gray',
                                   animated=True)
                ax1.set_title(f'Pixel Frame at {1000 * 0:.2f} ms')
                plt.colorbar(imgg1, ax=ax1)

        # Initialize DVS sensor
        dvs = DvsSensor("MySensor")
        dvs.initCamera(frame_size[1], 
                       frame_size[0],
                       lat=SensorParams['lat']*1e6, 
                       jit=SensorParams['latency_jitter']*1e6, 
                       ref=SensorBiases['refr'],
                       tau=SensorParams['tau_dark']*1e6, 
                       th_pos=SensorBiases['diff_on'], 
                       th_neg=SensorBiases['diff_off'], 
                       th_noise=SensorParams['threshold_noise'],
                       bgnp=bgnp, bgnn=bgnn, 
                       Idr=SensorParams['I_dark'],
                       pp=SensorParams['pixel_pitch'], 
                       qe=SensorParams['QE'], 
                       ff=SensorParams['fill_factor'],
                       tsf=SensorParams['tau_sf']*1e6, 
                       tdr=SensorParams['tau_dark']*1e6, 
                       q=1.602176634e-19)
        
        dvs.init_image(initial_frame)
        ea = SynchronousArbiter(0.1, 0, initial_frame.shape[0])
        
        # Create the display for events
        render_timesurface = 1
        dt_us = InitParams['dt']*1e6
        if DO_PLOTS:
            ed = EventDisplay("Events", frame_size[1], frame_size[0], dt_us, render_timesurface)

        # Epoc run in test
        # for ep in range(EPOCH):

        # Single simulation run	in epoch
        counter = 1
        while Dynamics['t'] < InitParams['t_end']:           
                # Create new intensity image frame, current target mask, and update dynamic parameters
                pixel_frame, Dynamics, target_frame_norm = frame_sim_functions(Dynamics,
                                                                                InitParams,
                                                                                SceneParams,
                                                                                OpticParams,
                                                                                TargetParams,
                                                                                BgParams,
                                                                                SensorBiases,
                                                                                SensorParams)
                
                # Update the memory binary mask to include new mask
                binary_target_mask_memory[:,:, counter % maskMemorySize] = dvs_warping_package.create_binary_mask(target_frame_norm)
                binary_target_mask = np.any(binary_target_mask_memory, axis = 2)

                t = Dynamics['t']

                # Run event simulator using the current image frame
                ev = dvs.update(pixel_frame, dt_us)
                ev_full.increase_ev(ev)

                # per event labelling based on the binary mask
                lables = dvs_warping_package.label_events(binary_target_mask, ev.x, ev.y)

                event_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('p', 'f4'), ('ts', 'f4'), ('l', 'i4')])
                events_array = np.zeros(len(ev.x), dtype=event_dtype)
                events_array['x']  = ev.x.flatten()
                events_array['y']  = ev.y.flatten()
                events_array['p']  = ev.p.flatten()
                events_array['ts'] = ev.ts.flatten()
                events_array['l']  = np.array(lables).flatten()
                
                all_labels.extend(lables.tolist())
            
                # Store the pixel displacement in the current data
                current_data = {
                    't': Dynamics['t'],
                    'i_azimuth': Dynamics['i_azimuth'],
                    'i_elevation': Dynamics['i_elevation'],
                    't_azimuth': Dynamics['t_azimuth'],
                    't_elevation': Dynamics['t_elevation'],
                    'imaging_los_speed': Dynamics['imaging_los_speed'],
                    'binary_target_mask': binary_target_mask,
                    'pixel_offset_x': OpticParams['focal_length']/SensorParams['pixel_pitch']*(Dynamics['i_azimuth']-Dynamics['t_azimuth']),
                    'pixel_offset_y': OpticParams['focal_length']/SensorParams['pixel_pitch']*(Dynamics['i_elevation']-Dynamics['t_elevation']),
                }
                simulation_data.append(current_data)
                
                # advance counter
                counter += 1    

                # plot graphics for debuging
                if DO_PLOTS>1:
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
                                                                                                        labeled_events.flatten().astype(np.int32),
                                                                                                        (vx_velocity.T,vy_velocity.T))
                    warped_image_segmentation_rgb_zero    = dvs_warping_package.rgb_render_advanced(cumulative_map_object_zero, seg_label_zero)
                
                    
                    only_sig = np.where(l==1)
                    cumulative_map_object_zero, seg_label_zero = dvs_warping_package.accumulate_cnt_rgb((binary_target_mask.shape[1],
                                                                                                        binary_target_mask.shape[0]),
                                                                                                        eventsT[only_sig],
                                                                                                        labeled_events[only_sig].flatten().astype(np.int32),
                                                                                                        (vx_velocity[only_sig].T,vy_velocity[only_sig].T))
                    warped_image_only_signal    = dvs_warping_package.rgb_render_advanced(cumulative_map_object_zero, seg_label_zero)
                    
                    
                    only_bckg = np.where(l==0)
                    cumulative_map_object_zero, seg_label_zero = dvs_warping_package.accumulate_cnt_rgb((binary_target_mask.shape[1],
                                                                                                        binary_target_mask.shape[0]),
                                                                                                        eventsT[only_bckg],
                                                                                                        labeled_events[only_bckg].flatten().astype(np.int32),
                                                                                                        (vx_velocity[only_bckg].T,vy_velocity[only_bckg].T))
                    warped_image_only_bckg    = dvs_warping_package.rgb_render_advanced(cumulative_map_object_zero, seg_label_zero)
                
                    
                    warped_image_only_signal.save(f"OUTPUT/only_signal/TargetParams_{TargetParams['target_radius']}_only_sig_{t:.3f}.png")
                    warped_image_only_bckg.save(f"OUTPUT/only_background/TargetParams_{TargetParams['target_radius']}_only_sig_{t:.3f}.png")
                    combined_image.save(f"OUTPUT/mask_overlay/TargetParams_{TargetParams['target_radius']}_combined_image_{t:.3f}.png")
                    warped_image_segmentation_rgb_zero.save(f"OUTPUT/labeled_image/TargetParams_{TargetParams['target_radius']}_warped_image_segmentation_rgb_zero_{t:.3f}.png")                
                elif DO_PLOTS:

                    # plot event frame in own frame
                    #ed.update(ev, dt_us)

                    # plot intensity frame in first subplot
                    imgg1.set_array(pixel_frame)
                    ax1.set_title(f'Pixel Frame at {1000 * t:.2f} ms')

                    # Plot the events in the second subplot
                    ax2.clear()
                    ax2.set_title("Event Display")
                    ax2.imshow(ed.get_image(), cmap='gray')

                    # Update the figures
                    plt.pause(0.001)           
            
                # Update initial frames to fit the new frames
                initial_frame = pixel_frame
                
                initial_target_frame = target_frame_norm

        
                simulation_data.append({'all_events': np.array(all_labels)}) # add events to mat file
                #all_labels = np.array(final_events)

        if scanned_params:
            Out_file_name = f"{InitParams['sim_name']}_{param}_{scanned_param_values[0][param_value_index]}"
        else:
            Out_file_name = f"{InitParams['sim_name']}"
        #dvs_warping_package.save_to_es(ev_full, f"OUTPUT/{config_file_name}/ev_{Out_file_name}.es")
        ev_full.write(f"OUTPUT/{config_file_name}/ev_{Out_file_name}.dat")
        np.savetxt(f"OUTPUT/{config_file_name}/labels_{Out_file_name}.txt", all_labels, fmt='%d')
        savemat(f"OUTPUT/{config_file_name}/simdata_{Out_file_name}.mat", {"simulation_data": simulation_data})

	
    if DO_PLOTS:
        plt.ioff() 
        plt.show()


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Run simulation with config file.')
    
    # Add a positional argument for the config file name
    parser.add_argument('config_file_name', type=str, help='The path to the configuration file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with the config file name provided from the command line
    run_simulation(args.config_file_name)