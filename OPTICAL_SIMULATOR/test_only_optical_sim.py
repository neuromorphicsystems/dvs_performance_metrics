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

from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import dvs_warping_package
from scipy.io import savemat
import scipy.io

sys.path.append("EVENT_SIMULATOR/src")
from event_buffer import EventBuffer
from dvs_sensor import DvsSensor
from event_display import EventDisplay
from arbiter import SynchronousArbiter, BottleNeckArbiter, RowArbiter


bgnp = 0.3 # ON event noise rate in events / pixel / s
bgnn = 0.3 # OFF event noise rate in events / pixel / s
psf_thr = 10

def run_simulation():    
    ini_file = 'config/simulation_config_1.ini'
    InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams, scanned_params = read_ini_file(ini_file)

    plt.ion()    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Two subplots, one for pixel_frame, one for EventDisplay

    simulation_data = []

    # Create the event buffer and arbiter
    ev_full = EventBuffer(1)
    
    if scanned_params:
        scanned_param_name = list(scanned_params.keys())
        scanned_param_values = list(scanned_params.values())
    else:
        scanned_param_name      = []
        scanned_param_values    = [0]

    for param_value_index in range(len(scanned_param_values[0])):
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
        
        # psf_peak = np.max(initial_target_frame)
        # threshold_value = psf_peak / psf_thr  # default value is 2.5
        # binary_target_mask = initial_target_frame > threshold_value
        # current_data = {
        #         't': Dynamics['t'],
        #         'i_azimuth': Dynamics['i_azimuth'],
        #         'i_elevation': Dynamics['i_elevation'],
        #         't_azimuth': Dynamics['t_azimuth'],
        #         't_elevation': Dynamics['t_elevation'],
        #         'imaging_los_speed': Dynamics['imaging_los_speed'],
        #         'binary_target_mask': binary_target_mask,
        #         'x': 1,
        #         'y': 1,
        #         'p': 1,
        #         'ts': 1
        #     }
        # simulation_data.append(current_data)
        
        # Create the initial frame and setup simulation plots
        if param_value_index == 0:
            imgg1 = ax1.imshow(initial_frame, 
                               cmap='gray', 
                               animated=True)
            ax1.set_title(f'Pixel Frame at {1000 * t:.2f} ms')
            plt.colorbar(imgg1, ax=ax1)

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
        
        dvs.init_image(im)
        ea = SynchronousArbiter(0.1, SensorParams['time'], im.shape[0])
        
        # Create the display for events
        render_timesurface = 1
        ed = EventDisplay("Events", frame_size[1], frame_size[0], SensorParams['dt'], render_timesurface)

        # Simulation loop
        while t < t_end:
            simulation_data.append({'t': t})
            # Create the camera pixel frame and target mask
            pixel_frame, Dynamics, target_frame_norm = frame_sim_functions(Dynamics,
                                                                           InitParams,
                                                                           SceneParams,
                                                                           OpticParams,
                                                                           TargetParams,
                                                                           BgParams,
                                                                           SensorBiases,
                                                                           SensorParams)
            
            binary_frame = dvs_warping_package.binarize_target_frame(target_frame_norm, 
                                                                     OpticParams['PSF_size'],
                                                                     InitParams['multiplier'],
                                                                     extra_percentage=1000)
            
            # psf_peak = np.max(target_frame_norm)
            # threshold_value = psf_peak / psf_thr  # default value is 2.5
            # binary_target_mask = target_frame_norm > threshold_value
            
            binary_target_mask = binary_frame
            
            t = Dynamics['t']
            
            # Run event simulator using the current image frame
            ev = dvs.update(pixel_frame, SensorParams['dt'])
            
            # per event labelling based on the binary mask
            l = dvs_warping_package.label_events(binary_target_mask, ev.x, ev.y)
            
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
                'l': l
            }
            simulation_data[-1].update(current_data)
            
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

            # warped_image_segmentation_raw    = dvs_warping_package.rgb_render(cumulative_map_object, seg_label)
            # warped_image_segmentation_raw.show()
            
            # combined_image = dvs_warping_package.overlay_masks(warped_image_segmentation_raw, binary_target_mask)
            
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
            combined_image.save(f"OUTPUT/combined_image_{t:.3f}.png")
            warped_image_segmentation_rgb_zero.save(f"OUTPUT/warped_image_segmentation_rgb_zero_{t:.3f}.png")
            warped_image_segmentation_raw.save(f"OUTPUT/warped_image_segmentation_raw_{t:.3f}.png")
            
            
    
            # combined_image.save(f"OUTPUT/0_img_{t:.3f}.png")
            # warped_image_segmentation_raw.save(f"OUTPUT/1_img_{t:.3f}.png")
            
            
            # Update EventDisplay with events
            ed.update(ev, SensorParams['dt'])
            ev_full.increase_ev(ev)
            
            # Update initial frames to fit the new frames
            initial_frame = pixel_frame
            initial_target_frame = target_frame_norm
            
            # Plot the pixel frame
            imgg1.set_array(pixel_frame)
            ax1.set_title(f'Pixel Frame at {1000 * t:.2f} ms')

            # Plot the events in the second subplot
            ax2.clear()
            ax2.set_title("Event Display")
            ax2.imshow(ed.get_image(), cmap='gray')
            
            # Update the figures
            plt.pause(0.001)

        ev_full.write('OUTPUT/events/ev_{}_{}_{}_{}_{}.dat'.format(SensorParams['lat'], 
                                                               SensorParams['jit'], 
                                                               SensorBiases['diff_on'], 
                                                               SensorBiases['diff_off'], 
                                                               SensorParams['threshold_noise']))
        
        scipy.io.savemat('OUTPUT/masks/target_frame_mask.mat', {'target_frame_mask': simulation_data})

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run_simulation()
