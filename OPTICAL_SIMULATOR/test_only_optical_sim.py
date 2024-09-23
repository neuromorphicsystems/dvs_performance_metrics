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

from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import dvs_warping_package

sys.path.append("EVENT_SIMULATOR/src")
from event_buffer import EventBuffer
from dvs_sensor import DvsSensor
from event_display import EventDisplay
from arbiter import SynchronousArbiter, BottleNeckArbiter, RowArbiter


bgnp = 0.3 # ON event noise rate in events / pixel / s
bgnn = 0.3 # OFF event noise rate in events / pixel / s

def run_simulation():    
    ini_file = 'config/simulation_config_1.ini'
    InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams, scanned_params = read_ini_file(ini_file)

    plt.ion()    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Two subplots, one for pixel_frame, one for EventDisplay

    
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
        
        # Create the initial frame and setup simulation plots
        if param_value_index == 0:
            imgg1 = ax1.imshow(initial_frame, cmap='gray', animated=True)
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

        # Create the event buffer and arbiter
        ev_full = EventBuffer(1)
        ea = SynchronousArbiter(0.1, SensorParams['time'], im.shape[0])

        # Create the display for events
        render_timesurface = 1
        ed = EventDisplay("Events", frame_size[1], frame_size[0], SensorParams['dt'], render_timesurface)

        # Simulation loop
        while t < t_end:
            # Create the camera pixel frame and target mask
            pixel_frame, Dynamics, target_frame_norm = frame_sim_functions(Dynamics, 
                                                                           InitParams, 
                                                                           SceneParams, 
                                                                           OpticParams, 
                                                                           TargetParams, 
                                                                           BgParams, 
                                                                           SensorBiases, 
                                                                           SensorParams)
            t = Dynamics['t']
            
            # Run event simulator using the current image frame
            ev = dvs.update(pixel_frame, SensorParams['dt'])

            # Update EventDisplay with events
            ed.update(ev, SensorParams['dt'])
            
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

    plt.ioff()
    plt.show()
        

if __name__ == '__main__':
    run_simulation()
