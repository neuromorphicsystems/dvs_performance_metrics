# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:29:52 2024

@author: 30067913
"""

sys.path.append("EVENT_SIMULATOR/src")

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

from event_buffer import EventBuffer
from dvs_sensor import DvsSensor
from event_display import EventDisplay
from arbiter import SynchronousArbiter, BottleNeckArbiter, RowArbiter


bgnp = 0.1 # ON event noise rate in events / pixel / s
bgnn = 0.1 # OFF event noise rate in events / pixel / s

def run_simulation():    
    ini_file = 'config/simulation_config_1.ini'
    InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams, scanned_params = read_ini_file(ini_file)

    plt.ion()    
    fig, ax = plt.subplots()
    
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
        if param_value_index==0:
            imgg = ax.imshow(initial_frame, cmap='gray', animated=True)
            title = ax.set_title(f'Time: {1000*t:.2f} ms')
            plt.colorbar(imgg, ax=ax)#, label='Intensity')
        
        im = initial_frame

        dvs = DvsSensor("MySensor")
        dvs.initCamera(frame_size[1], 
                       frame_size[0],
                       lat = SensorParams['lat'], 
                       jit = SensorParams['jit'], 
                       ref = SensorBiases['refr'],
                       tau = SensorParams['tau_dark'], 
                       th_pos = SensorBiases['diff_on'], 
                       th_neg = SensorBiases['diff_off'], 
                       th_noise = SensorParams['threshold_noise'],
                       bgnp=bgnp, bgnn=bgnn, 
                       Idr = SensorParams['Idr'],
                       pp = SensorParams['pixel_pitch'], 
                       qe = SensorParams['QE'], 
                       ff = SensorParams['fill_factor'],
                       tsf = SensorParams['tau_sf'], 
                       tdr = SensorParams['tau_dark'], 
                       q=SensorParams['q'])
        
        dvs.init_image(im)

        # Create the event buffer
        ev_full = EventBuffer(1)
        ea = SynchronousArbiter(0.1, SensorParams['time'], im.shape[0])

        # Create the display
        render_timesurface = 1
        ed = EventDisplay("Events",
                        frame_size[1], 
                        frame_size[0],
                        SensorParams['dt'],
                        render_timesurface)
        
        # Run the simulation loop until t exceeds t_end
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
            
            # Classify events to "target" and "background" according to 2 target masks
            # Classify the events = 2 frames  +event in between (target mask with PSF)
            
            
            # frame_timestamp.append((ev.ts[0],ev.ts[-2]))
            ed.update(ev, SensorParams['dt'])
            # Add the events to the buffer for the full video
            ev_full.increase_ev(ev)
            
            
            
            
            # Update initial frames to fit the new frames
            initial_frame = pixel_frame
            initial_target_frame = target_frame_norm
            
            # Plot new simulation frame (and classified events as well?)
            imgg.set_array(pixel_frame)
            # title.set_text(f'Time: {1000*t:.2f} ms')
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            # plt.pause(0.001)


plt.ioff()
plt.show()
    
# save event stream to file (put in folder together with an initial frame and param .ini file)
        

if __name__ == '__main__':
    run_simulation()
