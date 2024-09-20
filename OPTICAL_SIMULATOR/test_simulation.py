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
sys.path.append("EVENT_SIMULATOR/src")

from event_buffer import EventBuffer
# from dvs_sensor_cpp import DvsSensor
from dvs_sensor import DvsSensor
from event_display import EventDisplay
from arbiter import SynchronousArbiter, BottleNeckArbiter, RowArbiter

from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import scipy.io as sio


# All these need to go - all inputs are now from the config file
th_pos = 0.3        # ON threshold = 50% (ln(1.5) = 0.4)
th_neg = 0.4        # OFF threshold = 50%
th_noise = 0.01     # standard deviation of threshold noise
lat = 500           # latency in us
tau = 10            # front-end time constant at 1 klux in us
jit = 50            # temporal jitter standard deviation in us
bgnp = 0.01         # ON event noise rate in events / pixel / s
bgnn = 0.01         # OFF event noise rate in events / pixel / s
ref = 100           # refractory period in us
dt = 1000           # time between frames in us
time = 0
leakeage_current = 1e2
fmin             = 4.7e-5
F_max            = 10e6 * fmin

# Main function to run the simulation
def run_simulation():
    # Read the configuration file
    ini_file = 'config/simulation_config_1.ini'
    InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams = read_ini_file(ini_file)

    
    # Initialize simulation data function
    Dynamics, InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams = initialize_simulation_params(InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams)
    frame_size = [SensorParams['height'], SensorParams['width']]
    t = Dynamics['t']             # Start time (from initialized dynamics)
    t_end = InitParams['t_end']   # End time (from the INI file)    

    initial_frame, Dynamics, initial_target_frame = frame_sim_functions(Dynamics, InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams)

    # # Create the initial frame and setup simulation plots
    # plt.ion()    
    # fig, ax = plt.subplots()
    # imgg = ax.imshow(initial_frame, cmap='gray', animated=True)
    # title = ax.set_title(f'Time: {1000*t:.2f} ms')
    # plt.colorbar(imgg, ax=ax)#, label='Intensity')
    
    im = initial_frame
    # Initialize event simulator model   -    <<<<>>>> this needs updating according to our discussion - see suggestion below <<<<>>>>
    dvs = DvsSensor("MySensor")
    dvs.initCamera(frame_size[1], frame_size[0],
                    lat=lat, jit = jit, ref = SensorBiases['refr'],
                    tau = tau, th_pos = SensorBiases['diff_on'], th_neg = SensorBiases['diff_off'],
                    th_noise = th_noise, bgnp=bgnp, bgnn=bgnn,
                    lcurr=leakeage_current,fmax=F_max)

    # suggested input to init dvs cam:
    # dvs = DvsSensor("MySensor")
    # dvs.initCamera(frame_size[1], frame_size[0],
    #                 pixel_pitch = SensorParams['pixel_pitch'], QE = SensorParams['QE'], ff = SensorParams['fill_factor'],
    #                 tau_sf = SensorParams['tau_sf'], tau_dark = SensorParams['tau_dark'],
    #                 th_noise = SensorParams['threshold_noise'], jit = SensorParams['latency_jitter'], 
    #                 ref = SensorBiases['refr'], th_pos = SensorBiases['diff_on'], th_neg = SensorBiases['diff_off'],
    #                 bgnp=bgnp, bgnn=bgnn) # <- these last two might go away one day... but we will continue to use them for now
                    
    # To use the measured noise distributions, uncomment the following line
    dvs.init_bgn_hist("EVENT_SIMULATOR/data/noise_pos_161lux.npy", "EVENT_SIMULATOR/data/noise_neg_161lux.npy")
    # Set as the initial condition of the sensor
    dvs.init_image(im)
    # Create the event buffer
    ev_full = EventBuffer(1)
    # Create the arbiter - optional, pick from one below
    ea = SynchronousArbiter(0.1, t, im.shape[0])  # DVS346-like arbiter
    # Create the display
    render_timesurface = 1
    ed = EventDisplay("Events",
                    frame_size[1],
                    frame_size[0],
                    InitParams['dt'],
                    render_timesurface)

    # Run the simulation loop until t exceeds t_end
    while t < t_end:
        
        # Create the camera pixel frame and target mask
        pixel_frame, Dynamics, target_frame_norm = frame_sim_functions(Dynamics, InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams)
        t = Dynamics['t']
        
        # Run event simulator using the current image frame
        ev = dvs.update(im, dt)
        
        # Classify events to "target" and "background" according to 2 target masks
        # Classify the events = 2 frames  +event in between (target mask with PSF)
        
        
        # frame_timestamp.append((ev.ts[0],ev.ts[-2]))
        ed.update(ev, dt)
        # Add the events to the buffer for the full video
        ev_full.increase_ev(ev)
        
        
        
        
        # Update initial frames to fit the new frames
        initial_frame = pixel_frame
        initial_target_frame = target_frame_norm
        
        # # Plot new simulation frame (and classified events as well?)
        # imgg.set_array(pixel_frame)
        # title.set_text(f'Time: {1000*t:.2f} ms')
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # plt.pause(0.001)
        
        
    # plt.ioff()
    # plt.show()

    # save event stream to file (put in folder together with an initial frame and param .ini file)


if __name__ == '__main__':
    run_simulation()
