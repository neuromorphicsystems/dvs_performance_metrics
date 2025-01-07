# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:02:16 2024

@author: 30067913

This set of functions creates simulated intensity images of targets and
backgrounds, with basic tracking heuristics. The aim is to create simple "video"
frames for simulating event responce from an event camera.
Simulation parameters are defined in a dedicated .INI file, read by the 
read_ini_file function. TBD - genorate missing params as well
A initialization function genorates parameters required for frames using the
initialize_simulation_params function.
Each frame is genorated by the frame_sim_functions function, where both the
intensity frame (in photon flux units), and a binary "target" mask are output 
of this function.
"""

import numpy as np
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
import configparser
from astropy.convolution import AiryDisk2DKernel
import matplotlib.pyplot as plt
from PIL import Image
import random 

# random.seed(10)

def initialize_simulation_params(InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams):
    # Initialize optical parameters
    OpticParams['IFOV'] = SensorParams['pixel_pitch'] / OpticParams['focal_length'] # imager IFOV [rad]
    OpticParams['optical_gain'] = 4*OpticParams['Fnum']**2 # ratio between energy on optical aparture to energy on sensor
    if InitParams['lux_flag']:
        OpticParams['conversion_to_photon_flux'] = 1.12e16 # [photons/lumen] - used to convert from [lumen/m^2] on sensor plane to [photon/m^2/sec] per pixel (550e-9/(h*c)/251)
    else:
        h = 6.626068e-34
        c = 299792458
        OpticParams['conversion_to_photon_flux'] = 1e-6*InitParams['wavelength']/(h*c)
        
    OpticParams['FOV_x'] = (SensorParams['width'] * OpticParams['IFOV']) * 180 / np.pi # horizontal field of view [deg]
    OpticParams['FOV_h'] = (SensorParams['height'] * OpticParams['IFOV']) * 180 / np.pi # vertical field of view [deg]
    OpticParams['dX'] = OpticParams['IFOV'] * SceneParams['t_distance'] # pixel/m
    OpticParams['PSF'] = create_airy_disk(OpticParams['PSF_size'], None, InitParams['multiplier']) # Create Airy disk

    # Expand simulation to accommodate convolution edges
    InitParams['full_width'] = int(SensorParams['width'] + OpticParams['PSF_size'] + 2 * SceneParams['Jitter_amp'])
    InitParams['full_height'] = int(SensorParams['height'] + OpticParams['PSF_size'] + 2 * SceneParams['Jitter_amp'])
    InitParams['ind_to_trim'] = np.array([
        [int(np.floor((InitParams['full_height'] - SensorParams['height'])/2)),
         int(np.floor((InitParams['full_height'] + SensorParams['height'])/2))],
        [int(np.floor((InitParams['full_width'] - SensorParams['width'])/2)),
         int(np.floor((InitParams['full_width'] + SensorParams['width'])/2))]
    ])

    # Initialize target parameters
    TargetParams['kernel_radius'] = OpticParams['focal_length'] * TargetParams['target_radius'] / (
            SceneParams['t_distance'] * SensorParams['pixel_pitch'])

    # Create the target kernel based on target type
    if TargetParams['target_type'] == 'spot':
        ker_size = 2 * TargetParams['kernel_radius'] * InitParams['multiplier'] + 1
        TargetParams['kernel'] = np.zeros((int(np.floor(ker_size)), int(np.floor(ker_size))), dtype=np.uint8)
        center = ker_size / 2

        for i in range(int(np.floor(ker_size))):
            for j in range(int(np.floor(ker_size))):
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if distance <= (TargetParams['kernel_radius'] * InitParams['multiplier']):
                    TargetParams['kernel'][i, j] = 1

    elif TargetParams['target_type'] == 'g_flash':
        TargetParams['max_diam'] = TargetParams['kernel_radius'] * 2

    elif TargetParams['target_type'] == 'blinking_spot':
        ker_size = 2 * TargetParams['kernel_radius'] * InitParams['multiplier'] + 1
        TargetParams['kernel'] = np.zeros((int(np.floor(ker_size)), int(np.floor(ker_size))), dtype=np.uint8)
        center = ker_size / 2

        for i in range(int(np.floor(ker_size))):
            for j in range(int(np.floor(ker_size))):
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if distance <= (TargetParams['kernel_radius'] * InitParams['multiplier']):
                    TargetParams['kernel'][i, j] = 1
        TargetParams['init_intensity'] = TargetParams['target_brightness']

    # Initialize background parameters
    BgParams['bg_spatial_freq'] = BgParams['S_freq'] * SceneParams['bg_distance'] * SensorParams['pixel_pitch'] / OpticParams['focal_length']
    X, Y = np.meshgrid(
        np.arange(-InitParams['full_width'] / 2, InitParams['full_width'] / 2, (1 / InitParams['multiplier'])),
        np.arange(-InitParams['full_height'] / 2, InitParams['full_height'] / 2, (1 / InitParams['multiplier']))
    )
    BgParams['X'], BgParams['Y'] = X, Y

    if BgParams['BG_type'] == 'natural':
        df = 0.1
        fn = np.arange(df, 10 + df, df) # Various spatial frequencies
        BgParams['Norm_f'] = np.sum(fn ** -2) ** -1
        # BgParams['phase_rx'] = [np.newaxis,np.newaxis,np.random.uniform(0, 2 * np.pi, len(fn))]
        # BgParams['phase_ry'] = [np.newaxis,np.newaxis,np.random.uniform(0, 2 * np.pi, len(fn))]
        # BgParams['amp_rx'] = [np.newaxis,np.newaxis,np.random.uniform(0, 1, len(fn))]
        # BgParams['amp_ry'] = [np.newaxis,np.newaxis,np.random.uniform(0, 1, len(fn))]
        # BgParams['fn_fac'] = [np.newaxis,np.newaxis,fn]
        BgParams['phase_rx'] = np.random.uniform(0, 2 * np.pi, len(fn))
        BgParams['phase_ry'] = np.random.uniform(0, 2 * np.pi, len(fn))
        BgParams['amp_rx'] = np.random.uniform(0, 1, len(fn))
        BgParams['amp_ry'] = np.random.uniform(0, 1, len(fn))
        BgParams['fn_fac'] = fn

    # Initialize line-of-sight jitter if active
    if SceneParams['Jitter_amp'] and SceneParams['Jitter_speed']:
        SceneParams['fn'] = np.arange(0.5, 1000 + 0.5, 0.5)
        SceneParams['Norm_f'] = np.sum(SceneParams['fn'] ** -2) ** -1
        SceneParams['phase_dx'] = np.random.uniform(0, 2 * np.pi, len(SceneParams['fn']))
        SceneParams['phase_dy'] = np.random.uniform(0, 2 * np.pi, len(SceneParams['fn']))
        SceneParams['phase_cx'] = np.random.uniform(0, 1, len(SceneParams['fn']))
        SceneParams['phase_cy'] = np.random.uniform(0, 1, len(SceneParams['fn']))

    # Initialize dynamic parameters
    Dynamics = {
        't': -InitParams['dt'],  # Initial time [sec]
        'i_azimuth': SceneParams['i_azimuth'],  # Initial azimuth angle of imager
        'i_elevation': SceneParams['i_elevation'],  # Initial elevation angle of imager
        't_azimuth': SceneParams['t_azimuth'],  # Initial azimuth angle of target
        't_elevation': SceneParams['t_elevation'],  # Initial elevation angle of target
        'imaging_los_speed': SceneParams['imaging_los_speed']  # Initial imaging line-of-sight angular velocity
    }

    return Dynamics, InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams

def create_airy_disk(PSF_size, MTF, multiplier):
    # Create Airy disk of the optical blurring

    # x = np.linspace(-2 * PSF_size, 2 * PSF_size, int(np.floor(4 * PSF_size * multiplier) + 1))
    # X, Y = np.meshgrid(x * 3.832 / PSF_size, x * 3.832 / PSF_size)
    # r = np.sqrt(X ** 2 + Y ** 2)
    # airydisk = np.where(r == 0, 1, (2 * np.abs(np.sinc(r))) ** 2)  # Use numpy's sinc function for Airy disk
    # airydisk /= np.sum(airydisk)  # Normalize
    # return airydisk


    airydisk_2D_kernel = AiryDisk2DKernel(PSF_size*multiplier) #convolutional kernel (PSF) 
    return airydisk_2D_kernel


def frame_sim_functions(Dynamics, InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams):
    
    # Extract parameters for readability
    width = InitParams['full_width']
    height = InitParams['full_height']
    dt = InitParams['dt']
    obstruct = InitParams['obstruct']
    multiplier = InitParams['multiplier']
    #lux_flag = InitParams['lux_flag']
    #wavelength = InitParams['wavelength']
    
    t_distance = SceneParams['t_distance']
    t_velocity = SceneParams['t_velocity']
    
    #PSF_size = OpticParams['PSF_size']
    IFOV = OpticParams['IFOV']
    
    # conversion from irradiance to photon flux units
    BG_const = SceneParams['BG_const'] * OpticParams['optical_gain'] * OpticParams['conversion_to_photon_flux']
    BG_brightness = BgParams['BG_brightness'] * OpticParams['optical_gain'] * OpticParams['conversion_to_photon_flux']
    target_brightness = TargetParams['target_brightness'] * OpticParams['optical_gain'] * OpticParams['conversion_to_photon_flux']
    
    # Base frame (constant background illumination)
    BG_frame = np.ones((int(height * multiplier), int(width * multiplier))) * BG_const
   
    # Correct LOS for tracking
    target_angular_velocity = t_velocity / t_distance
    Dynamics['t_azimuth'] += dt * target_angular_velocity
    Dynamics['i_azimuth'] += dt * Dynamics['imaging_los_speed']
    # add elevation updating?
    
    
    angular_diff = Dynamics['t_azimuth'] - Dynamics['i_azimuth']
    # tracking according to method
    if SceneParams['tracking_mode'] == 'chase': #(1) chase the target
        if target_angular_velocity > (Dynamics['imaging_los_speed']+IFOV/dt):
            Dynamics['imaging_los_speed'] += dt * SceneParams['imaging_los_acc']
        elif target_angular_velocity < (Dynamics['imaging_los_speed']-IFOV/dt):
            Dynamics['imaging_los_speed'] -= dt * SceneParams['imaging_los_acc']
        else:
            Dynamics['imaging_los_speed'] = target_angular_velocity
            
    elif SceneParams['tracking_mode'] == 'perfect': #(2) always aligned with taregt
        Dynamics['imaging_los_speed'] = 0
        Dynamics['i_azimuth'] = Dynamics['t_azimuth']
        Dynamics['t_elevation'] = Dynamics['i_elevation']
        
    elif SceneParams['tracking_mode'] == 'leaps': #(3) periodically jump toward the target
        if np.mod(Dynamics['t']/SceneParams['leapTime'],1)>(1-SceneParams['leapDuty']):
            if abs(angular_diff) <IFOV:
                Dynamics['imaging_los_speed'] = target_angular_velocity
            else:
                ratio = min(1,abs(angular_diff)/dt)
                Dynamics['imaging_los_speed'] += np.sign(angular_diff)*dt*SceneParams['imaging_los_acc']*ratio
        elif np.mod(Dynamics['t']/SceneParams['leapTime'],1)<SceneParams['leapDuty']: # stop tracking
            if Dynamics['imaging_los_speed']*dt<2*IFOV:
                Dynamics['imaging_los_speed'] = 0
            else: # debug this
                Dynamics['imaging_los_speed'] = np.sign(Dynamics['imaging_los_speed']) * abs(Dynamics['imaging_los_speed'] - dt*SceneParams['imaging_los_acc'])
        else:
            Dynamics['imaging_los_speed'] = 0
 
    # Background pixel shift
    bg_pixel_shift_x = Dynamics['i_azimuth'] / OpticParams['IFOV']
    bg_pixel_shift_y = Dynamics['i_elevation'] / OpticParams['IFOV']
    # Add background feature
    BG_frame += BG_brightness * make_BG_frame(width, height, multiplier, BgParams, [bg_pixel_shift_x, bg_pixel_shift_y])
    
    # Calculate target location
    target_pix_loc = [
        width / 2 + (Dynamics['t_azimuth'] - Dynamics['i_azimuth']) / IFOV,
        height / 2 + (Dynamics['t_elevation'] - Dynamics['i_elevation']) / IFOV
    ]
    
    # Create target frame if within FOV
    if 0 < target_pix_loc[0] < width and 0 < target_pix_loc[1] < height:
        target_frame = make_target_frame(Dynamics['t'], target_pix_loc, width, height, multiplier, TargetParams)
    else:
        target_frame = np.zeros_like(BG_frame)
    
    # Add BG and target to a single layer
    offset = 1e-6
    target_frame_norm = target_frame / np.max(target_frame+offset)
    if np.sum(target_frame) and obstruct:
        target_frame_norm_2 = target_frame_norm ** 2
        out_frame = BG_frame * (1 - target_frame_norm_2) + target_brightness * target_frame * target_frame_norm_2
    else:
        out_frame = BG_frame + target_brightness * target_frame
    
    # Add optical blurring and LOS jitter
    if SceneParams['Jitter_amp'] and SceneParams['Jitter_speed']:
        PSF = make_PSF_with_motion(SceneParams, OpticParams['PSF'], multiplier, Dynamics['t'], dt)
    else:
        PSF = OpticParams['PSF']
    
    out_frame = fftconvolve(out_frame, PSF, mode='same')
    
    # Downsample to fit sensor size
    if multiplier != 1:
        pixel_frame_expanded = zoom(out_frame, (height / out_frame.shape[0], width / out_frame.shape[1]), order=1)
    else:
        pixel_frame_expanded = out_frame
    
    pixel_frame = pixel_frame_expanded[
        InitParams['ind_to_trim'][0, 0]:InitParams['ind_to_trim'][0, 1],
        InitParams['ind_to_trim'][1, 0]:InitParams['ind_to_trim'][1, 1]
    ]
    
    # Update dynamics (mock update)
    Dynamics['t'] += InitParams['dt']
    
    target_frame_norm = np.array(target_frame_norm)
    target_height = SensorParams['height']
    target_width = SensorParams['width']
    normalized_target_frame = (target_frame_norm - target_frame_norm.min()) / (target_frame_norm.max() - target_frame_norm.min()) * 255
    normalized_target_frame = np.nan_to_num(normalized_target_frame, nan=0.0, posinf=255, neginf=0)
    target_frame_image = Image.fromarray(normalized_target_frame.astype(np.uint8))
    resized_target_frame_image = target_frame_image.resize((target_width, target_height), Image.NEAREST)
    resized_target_frame_norm = np.array(resized_target_frame_image) / 255.0 * (target_frame_norm.max() - target_frame_norm.min()) + target_frame_norm.min()

    return pixel_frame, Dynamics, resized_target_frame_norm


def make_BG_frame(width, height, multiplier, BgParams, pixel_shift):
    if BgParams['BG_type'] == 'const':
        return np.ones((int(multiplier * height), int(multiplier * width)))
    elif BgParams['BG_type'] == 'lines':
        x = BgParams['X'] + pixel_shift[0]
        y = BgParams['Y'] + pixel_shift[1]
        return (np.cos(2 * np.pi * BgParams['bg_spatial_freq'] * (np.cos(np.radians(BgParams['S_dir'])) * x + np.sin(np.radians(BgParams['S_dir'])) * y) / 2) + 1) / 2
    elif BgParams['BG_type'] == 'natural':
        # X_struct = 1j * 2 * np.pi * np.einsum('k,ij->ijk' ,BgParams['bg_spatial_freq'] * BgParams['fn_fac'] * BgParams['amp_rx'] , (BgParams['X'] + pixel_shift[0]))
        # Y_struct = 1j * 2 * np.pi * np.einsum('k,ij->ijk' ,BgParams['fn_fac'] * BgParams['amp_ry'] * BgParams['bg_spatial_freq'] , (BgParams['Y'] + pixel_shift[1]))        
        # X_struct = 1j * 2 * np.pi * (BgParams['bg_spatial_freq'] * BgParams['fn_fac'] * BgParams['amp_rx'] * ([BgParams['X'] + pixel_shift[0],np.newaxis]))
        # Y_struct = 1j * 2 * np.pi * (BgParams['fn_fac'] * BgParams['amp_ry'] * BgParams['bg_spatial_freq'] * ([BgParams['Y'] + pixel_shift[1],np.newaxis]))
        # return 2 * BgParams['Norm_f'] * np.sum((np.real(np.exp(X_struct + Y_struct)) + 1) / 2, axis=2)
        frame_out = np.zeros([int(multiplier * height),int(multiplier * width)],float)
        for k in range(len(BgParams['fn_fac'])):
            X_comp = BgParams['bg_spatial_freq'] * BgParams['fn_fac'][k] * BgParams['amp_rx'][k] * (BgParams['X'] + pixel_shift[0])
            Y_comp = BgParams['bg_spatial_freq'] * BgParams['fn_fac'][k] * BgParams['amp_ry'][k] * (BgParams['Y'] + pixel_shift[1])
            total_phase = 2*np.pi*(X_comp + Y_comp)
            frame_out += BgParams['Norm_f'] * (np.cos(total_phase) + 1)
            
        return frame_out 
    

def make_target_frame(t, target_loc, width, height, multiplier, TargetParams):
    frame_ext = np.zeros((int(multiplier * height), int(multiplier * width)))
    x = int(target_loc[0] * multiplier)
    y = int(target_loc[1] * multiplier)
    
    if TargetParams['target_type'] == 'spot':
        frame_ext[y, x] = 1
        return fftconvolve(frame_ext, TargetParams['kernel'], mode='same')
    elif TargetParams['target_type'] == 'g_flash':
        ti = TargetParams['t_init']
        tc = TargetParams['t_constant']
        if ti < t < (ti + 2.7 * tc):
            frame_ext[y, x] = 1
            cur_br = 88 * ((t - ti) / tc) ** 3 / (np.exp(5 * (t - ti) / tc) - 1)
            cur_diam = TargetParams['max_diam'] * np.exp(-((t - ti - 0.5) / tc) ** 2)
            x_vals = np.arange(-np.floor(1.5 * cur_diam), np.floor(1.5 * cur_diam) + 1)
            gaussian_1d = np.exp(-(x_vals / cur_diam) ** 2)
            cur_kernel = np.outer(gaussian_1d, gaussian_1d)
            cur_kernel *= cur_br / np.max(cur_kernel)
            return fftconvolve(frame_ext, cur_kernel, mode='same')
        else:
            return frame_ext
    # Add cases for 'modulated_spot' and 'blinking_spot'
    return frame_ext


def make_PSF_with_motion(SceneParams, PSF, multiplier, t, dt):
    x = round(multiplier * SceneParams['Jitter_amp'] * SceneParams['Norm_f'] * np.sum(SceneParams['fn'] ** -2 * np.cos(SceneParams['Jitter_speed'] * 2 * np.pi * t + SceneParams['phase_dx'])))
    y = round(multiplier * SceneParams['Jitter_amp'] * SceneParams['Norm_f'] * np.sum(SceneParams['fn'] ** -2 * np.cos(SceneParams['Jitter_speed'] * 2 * np.pi * t + SceneParams['phase_dy'])))
    SceneParams['phase_dx'] += SceneParams['phase_cx'] * dt
    SceneParams['phase_dy'] += SceneParams['phase_cy'] * dt
    
    if x > 0:
        PSF_temp = np.pad(PSF, ((0, 0), (x, 0)), mode='constant')
    elif x < 0:
        PSF_temp = np.pad(PSF, ((0, 0), (0, -x)), mode='constant')
    else:
        PSF_temp = PSF
    
    if y > 0:
        return np.pad(PSF_temp, ((y, 0), (0, 0)), mode='constant')
    elif y < 0:
        return np.pad(PSF_temp, ((0, -y), (0, 0)), mode='constant')
    
    return PSF_temp

# Function to strip comments and convert values to the correct type
def get_clean_value(value, dtype):
    # Strip comments after ';' and remove leading/trailing spaces
    clean_value = value.split(';')[0].strip()

    # Convert to the appropriate type
    if dtype == int:
        out_val = list(map(int,clean_value.split(',')))
        if len(out_val)>1:
            return out_val
        else:
            return out_val[0]
    elif dtype == float:
        out_val = list(map(float,clean_value.split(',')))
        if len(out_val)>1:
            return out_val
        else:
            return out_val[0]
    elif dtype == bool:
        return clean_value.lower() in ['true', '1', 'yes']
    elif dtype == np.ndarray:
        return np.array(eval(clean_value))  # Use eval to parse arrays
    else:
        return clean_value  # Keep as string for general cases

# Function to read the INI file and initialize parameters
def read_ini_file(ini_file):
    config = configparser.ConfigParser()
    config.read(ini_file)

    # Initialize dictionaries for each parameter group
    InitParams = {
        'sim_name': get_clean_value(config['InitParams']['sim_name'], str),
        't_end': get_clean_value(config['InitParams']['t_end'], float),
        'dt': get_clean_value(config['InitParams']['dt'], float),
        'obstruct': get_clean_value(config['InitParams']['obstruct'], bool),
        'multiplier': get_clean_value(config['InitParams']['multiplier'], int),
        'lux_flag': get_clean_value(config['InitParams']['lux_flag'], bool),
        'wavelength': get_clean_value(config['InitParams']['wavelength'], float),
        'sensor_model': get_clean_value(config['InitParams']['sensor_model'], str),
    }

    SceneParams = {
        'BG_const': get_clean_value(config['SceneParams']['BG_const'], float),
        't_distance': get_clean_value(config['SceneParams']['t_distance'], float),
        'bg_distance': get_clean_value(config['SceneParams']['bg_distance'], float),
        't_velocity': get_clean_value(config['SceneParams']['t_velocity'], float),
        't_elevation': get_clean_value(config['SceneParams']['t_elevation'], float),
        't_azimuth': get_clean_value(config['SceneParams']['t_azimuth'], float),
        'i_elevation': get_clean_value(config['SceneParams']['i_elevation'], float),
        'i_azimuth': get_clean_value(config['SceneParams']['i_azimuth'], float),
        'imaging_los_speed': get_clean_value(config['SceneParams']['imaging_los_speed'], float),
        'imaging_los_acc': get_clean_value(config['SceneParams']['imaging_los_acc'], float),
        'Jitter_amp': get_clean_value(config['SceneParams']['Jitter_amp'], float),
        'Jitter_speed': get_clean_value(config['SceneParams']['Jitter_speed'], float),
        'tracking_mode': get_clean_value(config['SceneParams']['tracking_mode'], str),
        'leapTime': get_clean_value(config['SceneParams']['leapTime'], float),
        'leapDuty': get_clean_value(config['SceneParams']['leapDuty'], float)       
    }

    OpticParams = {
        'focal_length': get_clean_value(config['OpticParams']['focal_length'], float),
        'Fnum': get_clean_value(config['OpticParams']['Fnum'], float),
        'PSF_size': get_clean_value(config['OpticParams']['PSF_size'], float),
    }

    TargetParams = {
        'target_type': get_clean_value(config['TargetParams']['target_type'], str),
        'target_radius': get_clean_value(config['TargetParams']['target_radius'], float),
        'target_brightness': get_clean_value(config['TargetParams']['target_brightness'], float),
        'target_brightness_min': get_clean_value(config['TargetParams']['target_brightness_min'], float),
        'mod_freq': get_clean_value(config['TargetParams']['mod_freq'], float),
        'mod_duty_cycle': get_clean_value(config['TargetParams']['mod_duty_cycle'], float),
        't_init': get_clean_value(config['TargetParams']['t_init'], float),
        't_constant': get_clean_value(config['TargetParams']['t_constant'], float),
    }

    BgParams = {
        'BG_type': get_clean_value(config['BgParams']['BG_type'], str),
        'BG_brightness': get_clean_value(config['BgParams']['BG_brightness'], float),
        'S_freq': get_clean_value(config['BgParams']['S_freq'], float),
        'S_dir': get_clean_value(config['BgParams']['S_dir'], float),
    }
    
    SensorBiases = {
        'diff_on': get_clean_value(config['SensorBiases']['diff_on'], float),
        'diff_off': get_clean_value(config['SensorBiases']['diff_off'], float),
        'refr': get_clean_value(config['SensorBiases']['refr'], float),
    }

    
    if InitParams["sensor_model"]=="Gen4":
        config2 = configparser.ConfigParser()
        config2.read("config/Gen4_config.ini")
        SensorParams = {
            'width': get_clean_value(config2['SensorParams']['width'], int),
            'height': get_clean_value(config2['SensorParams']['height'], int),
            'pixel_pitch': get_clean_value(config2['SensorParams']['pixel_pitch'], float),
            'fill_factor': get_clean_value(config2['SensorParams']['fill_factor'], float),
            'tau_sf': get_clean_value(config2['SensorParams']['tau_sf'], float),
            'tau_dark': get_clean_value(config2['SensorParams']['tau_dark'], float),
            'QE': get_clean_value(config2['SensorParams']['QE'], float),
            'threshold_noise': get_clean_value(config2['SensorParams']['threshold_noise'], float),
            'latency_jitter': get_clean_value(config2['SensorParams']['latency_jitter'], float),
            'latency': get_clean_value(config2['SensorParams']['latency'], float),
            'I_dark': get_clean_value(config2['SensorParams']['I_dark'], float),
            }
    else:
        SensorParams = {
            'width': get_clean_value(config['ManualSensorParams']['width'], int),
            'height': get_clean_value(config['ManualSensorParams']['height'], int),
            'pixel_pitch': get_clean_value(config['ManualSensorParams']['pixel_pitch'], float),
            'fill_factor': get_clean_value(config['ManualSensorParams']['fill_factor'], float),
            'tau_sf': get_clean_value(config['ManualSensorParams']['tau_sf'], float),
            'tau_dark': get_clean_value(config['ManualSensorParams']['tau_dark'], float),
            'QE': get_clean_value(config['ManualSensorParams']['QE'], float),
            'threshold_noise': get_clean_value(config['ManualSensorParams']['threshold_noise'], float),
            'latency_jitter': get_clean_value(config['ManualSensorParams']['latency_jitter'], float),
            'latency': get_clean_value(config['ManualSensorParams']['latency'], float),
            'I_dark': get_clean_value(config['ManualSensorParams']['I_dark'], float),
            }

    scanned_params = {}
    for section_name, section_dict in [('InitParams', InitParams), ('SceneParams', SceneParams), ('OpticParams', OpticParams), ('TargetParams', TargetParams), ('BgParams', BgParams), ('SensorBiases', SensorBiases), ('SensorParams', SensorParams)]:
        for param_name, value in section_dict.items():
            if isinstance(value, list) and len(value)>1:
                # scanned_params[f'{section_name}["'f'{param_name}''"]'] = value  # Track scanned parameters
                scanned_params[f'{section_name}.{param_name}'] = value  # Track scanned parameters

    
    return InitParams, SceneParams, OpticParams, TargetParams, BgParams, SensorBiases, SensorParams, scanned_params