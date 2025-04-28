# Performance Metrics for Neuromorphic Imaging


Simulation and analysis code for: **Performance metrics for neuromorphic imaging** paper  
*N. Kruger, S. Arja, E. Andrew, T. Monk & A. van Schaik (SPIE 13376, 2025)*  
[▶ Read the full paper on SPIE Digital Library](https://ebooks.spiedigitallibrary.org/conference-proceedings-of-spie/13376/133760D/Performance-metrics-for-neuromorphic-imaging/10.1117/12.3041873.full)


# Summary
This is a framework to simulate basic object motion scenes, as viewed by an event camera, and calculate the expected performance of the sensor for various imaging system parameters.

This repository provides:

1. **Config-driven scene setup** via simple `.ini` files (sensor, optics, scene).  
2. **Synthetic frame generation** based on those parameters.  
3. **Event‐stream simulation** with classification of target vs. background events.  
4. **Performance analysis** scripts to extract metrics from the event data.

---



# Setup

## Requirements
- python: 3.9.x, 3.10.x

## Tested environments
- Ubuntu 22.04 and Windows 11
- Conda 23.1.0
- Python 3.9.18

## Installation

```sh
git clone https://github.com/neuromorphicsystems/dvs_performance_metrics.git
cd dvs_performance_metrics
conda env create -f environment.yml
source ~/.bashrc && conda activate dvs_performance_metric
python3 -m pip install -e .
```

## Run Simulation

Follow these steps to configure and launch your simulations:

### 1. Prepare your configuration files  
- Create one or more `<name>.ini` files in the `config/` folder.  
- Each `.ini` contains your imaging parameters (sensor, optics, scene, timing). 
- See examples in `config/` 
 
### 2. Launch the simulator  

#### Single‐config-run  
```bash
python3 run_simulation.py -filename "<config_name>"
```

#### Optional flags

Edit the top of ```run_simulation.py```

```sh
DO_PLOTS    = 0    # 1 = enable real-time matplotlib display
SAVE_FRAMES = 0    # 1 = save per-frame PNGs into OUTPUT/<config>/
epoch       = 5    # number of repeats per config
bgnp, bgnn  = 0.2, 0.2  # background event rates
blankEvRun  = 0.5  # seconds of noise “warm-up”
```


<!-- 
#### Multiple‐config-run  
```bash
python3 run_several_sims.py -filename "<config_name_1>" "<config_name_2>" "<config_name_3>"
``` -->


<!-- **Step 2** – Use “run_simulation.py” with single config file input argument, or “run_several_sims.py” for calling several config files consecutively (with specific naming conventions). This will both generate the synthetic frames and the event streams. -->

### 3. Data Analyis and Post-processing

When all data files are created, use analysis scripts to examine the data and calculate change in various metrics. Examples include MATLAB scripts such as `FullTestAnalysis.m`, and these need to be adapted to the parameter of interest of each simulation run.  


## Content of each folder

- **`OPTICAL_SIMULATOR/`**  
  Core routines for generating synthetic image frames.  
  - **`basic_tar_bg_simulation.py`**:  
    - Parses your `.ini` config.  
    - Creates target and background images based on scene, optics, and sensor parameters.


- **`EVENT_SIMULATOR/`**  
  Event‐stream generation engine (forked and extended from Joubert *et al.* 2021):  
  <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.702765/full>  
  - Adds physical-to-voltage conversion (illumination → pixel voltage).  
  - Labels each event as **signal**, **background**, or **noise**.

- **`PERFORMANCE_METRICS/`**  
  Utility functions that support the MATLAB analysis scripts (e.g. motion alignment, metric calculators).

- **`dvs_warping_package/` & `dvs_warping_package_extension/`**  
  Python and C++ libraries for: 
  - Event denoising  
  - Motion compensation (event “warping”)
  - Only used when needed

- **`config/`**  
  Your simulation `.ini` files (scene, optics, sensor, noise, scan parameters).


**`OUTPUT/`**: 
Simulation data output files

```
OUTPUT/
└─ <config_name>/
   ├─ events_and_labels/
   │  ├─ simdata_<sim_name>_<ep>.mat    # MATLAB struct “simulation_data”
   │  └─ ev_<sim_name>_<ep>.txt         # [x, y, p, ts, l] per event
   ├─ raw_event_image/        (if SAVE_FRAMES=1)
   ├─ only_signal_image/      (if SAVE_FRAMES=1)
   ├─ only_background_image/  (if SAVE_FRAMES=1)
   ├─ only_noise_image/       (if SAVE_FRAMES=1)
   ├─ mask_overlay/           (if SAVE_FRAMES=1)
   └─ labeled_image/          (if SAVE_FRAMES=1)
```


# Config file creation and structure

**A few notes:**

1) There are two type of config files - one to be called by for simulations, and the second for generic sensor configurations. We will only detail the simulation parameters config file here, as the sensor config file is only a sub-set of the former.

2) Each configuration file can include ONE parameter to be "scanned" by the simulator, meaning it can be included as a vector, and not a scalar value. The difference being that the values for the chosen scanned parameter are a row of values, seperated be a comma "," only. This can only be done on fields that are a scalar or integer inputs (not for text input).

3) Responsibility on ensuring legal values is on the operator of the simulator. Ensure you choose values that fit real-world scenarios and hardware parameters.

4) brightness levels are those expected on the sensor plane. As this simulation doesn't include translating target illumination levels to these expected on the sensor plane, the analytical process of estimated precieved brightness by the sensor is left for the user to derive.


## [InitParams]
general parameters for the simulation.

**sim_name**: The simulation name - will be used also in nameing of result output files

**t_end**: Scalar value of simulation time end [sec]

**dt**: Scalar value of simulation time increment [sec]

**obstruct**: = True or False - True = target obscures the BG image. False = target and BG are aditive. Most cases this is set to True.

**multiplier**: Integer value - multiplier for oversampling pixel format - simulation run is properly tested for the value of 1,

**lux_flag**: True or False - flag for use of lumen brightness units (alternative: W/m^2 analysis for a specific wavelength)

**wavelength**: illumination average wavelngth [um] - used for enerdy conversion only when "lux_flag" is false.

**sensor_model**: Chosen sensor model. options = {'Gen4', 'Gen3', 'Davis346','Manual'} - all option other than "Manual" pull information from dedicated sensor config files. When "Manual" is chosen, the sensor parameters are taking for this file (under "ManualSensorParams" section at the end)


 ## [SceneParams]
 All parameters regarding the scene motion

**BG_const**: Scalar value of [lumen] or [W/m^2] - the background brightness level

**t_distance**: Scalar value of [m] - target nominal distance from imager. The target is imagined as a flat object at this distance, with a backdrop of a flat background. 

**bg_distance**: Scalar value of [m] - the background nominal distance from imager.

**t_velocity**: Scalar value of [m/s] - the target velocity. Psotive moving right in LOS

**t_elevation**: Scalar value of [rad] - the taget elevation above the horizon

**t_azimuth**: Scalar value of [rad] - the initial taget azimutal direction in relation to imager (0 being directly north)

**i_elevation**: Scalar value of [rad] - the imager viewing angle elvation. Make this equal to t_elevation to ensure target is in the vertical centre of FOV.

**i_azimuth**: Scalar value of [rad] - the imager viewing angle azimuth. Even here - make sure this is equal to t_azimuth to ensure target is in centre of FOV at t=0.

**Jitter_amp**: Scalar value of [pixels] - amplitude of 1/f jitter (vibration) of LOS in pixels

**Jitter_speed**: Scalar value of [Hz] - base frequncy as a factor of 1 Hz vibration

**imaging_los_speed**: Scalar value of [rad/sec] initial speed of line-of-sight (LOS). Make sure this matches the target motion speed if you want good tracking for initial values. (in case of "perfect" tracking, this value is overwritten)

**imaging_los_acc**: Scalar value of [rad/sec^2] - LOS acceleration when tracking (for chase or leaps tracking heuristics)

**tracking_mode**: Choose tracking heuristics. options = {'chase', 'leaps', 'perfect'}. 'chase' is the attempt of the imager to catch up with a target in motion. 'leaps' is a jump and stare model tracking (durations of jump and stare are defined below. 'perfect' is just assigning a constant angular velocity of the imager to match that of the target. This doesn't account for imager vibration, and these will induce apparent target motion in the FOV.

**leapTime**: Scalar value of [sec] - cycle time for leap (how often does the tracker jump towards target) - MAKE SURE acceleration is good... is too large is will overshoot, and if too small is won't catch up with the target.

**leapDuty**: Scalar value for duty cycle for motion in leap tracking mode (after which the same cycle duration is dedicated to stopping the imager motion)


## [OpticParams]
All optical parameters of the imaging system

**focal_length**: Scalar value of [m] - imager focal length

**Fnum**: Scalar value of optics F# (focal_length/diameter)

**PSF_size**: Scalar value of [pixels] - Gaussian Point Spread Function (PSF) Full-width-half-max size


## [TargetParams]
Define all target related parameters

**target_type**: choose from options: {'spot', 'g_flash'}. 'spot' being a simple constant brightnes round target. 'g_flash' is a round target coming into existance only once with a distinct temporal-spatial profile.
(future options include 'blinking_spot' and 'modulated_spot' - TBD)   

**target_radius**: Scalar value of [m] - Target radius

**target_brightness**: Scalar value of [lumen] or [W/m^2] - Target uniform (or maximal) brightness on sensor

**t_init**: Scalar value of [sec] - flash onset for 'g_flash'

**t_constant**: Scalar value of [sec] - flash time (~FWHM) for 'g_flash'

**target_brightness_min**: Placeholder (Scalar value of [lumen] or [W/m^2] minimal brightness for 'blinking_spot' and 'modulated_spot')

**mod_freq**: Placeholder (Scalar value of [Hz] modulation frequency for 'blinking_spot' & 'modulated_spot')

**mod_duty_cycle**: Placeholder (Scalar value of blink on duty cycle for 'blinking_spot')


## [BgParams]
Background parameters

**BG_type**: Choose options from: {'const', 'lines', 'natural'}. 'const' is constant illumination for entire FOV. 'lines' is a line structure, where minimal value is the "BG_const" value chosen under "SceneParams" section, and maximal value is defined below under "BG_brightness". 'natural' is a random natural distribution of intensity pattern (with max and min brightness defined similar to the lines option).

**S_freq**: Scalar value of [lines/m] - spatial frequency of background lines. For 'natural' background, this also indicates the general spatial frequancy of the random pattern.

**S_dir**: Scalar value of direction of 'lines' spatial frequency (degrees), 0 = left-right oriented, 90 = up-down oriented

**BG_brightness**: Scalar value of [lumen] or [W/m^2] - max background pattern brightness. For constant BG brightness choose this to be 0, as the brightness is defined by "BG_const" under "SceneParams" section.


## [SensorBiases]
Sensor parameters that are "user defined" - namely the diff-on and diff-off thresholds, and the refractory period.

**diff_on**: Scalar value of ON event threshold 

**diff_off**: Scalar value of OFF event threshold 

**refr**: Scalar value of refractory period in [us]


## [ManualSensorParams]
For override of fixed sensor parameters for certain sensor models, values can be chosen here. Only considered is "sensor_model" under "InitParams" section is chosen to be 'Manual'.

**width**: Integer number of horizontal pixels

**height**: Integer number of vertical pixels (this is usefull to keep small to reduce runtime, as most of the top and bottom sensor region is not participating in the horizontal motion of the scenes here)

**pixel_pitch**: Scalar value of pixel pitch [m]

**fill_factor**: Scalar value of pixel active area fill factor

**tau_sf**: Scalar value of maximal source follower pixel time constant [sec] (see "Re-interpreting the step-response probability curve to extract fundamental physical parameters of event-based vision sensors", 10.1117/12.3022308)

**tau_dark**: Scalar value of Dark current time constant [sec] (the relatively long time scale of the circuit when illumination levels are low)

**QE**: Scalar value of quantum efficiency (between 0 and 1)

**threshold_noise**: Scalar value of mean reset noise standard deviation of the transistor

**latency_jitter**: Scalar value of mean latency noise standard deviation [sec]

**latency**: Scalar value of latency in [sec] (this is in addition to the latency calculated by pixels response from first order filter calculation - can be attributed to arbiter latency)

**I_dark**: Scalar value of mena pixel dark current [A]. must be bigger than 0. (again, see "Re-interpreting the step-response probability curve to extract fundamental physical parameters of event-based vision sensors", 10.1117/12.3022308)




### Citations

If you use this work in an academic context, please cite the following:


```bibtex
@inproceedings{kruger2025performance,
  title        = {Performance metrics for neuromorphic imaging},
  author       = {Kruger, Nimrod and Arja, Sami and Andrew, Evie and Monk, Travis and van Schaik, André},
  booktitle    = {Quantum Sensing and Nano Electronics and Photonics XXI},
  volume       = {13376},
  pages        = {74--82},
  year         = {2025},
  organization = {SPIE}
}

@article{joubert2021event,
  title     = {Event camera simulator improvements via characterized parameters},
  author    = {Joubert, Damien and Marcireau, Alexandre and Ralph, Nic and Jolley, Andrew and Van Schaik, André and Cohen, Gregory},
  journal   = {Frontiers in Neuroscience},
  volume    = {15},
  pages     = {702765},
  year      = {2021},
  publisher = {Frontiers Media SA}
}
```