# Performance Metrics for Neuromorphic Imaging - simulation and analysis code

# description
This is a framework to simulate basic object motion scenes, as viewed by an event camera, and calculate the expected performance of the sensor for various imaging system parameters.
The code is divided into several sections:
1)	Broad description of imaging parameters (sensor, optics, and scene) via manual editing of configuration files.
2)	Generate synthetic frames according to these parameters.
3)	Simulating Even stream data according to these synthetic frames, with classification of target events and background events.
4)	Running the performance analysis for the event-stream data sets.

**Step 1** – create a single or multiple .ini config files, placed in the “\config” folder, to fit the imaging system and scenario under examination. 

**Step 2** – use “run_simulation.py” with single config file input argument, or “run_several_sims.py” for calling several config files consecutively (with specific naming conventions). This will both generate the synthetic frames and the event streams.

Example single config file simulation: 
```sh
python3 run_simulation.py -filename "Test_debug"
```

**Step 3** – when all data files are created, use analysis scripts to examine the data and calculate change in various metrics. Examples include MATLAB scripts such as “FullTestAnalysis.m”, and these need to be adapted to the parameter of interest of each simulation run.  

Related publication: **Performance metrics for neuromorphic imaging**, N.Kruger, S.Arja, E.Andreq, T.Monk, A.van Schaik [2025]
https://ebooks.spiedigitallibrary.org/conference-proceedings-of-spie/13376/133760D/Performance-metrics-for-neuromorphic-imaging/10.1117/12.3041873.full


# Setup

## Requirements
- python: 3.9.x, 3.10.x

## Tested environments
- Ubuntu 22.04
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

# Content of each folder

**OPTICAL_SIMULATOR**: functions used to generate frames from parameter files. main function if the "basic_tar_bg_simulation.py" - containing config file read functions, and frame creation for target and background.

**EVENT_SIMULATOR**: Contains the event simulation. This version is an unpdated version of the simulation published in:
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.702765/full
Changes include adding physical values for conversion from pixel illumination flux to voltage readout, and the ability to classify events according to input.

**PERFORMANCE_METRICS**: functions used to support the MATLAB performance analysis scripts (such as event motion allignment)

**OUTPUT**: simulation data output files

**dvs_warping_package and dvs_warping_package_extension**: Python and C++ packages to enable denoising and event warping

**config**: the .ini config files for all optical system, scene, and sensor parameters used to simulated the frames and the corresponding event streams. See details in the next section here.

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
