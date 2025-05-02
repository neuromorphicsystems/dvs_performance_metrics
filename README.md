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
python3 run_simulation.py -filename "<config_name>" #no need to add the file format ".ini" for config filename

# Example command
python3 run_simulation.py -filename "frequency_size_heatmap_size50"
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

4) Brightness levels are those expected on the sensor plane. As this simulation doesn't include translating target illumination levels to these expected on the sensor plane, the analytical process of estimated precieved brightness by the sensor is left for the user to derive.

## 🚀 InitParams

General settings for the simulation.

| Parameter      | Type    | Unit         | Description                                                                                                                                   |
| -------------- | ------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `sim_name`     | string  | —            | Unique name for this run (also used to name output files).                                                                                    |
| `t_end`        | float   | seconds      | End time of the simulation.                                                                                                                   |
| `dt`           | float   | seconds      | Time increment for each simulation step.                                                                                                      |
| `obstruct`     | bool    | —            | **True**: target obscures the background; **False**: target and background additively combine. (Default: `True`.)                               |
| `multiplier`   | int     | —            | Oversampling factor for pixel grid. (Only `1` has been fully tested.)                                                                         |
| `lux_flag`     | bool    | —            | **True**: use lumen units; **False**: use W/m² at a given wavelength.                                                                         |
| `wavelength`   | float   | μm           | Mean illumination wavelength (used only when `lux_flag = False`).                                                                             |
| `sensor_model` | string  | —            | Which sensor to load: `Gen4`, `Gen3`, `Davis346` or `Manual`. Non-Manual options pull from built-in config files. `Manual` uses the **ManualSensorParams** below. |

---

## 🎬 SceneParams

Controls the geometry and motion of target & background.

| Parameter           | Type   | Unit            | Description                                                                                                                                       |
| ------------------- | ------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BG_const`          | float  | lumen / W/m²    | Constant background illumination level.                                                                                                           |
| `t_distance`        | float  | meters          | Nominal distance of the (flat) target from the imager.                                                                                            |
| `bg_distance`       | float  | meters          | Nominal distance of the (flat) background from the imager.                                                                                        |
| `t_velocity`        | float  | m/s             | Target speed (positive = rightward in LOS).                                                                                                       |
| `t_elevation`       | float  | radians         | Target elevation above the horizon.                                                                                                               |
| `t_azimuth`         | float  | radians         | Initial target azimuth (0 = North).                                                                                                               |
| `i_elevation`       | float  | radians         | Imager’s elevation angle—set equal to `t_elevation` to center the target in FOV.                                                                  |
| `i_azimuth`         | float  | radians         | Imager’s azimuth angle—set equal to `t_azimuth` to center the target in FOV at _t_ = 0.                                                            |
| `Jitter_amp`        | float  | pixels          | Amplitude of 1/f LOS jitter.                                                                                                                      |
| `Jitter_speed`      | float  | Hz              | Base frequency multiplier for LOS vibration.                                                                                                       |
| `imaging_los_speed` | float  | rad/s           | Initial angular velocity of LOS (overwritten in “perfect” tracking).                                                                               |
| `imaging_los_acc`   | float  | rad/s²          | LOS angular acceleration (used by “chase” & “leaps” heuristics).                                                                                    |
| `tracking_mode`     | string | —               | Heuristic: `chase`, `leaps` or `perfect`.                                                                                                         |
| `leapTime`          | float  | seconds         | Cycle time between “leaps” toward the target.                                                                                                      |
| `leapDuty`          | float  | —               | Duty cycle (fraction of `leapTime`) spent moving vs. stopping in “leaps” mode.                                                                     |

---

## 🔭 OpticParams

Defines the imaging optics.

| Parameter      | Type   | Unit    | Description                             |
| -------------- | ------ | ------- | --------------------------------------- |
| `focal_length` | float  | meters  | Lens focal length.                      |
| `Fnum`         | float  | —       | f-number (focal_length ÷ aperture dia). |
| `PSF_size`     | float  | pixels  | Gaussian PSF full-width at half max.    |

---

## 🎯 TargetParams

Shape, timing & brightness of the target.

| Parameter               | Type    | Unit           | Description                                                                                       |
| ----------------------- | ------- | -------------- | ------------------------------------------------------------------------------------------------- |
| `target_type`           | string  | —              | `spot` (constant) or `g_flash` (single-pulse).                                                    |
| `target_radius`         | float   | meters         | Radius of the target disk.                                                                        |
| `target_brightness`     | float   | lumen / W/m²   | Peak brightness on the sensor.                                                                    |
| `t_init`                | float   | seconds        | Flash onset time (for `g_flash`).                                                                 |
| `t_constant`            | float   | seconds        | Flash duration (approx. FWHM).                                                                    |
| `target_brightness_min` | float   | lumen / W/m²   | Min. brightness (`blinking_spot` & `modulated_spot`, TBD).                                         |
| `mod_freq`              | float   | Hz             | Modulation frequency (`blinking_spot` & `modulated_spot`, TBD).                                   |
| `mod_duty_cycle`        | float   | —              | Duty cycle of “on” time (`blinking_spot`, TBD).                                                   |

---

## 🌄 BgParams

Background pattern settings.

| Parameter       | Type    | Unit          | Description                                                                                       |
| --------------- | ------- | ------------- | ------------------------------------------------------------------------------------------------- |
| `BG_type`       | string  | —             | `const`, `lines` or `natural`.                                                                    |
| `S_freq`        | float   | lines/m       | Spatial frequency (or characteristic scale for `natural`).                                        |
| `S_dir`         | float   | degrees       | Orientation of lines (0 = left–right, 90 = up–down).                                              |
| `BG_brightness` | float   | lumen / W/m²  | Maximum brightness for `lines` or `natural`. (Set to 0 for constant BG.)                          |

---

## ⚙️ SensorBiases

User-tunable thresholds & refractory time.

| Parameter   | Type    | Unit  | Description           |
| ----------- | ------- | ----- | --------------------- |
| `diff_on`   | float   | —     | ON-event threshold.   |
| `diff_off`  | float   | —     | OFF-event threshold.  |
| `refr`      | float   | μs    | Refractory period.    |

---

## 🛠️ ManualSensorParams

Override built-in sensor specs when `sensor_model = Manual`.

| Parameter         | Type    | Unit     | Description                                                                                                                            |
| ----------------- | ------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `width`           | int     | pixels   | Horizontal resolution.                                                                                                                 |
| `height`          | int     | pixels   | Vertical resolution.                                                                                                                   |
| `pixel_pitch`     | float   | meters   | Pixel pitch.                                                                                                                           |
| `fill_factor`     | float   | —        | Active-area fraction.                                                                                                                  |
| `tau_sf`          | float   | seconds  | Source-follower time constant<sup>[1]</sup>.                                                                                           |
| `tau_dark`        | float   | seconds  | Dark-current time constant.                                                                                                            |
| `QE`              | float   | —        | Quantum efficiency (0–1).                                                                                                              |
| `threshold_noise` | float   | —        | Reset-noise standard deviation.                                                                                                        |
| `latency_jitter`  | float   | seconds  | Jitter of event latency.                                                                                                               |
| `latency`         | float   | seconds  | Fixed additional latency (e.g., arbiter delay).                                                                                        |
| `I_dark [1]`          | float   | amps     | Mean dark current (must be >0).                                                                                                        |

<sup>[1]</sup> See “Re-interpreting the step-response probability curve to extract fundamental physical parameters of event-based vision sensors” (**SPIE Proc. 10.1117/12.3022308**)  


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