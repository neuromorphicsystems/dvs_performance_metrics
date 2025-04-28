# Performance Metrics for Neuromorphic Imaging
## description
This is a framework to simulate basic object motion scenes, as viewed by an event camera, and calculate the expected performance of the sensor for various imaging system parameters.
The code is divided into several sections:
1)	Broad description of imaging parameters (sensor, optics, and scene) via manual editing of configuration files.
2)	Generate synthetic frames according to these parameters.
3)	Simulating Even stream data according to these synthetic frames, with classification of target events and background events.
4)	Running the performance analysis for the event-stream data sets.

Step 1 – create a single or multiple .ini config files, placed in the “\config” folder, to fit the imaging system and scenario under examination.
Step 2 –use “run_simulation.py” with single config file input argument, or “run_several_sims.py” for calling several config files consecutively (with specific naming conventions). This will both generate the synthetic frames and the event streams.
Step 3 – when all data files are created, use analysis scripts to examine the data and calculate change in various metrics. Examples include MATLAB scripts such as “FullTestAnalysis.m”, and these need to be adapted to the parameter of interest of each simulation run.  

Related publication: Performance metrics for neuromorphic imaging, N.Kruger et al, 2025
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

To pull content from another branch:

`git checkout origin/<brance-name> -- <folder-name/file-name>`


## tmux: create session
tmux new-session -d -s performance
## tmux: access a session
tmux attach-session -t performance
## tmux: exit a session
Ctrl+B -> d

## tmux: new terminal
Ctrl+B -> c
## tmux: see active terminals
Ctrl+B -> w
## tmux: name terminal
Crtl+B -> ,

# to do in each terminal:
Ctrl+B -> c
Crtl+B -> , _name_
conda activate dvs_performance_metric && 
python run_simulation.py -filename "frequency_amplitude_heatmap_amp_2"

## tmux: show status
htop
# exit:
Ctrl+c