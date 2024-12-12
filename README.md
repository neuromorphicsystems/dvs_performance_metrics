# Performance Metrics for Neuromorphic Imaging


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


**OPTICAL_SIMULATOR**: 


**EVENT_SIMULATOR**: 


**PERFORMANCE_METRICS**: 


**OUTPUT**:


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
conda activate dvs_performance_metric && python run_simulation.py -filename "frequency_amplitude_heatmap_amp_24"

## tmux: show status
htop
# exit:
