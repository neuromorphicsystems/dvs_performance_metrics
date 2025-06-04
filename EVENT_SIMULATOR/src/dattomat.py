import loris
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt


event_data_file = "ev_100_10_100_40_0.7_0.01"
my_file = loris.read_file(f"./Code/gitlib/IEBCS_Sami/examples/00_video_2_events/outputs/{event_data_file}.dat")
events = my_file['events']
print(events)


td = {'x': events["x"], 'y': events["y"], 'p': events["p"], 'ts': events["ts"]}

# Save the dictionary into a MATLAB .mat file
savemat(f'./Code/gitlib/IEBCS_Sami/examples/00_video_2_events/outputs/{event_data_file}.mat', {'td': td})
