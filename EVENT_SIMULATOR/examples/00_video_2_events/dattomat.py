import loris
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt


event_data_file = "ev_500.0_350.0_0.6_0.5_0.02"
my_file = loris.read_file(f"OUTPUT/events/{event_data_file}.dat")
events = my_file['events']
print(events)


td = {
      'x': events["x"], 
      'y': events["y"], 
      'p': events["p"], 
      'ts': events["ts"]
      }

# Save the dictionary into a MATLAB .mat file
savemat(f'OUTPUT/events/{event_data_file}.mat', {'td': td})
