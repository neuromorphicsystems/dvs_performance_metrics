from astropy.table.groups import table_group_by
import numpy as np
import scipy.io
import sys
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
sys.path.append("EVENT_SIMULATOR/src")
from dat_files import load_dat_event

''' Example
# Sample data
Signal_events = {
    'x': np.array([1, 2, 2, 3]),
    'y': np.array([1, 1, 2, 3]),
    'on': np.array([1, 0, 1, 0])
}

BG_events = {
    'x': np.array([1, 2, 3, 3]),
    'y': np.array([1, 2, 2, 3]),
    'on': np.array([0, 1, 0, 1])
}

matrix_size = (3, 3)

# Function call
FSNR, F_sig, F_bg = calc_FlickSNR(Signal_events, BG_events, matrix_size)

print("FSNR:", FSNR)
print("F_sig:\n", F_sig)
print("F_bg:\n", F_bg)
'''
# def calc_FlickSNR(Signal_events, BG_events, matrix_size):
#     F_sig = np.zeros((matrix_size[0], matrix_size[1]))
#     F_bg = np.zeros((matrix_size[0], matrix_size[1]))

#     for xi in tqdm(range(matrix_size[0])):
#         for yi in range(matrix_size[1]):
#             xi_index = xi + 1
#             yi_index = yi + 1

#             ind_sig = (Signal_events['x'] == xi_index) & (Signal_events['y'] == yi_index)
#             if np.any(ind_sig):
#                 on_sig = Signal_events['on'][ind_sig]
#                 F_sig[xi, yi] = 2 * np.sum(on_sig) * np.sum(1 - on_sig) / len(on_sig)

#             ind_bg = (BG_events['x'] == xi_index) & (BG_events['y'] == yi_index)
#             if np.any(ind_bg):
#                 on_bg = BG_events['on'][ind_bg]
#                 F_bg[xi, yi] = 2 * np.sum(on_bg) * np.sum(1 - on_bg) / len(on_bg)

#     FSNR = np.max(F_sig) / np.std(F_bg)
#     return FSNR

def calc_FlickSNR(Signal_events, BG_events, matrix_size):
    # Convert 'x' and 'y' indices to integers
    Signal_events['x'] = Signal_events['x'].astype(int)
    Signal_events['y'] = Signal_events['y'].astype(int)
    BG_events['x'] = BG_events['x'].astype(int)
    BG_events['y'] = BG_events['y'].astype(int)

    # Create arrays to store F_sig and F_bg
    F_sig = np.zeros(matrix_size)
    F_bg = np.zeros(matrix_size)

    # Vectorize computation for Signal events
    sig_counts = np.zeros(matrix_size)
    np.add.at(sig_counts, (Signal_events['x'] - 1, Signal_events['y'] - 1), 1)
    
    # Get "on" event counts and total counts per pixel for Signal
    on_counts_sig = np.zeros(matrix_size)
    np.add.at(on_counts_sig, (Signal_events['x'] - 1, Signal_events['y'] - 1), Signal_events['on'])
    
    valid_sig = sig_counts > 0
    F_sig[valid_sig] = 2 * on_counts_sig[valid_sig] * (sig_counts[valid_sig] - on_counts_sig[valid_sig]) / sig_counts[valid_sig]
    
    # Vectorize computation for Background events
    bg_counts = np.zeros(matrix_size)
    np.add.at(bg_counts, (BG_events['x'] - 1, BG_events['y'] - 1), 1)
    
    on_counts_bg = np.zeros(matrix_size)
    np.add.at(on_counts_bg, (BG_events['x'] - 1, BG_events['y'] - 1), BG_events['on'])
    
    valid_bg = bg_counts > 0
    F_bg[valid_bg] = 2 * on_counts_bg[valid_bg] * (bg_counts[valid_bg] - on_counts_bg[valid_bg]) / bg_counts[valid_bg]

    # Calculate Flicker SNR
    FSNR = np.max(F_sig) / np.std(F_bg)
    
    return FSNR,F_sig,F_bg

if __name__ == '__main__':
    target_radius = [0.1,0.3,0.5,0.7,0.9,1.1,1.3,2,3,4,5,6] # TODO: directly retrieve from the .ini file
    FSNR_all      = np.zeros((len(target_radius)))
    
    for i in range(0,len(target_radius)):
        simulation_data_path = f'OUTPUT/masks/simulation_data_target_radius_{float(target_radius[i])}.mat'

        # Check if the file exists
        if not os.path.exists(simulation_data_path):
            print(f"File not found: {simulation_data_path}. Skipping to the next target radius.")
            continue
        
        mat_data = scipy.io.loadmat(simulation_data_path,
                                    struct_as_record=False,
                                    squeeze_me=True)

        simulation_data = mat_data['simulation_data']
        last_simulation = simulation_data[-1]
        event_labels = last_simulation.all_events
        
        event_x = event_labels[:,0]
        event_y = event_labels[:,1]
        event_p = event_labels[:,2]
        event_ts = event_labels[:,3]
        event_l  = event_labels[:,4]
        
        # Indices of signal and background events
        sig_indices = np.where(event_l == 1)[0]
        bg_indices = np.where(event_l == 0)[0]

        # Structure Signal_events
        Signal_events = {
            'x': event_x[sig_indices],
            'y': event_y[sig_indices],
            'on': event_p[sig_indices]
        }

        # Structure BG_events
        BG_events = {
            'x': event_x[bg_indices],
            'y': event_y[bg_indices],
            'on': event_p[bg_indices]
        }
        
        matrix_size = (int(np.max(event_x)) + 1, int(np.max(event_y)) + 1)

        FSNR,F_sig,F_bg = calc_FlickSNR(Signal_events, BG_events, matrix_size)
        print(f"target_radius: {target_radius[i]} FSNR: {FSNR}")
        FSNR_all[i] = FSNR
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(target_radius, FSNR_all, marker='o', linestyle='-', linewidth=2, markersize=8)

    plt.xlabel('Target Radius', fontsize=16, labelpad=10)
    plt.ylabel('Flicker SNR', fontsize=16, labelpad=10)
    plt.title('Performance Metrics', fontsize=20, pad=20)

    # Set x-ticks to match target_radius values
    plt.xticks(target_radius, fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.gca().set_facecolor('#f0f0f0')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig("OUTPUT/FlickSNR_metric_target_radius.png", dpi=300, bbox_inches='tight')
    plt.show()
        
    
