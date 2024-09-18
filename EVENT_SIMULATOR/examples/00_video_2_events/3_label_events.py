from os import XATTR_CREATE
# import cv2
import numpy as np
import sys, os
sys.path.append("../../src")
from dat_files import load_dat_event
# import dvs_sparse_filter
import glob as gb
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from scipy.io import savemat
import dvs_sparse_filter

parent_folder       = f'../../videos'
filename            = f'{parent_folder}/ev_500_50_100_10_0.3_0.01.dat'
ts, x, y, p         = load_dat_event(filename)
frames_timestamp    = np.load("/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/frame_timestamp.npy")

sensor_size = (max(x)+1,max(y)+1)
print(f"res (wxh): {max(x)+1,max(y)+1}")


if not os.path.exists(f"/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/event_frames"):
    os.mkdir(f"/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/event_frames")

if not os.path.exists(f"/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/overlay"):
    os.mkdir(f"/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/overlay")

events = np.zeros(len(x), dtype=[('t', '<u8'),
                                 ('x', '<u2'),
                                 ('y', '<u2'),
                                 ('on', '?'),
                                 ('label', '<i2'),
                                 ])
events['t']     = ts
events['x']     = x
events['y']     = y
events['on']    = p
events['label'] = np.zeros((len(x)))


ground_truth_path = "/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/masks/"
ground_truth_frames = [f for f in os.listdir(ground_truth_path)]

sorted_frames = sorted(ground_truth_frames, key=lambda x: int(x.split('_')[0]))

# image_paths   = gb.glob(f"/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/event_frames/*.png") 
mask_path     = gb.glob(f"/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/masks/*.png")

def extract_number_img(file_path):
    # Find the numeric part using regex
    match = re.search(r'(\d+)_img\.png$', file_path)
    if match:
        return int(match.group(1))
    else:
        return -1
    
def extract_number_mask(file_path):
    # Find the numeric part using regex
    match = re.search(r'(\d+)_labelled_img\.png$', file_path)
    if match:
        return int(match.group(1))
    else:
        return -1

# Sort image paths based on the extracted number
# image_paths_sorted = sorted(image_paths, key=extract_number_img)
mask_path_sorted   = sorted(mask_path, key=extract_number_mask)

labels = np.zeros(len(events), dtype=int)

for tt in tqdm(range(len(frames_timestamp)-1)):
    # previous_frame_path = image_paths_sorted[tt]
    # next_frame_path = image_paths_sorted[tt+1]

    # Load the masks as PIL Images first
    prev_mask = np.array(Image.open(mask_path_sorted[tt]))
    next_mask = np.array(Image.open(mask_path_sorted[tt+1]))
    
    
    previous_timestamp  = frames_timestamp[tt][0]
    next_timestamp      = frames_timestamp[tt][1]
    ii                  = np.where(np.logical_and(events["t"] >= previous_timestamp, events["t"] <= next_timestamp))
    sub_events          = events[ii]
    
    warped_image = dvs_sparse_filter.accumulate(sensor_size, sub_events, (0, 0))
    rendered_image = dvs_sparse_filter.render(warped_image, colormap_name="magma", gamma=lambda image: image ** (1 / 3))
    rendered_image.save(f"/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/event_frames/{tt+1}_img.png")

    binary_image = Image.open(mask_path_sorted[tt])
    rendered_image = rendered_image.convert("RGBA")  # Ensure the rendered image has an alpha channel
    binary_image = binary_image.resize(rendered_image.size)
    binary_image = binary_image.convert("L")
    binary_image = ImageOps.flip(binary_image)
    mask = ImageOps.invert(binary_image)  # Invert so white becomes black and black becomes white
    mask = mask.point(lambda p: 255 if p > 0 else 0)  # Convert all non-black pixels to white
    overlay_image = Image.composite(rendered_image, Image.new("RGBA", rendered_image.size, (0, 0, 0, 0)), mask)
    overlay_image.save(f"/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/overlay/{tt+1}_img.png")
    
    labels = np.zeros(len(sub_events), dtype=int)
    for j, ev in enumerate(sub_events):
        x, y = int(ev[1]), int(ev[2]) 
        
        label_assigned  = False  # Flag to check if a label has been assigned
        radius = 1
        # Check the pixels within the defined radius
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                new_x, new_y = x+dx, y+dy
                
                # Ensure new coordinates are within the image boundaries
                if 0 <= new_x < prev_mask.shape[1] and 0 <= new_y < prev_mask.shape[0]:
                    if prev_mask[new_y, new_x] and next_mask[new_y, new_x]:
                        labels[j] = 1
                        label_assigned = True
                        break # Exit the loop once a label is assigned
            
            if label_assigned: # Exit the outer loop if a label is assigned
                break

    events['label'][ii] = labels

# savemat('/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/labelled_events.mat', {'events': events})
# savemat('/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/labelled_events.mat',{'events': events},do_compression=True)

text_file_path = '/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/labelled_events.txt'
np.savetxt(text_file_path, events, fmt='%d %d %d %d %d', header='t x y on label', comments='')
print("File saved...")

# vx_star_velocity = np.zeros((len(events["x"]), 1)) + 0 / 1e6
# vy_star_velocity = np.zeros((len(events["x"]), 1)) + 0 / 1e6
# cumulative_map_coloured, seg_label_zero = dvs_sparse_filter.accumulate_cnt_rgb(sensor_size, 
#                                                                                events, 
#                                                                                events['label'].astype(np.int32), 
#                                                                                (vx_star_velocity.T, vy_star_velocity.T))
# warped_image_segmentation_coloured = dvs_sparse_filter.rgb_render_white(cumulative_map_coloured, seg_label_zero)
# warped_image_segmentation_coloured.save(f"/home/samiarja/Desktop/PhD/Code/dvs_metrics/videos/labelled_dot_white.png")
