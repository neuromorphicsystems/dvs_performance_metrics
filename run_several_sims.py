# import subprocess
# import sys

# # Path to your virtual environment's Python executable
# python_executable = r'C:\Users\30067913\Anaconda3\envs\dvs_performance_metric\python.exe'
# test = 1
# # Loop through simulations
# for sim in range(1, 2):
#     for ep in range(1,3):
#         # Construct the argument as in MATLAB (e.g., T1_1, T1_2, ...)
#         arg = f'-c T{test}_{sim} {ep}'
        

#         # Run the Python script with the specific argument
#         subprocess.run([python_executable, 'run_simulation.py', arg], check=True)

#         # Print the progress message
#         print(f'done running simulation epoc #{ep}, run #{sim}, from test #{test}1')


import subprocess
import tqdm
import dvs_warping_package

# Path to your virtual environment's Python executable (update this with the correct path)
python_executable = '/home/sami/anaconda3/envs/dvs_performance_metric/bin/python'
test = 1
sim  = 6

# Loop through simulations
# for sim in range(1, 7):

for ep in range(1, 3):
    # Construct the argument as in MATLAB (e.g., T1_1, T1_2, ...)
    dvs_warping_package.print_message(f"Config file: T{test}_{sim} epoch: {ep}", color='red', style='bold')

    arg = f'-c T{test}_{sim} {ep}'
    
    # Run the Python script with the specific argument
    subprocess.run([python_executable, 'run_simulation.py', arg], check=True)

    # Print the progress message
    print(f'done running simulation epoc #{ep}, run #{sim}, from test #{test}1')
