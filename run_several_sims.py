import subprocess
import sys

# Path to your virtual environment's Python executable
python_executable = r'C:\Users\30067913\Anaconda3\envs\dvs_performance_metric\python.exe'

# Loop through simulations
for sim in range(1, 2):
    # Construct the argument as in MATLAB (e.g., T1_1, T1_2, ...)
    arg = f'T12_{sim}'

    # Run the Python script with the specific argument
    subprocess.run([python_executable, 'run_simulation.py', arg], check=True)

    # Print the progress message
    print(f'done running simulation run #{sim} from test #1')