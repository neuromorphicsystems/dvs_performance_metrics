import subprocess
from tqdm import tqdm
import dvs_warping_package
import argparse
import time


''' Example:
python run_several_sims.py --amp 4
python run_several_sims.py --amp 8
python run_several_sims.py --amp 12
python run_several_sims.py --amp 16
python run_several_sims.py --amp 20
python run_several_sims.py --amp 24
'''

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run simulations for different configurations.")
parser.add_argument('--amp', type=int, required=True, help='Simulation index')
args = parser.parse_args()

python_executable = '/home/samiarja/anaconda3/envs/dvs_performance_metric/bin/python' #'/home/sami/anaconda3/envs/dvs_performance_metric/bin/python'
test = 1
amp = args.amp
max_retries = 3

for ep in tqdm(range(1, 3)):
    dvs_warping_package.print_message(f"Config file: T{test}_{amp} epoch: {ep}", color='red', style='bold')
    arg = f'-c frequency_amplitude_heatmap_amp_{amp} {ep}'

    for attempt in range(max_retries):
        try:
            subprocess.run([python_executable, 'run_simulation.py', arg], check=True, timeout=3600)
            break  # Exit retry loop if successful
        except subprocess.TimeoutExpired:
            print(f"Timeout: Retrying {attempt + 1}/{max_retries}...")
            time.sleep(5)
        except subprocess.CalledProcessError as e:
            print(f"Error: Subprocess failed with return code {e.returncode} for epoch {ep}, sim {amp}. Command: {e.cmd}")
            break
        except Exception as e:
            print(f"Unexpected error for epoch {ep}, sim {amp}: {e}")
            break
    else:
        print(f"Failed to complete simulation for epoch {ep}, sim {amp} after {max_retries} attempts.")
