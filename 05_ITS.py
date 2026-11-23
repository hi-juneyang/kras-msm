import pyemma
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging

# =================
# Implied timescales (ITS) plot (x axis = lag time, y axis = implied time scale)
# Use discrete trajectories from k-means to calculate ITS
# Check the plots to select the lag time for MSM
# correct lag time to choose = where implied time scale starts to plateau  (in which frame the system stablizes?)
# =================

# Parameters
candidates = [1, 2, 5, 10, 20, 40, 80] # lag times to test in frames

# Output setup
os.makedirs('results/its', exist_ok=True)

logging.basicConfig(
    filename='results/its/its.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

logging.info('Starting ITS analysis')
logging.info(f'Lag time candidates: {candidates}')

# Load system information from VAMP
with open('vamp/features/selected/selection.json', 'r') as file:
    selection_info = json.load(file)

systems = selection_info['systems']
logging.info(f'Systems: {systems}')

# Load discrete trajectories and calculate ITS for each system
for system in systems:
    dtraj = np.load(
    f'results/kmeans/{system}_dtraj.npy'
    )

    its_model = pyemma.msm.its(
        [dtraj],
        lags=candidates,
        nits=5,
        errors=None
    )

    # Plot ITS to see where timescales become stable
    plt.figure()
    pyemma.plots.plot_implied_timescales(
        its_model,
        units='frames'
    )

    plt.title(f'Implied Timescales - {system}')
    plt.savefig(
        f'results/its/{system}_its.png',
        dpi=300
    )
    plt.close()

logging.info('ITS analysis complete')