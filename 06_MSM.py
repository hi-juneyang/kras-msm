import pyemma
import numpy as np
import os
import json
import logging

# =================
# Markov State Model (MSM)
# Estimate transition probabilities between microstates
# for all 8 KRAS systems
# =================

# Parameters
msm_lag = 10 # arbitrary number, update after ITS analysis detremines the correct lag time

# Output setup
os.makedirs('results/msm', exist_ok=True)

logging.basicConfig(
    filename='results/msm/msm.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Load system information from VAMP
with open('vamp/features/selected/selection.json', 'r') as file:
    selection_info = json.load(file)

systems = selection_info['systems']
logging.info(f'Systems: {systems}')

# Build MSM for each system
for system in systems:
    logging.info(f'Building MSM for {system}')

    # Load discrete trajectory from k-means
    dtraj = np.load(
        f'results/kmeans/{system}_dtraj.npy'
    )

    # Estimate MSM
    msm_model = pyemma.msm.estimate_markov_model(
        [dtraj],
        lag=msm_lag
    )

    # Save transition matrix
    transition_matrix = msm_model.transition_matrix
    np.save(
        f'results/msm/{system}_transition_matrix.npy',
        transition_matrix
    )

    # Save MSM model for PCCA+ and future analysis
    msm_model.save(
        f'results/msm/{system}_model.pyemma',
        overwrite=True
    )

    logging.info(f'{system}: MSM complete')