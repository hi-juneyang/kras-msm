import pyemma
import numpy as np
import os
import json
import logging

# =================
# Time-lagged Independent Component Analysis (tICA)
#  Reduce selected features into slow coordinates across all 8 KRAS systems
# we've 'pooled' the 8 systems and using the same data too run tICA clustering. 
# I believe it is good for comparison, but we need to verify if it's the right approach methodologically. 
# =================

# Parameters
tica_lag = 10       # preliminary lag time (frames)
tica_dim = 5        # preliminary number of components

# Create output directory
os.makedirs('vamp/tica', exist_ok=True)
logging.basicConfig(
    filename='vamp/tica/tica.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logging.info('Starting tICA analysis')
logging.info(f'Lag time: {tica_lag} frames')
logging.info(f'dimension: {tica_dim}')

# Load selected feature information
with open('results/features/selected/selection.json', 'r') as file: # load the vamp output
    selection_info = json.load(file)

systems = selection_info['systems']
selected_feature = selection_info['feature_set']

logging.info(f'Selected feature representation: {selected_feature}')
logging.info(f'Systems: {systems}')

# Load selected feature arrays
feature_data = []
for system in systems:

    data = np.load(
        f'vamp/features/selected/{system}.npy'
    )
    feature_data.append(data)
    logging.info(f'{system} shape: {data.shape}')

# Fit pooled tICA model
tica_model = pyemma.coordinates.tica(
    feature_data,
    lag = tica_lag, # to identify slow tICA components
    dim = tica_dim # arbitrary number, represent number of coordinates to cluster
)

# Transform features into tICA coordinates
tica_data = tica_model.get_output()

# Save tICA projections
os.makedirs('results/tica', exist_ok=True)
for system, data in zip(systems, tica_data):
    np.save(
        f'results/tica/{system}_tica.npy',
        data
    )
    logging.info(f'{system} tICA shape: {data.shape}')
logging.info('tICA completed')
