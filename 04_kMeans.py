import pyemma
import numpy as np
import os
import json
import logging

# =================
# K-means clustering
# Discretize tICA coordinates into microstates across all 8 KRAS systems
# =================

# Parameters, arbitrary number 
n_clusters = 100
max_iter = 50
random_seed = 42

# Output setup
os.makedirs('results/kmeans', exist_ok=True)

logging.basicConfig(
    filename='results/kmeans/kmeans.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logging.info('Starting K-means clustering')

# Load selected featurization saved from VAMP 
with open('results/features/selected/selection.json', 'r') as file:
    selection_info = json.load(file)

systems = selection_info['systems']

logging.info(f'Systems: {systems}')
logging.info(f'Number of clusters: {n_clusters}')

# Load tICA
tica_data = []

for system in systems:
    data = np.load(
        f'results/tica/{system}_tica.npy'
    )
    tica_data.append(data)

# K-means clustering
cluster_model = pyemma.coordinates.cluster_kmeans(
    tica_data,
    k=n_clusters,
    max_iter=max_iter,
    fixed_seed=random_seed
)

# Discrete trajectories
dtrajs = cluster_model.dtrajs

# Save discrete trajectories for downstream analysis 
for system, dtraj in zip(systems, dtrajs):
    np.save(
        f'results/kmeans/{system}_dtraj.npy',
        dtraj
    )
    logging.info(f'{system}: discrete trajectory {dtraj.shape}')

# Clustering diagnostics
all_dtrajs = np.concatenate(dtrajs)
unique_states = np.unique(all_dtrajs)

logging.info(f'Total frames: {len(all_dtrajs)}')
logging.info(f'Populated clusters: {len(unique_states)} / {n_clusters}')
logging.info('K-means clustering completed!')
