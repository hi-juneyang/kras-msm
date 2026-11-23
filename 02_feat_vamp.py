import mdtraj as md
import pyemma
import numpy as np
import itertools
import pandas as pd
import os
import json
import logging

# =================
# Try possible featurization (torsion angles vs distances)
# and select preliminary features using pooled VAMP-2
# Save feature arrays for ITS and MSM analysis
# VAMP = Variational Approach for Markov Processes.
# - Uses time-lagged correlations to identify combinations of features that capture predictable molecular dynamics.
# - Ensure all file paths are correct and adjust code as required. 
# =================

# Parameters

# python list 'systems' allow us to loop the code for each item
systems = [
    'WT_GDP',
    'WT_GTP',
    'G12D_GDP',
    'G12D_GTP',
    'acK104_GDP',
    'acK104_GTP',
    'acK147_GDP',
    'acK147_GTP'
]

lag_frames = 10  # Preliminary VAMP lag time, in frames
vamp_dim = 5     # Preliminary number of VAMP components (dimensions) for scoring

# Initialize lists of feature arrays for all 8 systems
angle_data = []
distance_data = []
combined_data = []

distance_resseq = [ 
    12, 32, 34, 36, 48, 56, 59, 63, 66,
    67, 74, 105, 108, 122, 126, 138, 148, 153  # verify this data because this is based on the 'peaks' on the RMSD plot, and  18 elements will produce 153 unique pairs of distances, which is a lot potentially...
]

# Create directories to save file
os.makedirs('vamp/features', exist_ok=True)
os.makedirs('vamp/scoring', exist_ok=True)

# Create log file
logging.basicConfig(
    filename='vamp/vamp.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', # give us timestamps. so we can have a record of how long compute canada takes in each process
    force=True
)

logging.info('Starting pooled VAMP-2 analysis')
logging.info(f'Systems: {systems}')
logging.info(f'Lag time: {lag_frames} frames')
logging.info(f'VAMP dimension: {vamp_dim}')


# Extract features from all 8 systems
for system in systems:

    pdb_file = f'clean/{system}_clean.pdb'
    dcd_file = f'clean/{system}_clean.dcd'

    logging.info(f'Extracting features from {system}...')

    # Torsion angles
    feat_angle = pyemma.coordinates.featurizer(pdb_file)

    feat_angle.add_backbone_torsions(
        selstr='resSeq 25 to 40 or resSeq 60 to 75', # based off on the switch regions
        cossin=True,
        deg=False
    )

    extractAngle = pyemma.coordinates.load(
        dcd_file,
        features=feat_angle
    )

    # Alpha carbon distances
    feat_dist = pyemma.coordinates.featurizer(pdb_file)
    top = md.load(pdb_file).topology

    # Map PDB residue sequence numbers to MDTraj residue indices
    resseq_to_index = {}

    for residueList in top.residues:
        if residueList.is_protein:
            resseq_to_index[residueList.resSeq] = residueList.index

    distance_residues = [
        resseq_to_index[number]
        for number in distance_resseq
    ]

    dist_pairs = list(
        itertools.combinations(distance_residues, 2)
    )

    feat_dist.add_distances_ca(
        dist_pairs,
        periodic=False
    )

    extractDist = pyemma.coordinates.load(
        dcd_file,
        features=feat_dist
    )

    # Alpha carbon distance + torsion angles combined
    extractCombined = np.concatenate(
        [extractAngle, extractDist],
        axis=1
    )

    # Store extracted features
    angle_data.append(extractAngle)
    distance_data.append(extractDist)
    combined_data.append(extractCombined)

    # Save all feature representations
    np.save(
        f'vamp/features/{system}_torsions.npy',
        extractAngle
    )
    np.save(
        f'vamp/features/{system}_dist.npy',
        extractDist
    )
    np.save(
        f'vamp/features/{system}_combined.npy',
        extractCombined
    )

# Compare candidate feature representations
candidates = {
    'torsions': angle_data,
    'distances': distance_data,
    'combined': combined_data
}

# Save scores for each candidate
scores = []
for feature, data in candidates.items():
    logging.info(f'Scoring {feature}...')

    n_features = data[0].shape[1] # shape[1] = number of feature (columns)

    for trajectory in data:
        if trajectory.shape[1] != n_features:
            raise ValueError(
                f'Inconsistent feature dimensions: {feature}' #raise an error if the number of feature extracted is inconsistent across systems
            )

    if n_features < vamp_dim: # We're performing a dimensionality reduction, so we need to have more features than the number of dimensions we want to reduce to. If not, we skip this feature representation.
        logging.warning(f'Skipping {feature}')
        continue

    # construct a VAMP model pooling all 8 systems per candidate
    vamp_model = pyemma.coordinates.vamp(
        data,
        lag=lag_frames,
        dim=vamp_dim
    )
    score = vamp_model.score(
        score_method='VAMP2'
    )
    logging.info(f'{feature} VAMP-2 score: {score}')
    scores.append({
        'feature_set': feature,
        'lag_frames': lag_frames,
        'dimension': vamp_dim,
        'VAMP2_score': score
    })


# Save VAMP results
results = pd.DataFrame(scores)

results.to_csv(
    'vamp/scoring/vampScores.csv', # summarizing the results in CSV table for our reference
    index=False
)


# Select preliminary feature representation
best_index = results['VAMP2_score'].idxmax()
selected_feature = results.loc[best_index, 'feature_set']

logging.info(f'Selected feature: {selected_feature}')


# Save selected features for ITS (we're saving the file again, so we are extra sure about which one we're using for the downstream code without confusion)
os.makedirs('vamp/features/selected', exist_ok=True)

selected_data = candidates[selected_feature]

for system, data in zip(systems, selected_data):
    np.save(
        f'vamp/features/selected/{system}.npy',
        data
    )
selection_info = {
    'feature_set': selected_feature,
    'systems': systems
}

logging.info('Pooled VAMP-2 analysis completed successfully')
