# Markov State Modeling (MSM) Pipeline

### Development of Python-based computational pipeline to analyze Molecular Dynamics (MD) data using Markov State Modelling (MSMs)


This project is part of computational biochemistry research at the Schulich School of Medicine & Dentistry, Western University, under the supervision of Dr. Teklab Gebregiworgis.

This repository documents MSM analysis workflow architecture and Python scripts.

My contributions include developing the computational methodology, writing and reviewing Python scripts, and designing the workflow around research objectives and computational constraints, including computation resources and analysis across multiple simulation systems.

## Research Context

The objective of this project is to investigate how KRAS mutations influence protein conformational dynamics in GDP-bound and GTP-bound states.

Eight KRAS systems are investigated:
- Wild-type KRAS (GDP/GTP)
- G12D mutant KRAS (GDP/GTP)
- K104-acetylated KRAS (GDP/GTP)
- K147-acetylated KRAS (GDP/GTP)

Each system is solvated and ionized using VMD, with corresponding PDB and PSF files. Molecular dynamics trajectories are generated using NAMD, with 300 million simulation steps and three independent random seeds per system.
The analysis pipeline applies feature extraction, dimensionality reduction, clustering, and kinetic modeling to characterize conformational states and their transitions.

## Computational Workflow
1. **Trajectory preprocessing:** Clean MD trajectories and select relevant molecular components.
2. **Feature extraction and VAMP-2 scoring:** Evaluate backbone torsion angles, Cα distances, and combined representations.
3. **Time-lagged Independent Component Analysis (tICA):** Reduce the feature space to slow collective coordinates.
4. **K-means clustering:** Discretize conformations into microstates.
5. **Implied Timescales (ITS):** Evaluate MSM lag-time convergence.
6. **Markov State Model (MSM):** Estimate transition probabilities and stationary distributions.
7. **PCCA+:** Identify metastable conformational macrostates.

The pipeline is designed to process eight KRAS systems while preserving intermediate outputs and logging the process. 

## Technical Details
- **Language:** Python
- **Libraries:** MDTraj, PyEMMA 2.5.7, NumPy, Matplotlib, Pandas
- **Additional modules:** os, json, logging, itertools
- **Environment:** High-performance computing (HPC) clusters
- **Simulation software:** NAMD, VMD
- **Input files:** PDB, PSF, DCD
- **Outputs:** NumPy arrays, transition matrices, model files, plots, and analysis summaries

## Development Status
The scripts have been is very preliminary with analysis workflow. Further development includes optimizing modeling parameters, validation of methodology from biological perspective.
Any parameters subject to validation or arbitrary values are indicated as a comment. 
