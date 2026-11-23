import os
import mdtraj as md

# ==============
# Use mdtraj library to process the raw NAMD data, remove the water molecules 
# We use the 'cleantraj' as an input for pyemma
# ==============

rawtraj = md.load('system1.dcd', top='system1.pdb') 

# give topology information of rawtraj (DCD + pdb) 
topology = rawtraj.topology

# select everything except for water from top, and save
selection = topology.select("protein or resname GDP or resname GTP or element MG") 
 
# put selection into a trajectory and save 
cleantraj = rawtraj.atom_slice(selection)
cleantraj.save_dcd("clean.dcd")
cleantraj.save_pdb ("clean.pdb")