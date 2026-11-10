from Subject import Subject
from config import *
import utils.utils as utils
import os
import numpy as np
import networkx as nx


# Get info for each cohort
pattern_list = ["BCN-*.pkl", "NAP-*.pkl"]


def compute_density(matrix):
    """
    Compute the density of a graph represented by an adjacency matrix.
    
    Parameters:
    matrix (numpy.ndarray): The adjacency matrix of the graph.
    
    Returns:
    float: The density of the graph.
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(matrix)
    
    # Compute the density
    density = nx.density(G)
    
    return density

# global store densities for each subject
glb_densities_fa = []
glb_densities_gm = []
glb_densities_rs = []

for pattern in pattern_list:
    # load data
    subjects = utils.load_subjects(path_input=PATH, subject_pattern=pattern, debug=False)
    # temporarily store densities for each subject
    densities_fa = []
    densities_gm = []
    densities_rs = []
    
    for subject in subjects:
        # Compute density for each matrix
        fa_density = compute_density(subject.get_matrix("FA"))
        gm_density = compute_density(subject.get_matrix("GM"))
        rs_density = compute_density(subject.get_matrix("RS"))
        
        densities_fa.append(fa_density)
        densities_gm.append(gm_density)
        densities_rs.append(rs_density)

        glb_densities_fa.append(fa_density)
        glb_densities_gm.append(gm_density)
        glb_densities_rs.append(rs_density)

    # Print the results
    print("")
    print("*" * 60)
    print(f"Density results for subjects with pattern '{pattern}':")
    print("*" * 60)
    print("FA Density: Mean = {:.4f}, Std = {:.4f}".format(np.mean(densities_fa), np.std(densities_fa)))
    print("GM Density: Mean = {:.4f}, Std = {:.4f}".format(np.mean(densities_gm), np.std(densities_gm)))
    print("RS Density: Mean = {:.4f}, Std = {:.4f}".format(np.mean(densities_rs), np.std(densities_rs)))
    print("")

# Print the global results
print("")
print("*" * 40)
print("Global Density results for all subjects:")
print("*" * 40)
print("FA Density: Mean = {:.4f}, Std = {:.4f}".format(np.mean(glb_densities_fa), np.std(glb_densities_fa)))
print("GM Density: Mean = {:.4f}, Std = {:.4f}".format(np.mean(glb_densities_gm), np.std(glb_densities_gm)))
print("RS Density: Mean = {:.4f}, Std = {:.4f}".format(np.mean(glb_densities_rs), np.std(glb_densities_rs)))
print("")
