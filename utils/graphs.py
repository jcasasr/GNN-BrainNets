import os
import pickle
import numpy as np
import networkx as nx
from typing import List
import logging
from datetime import datetime
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_diagram

from Subject import Subject
import utils.metrics as metrics
from config import *


def anatomical_contiguous_matrix(matrix, mask:str=""):
    # load mask from csv file
    if mask=="" or not os.path.isfile(mask):
        raise FileNotFoundError("Mask file not found: {}".format(mask))
    mask_data = np.loadtxt(mask, delimiter=',', dtype=int)
    if mask_data.shape != (NUM_NODES,NUM_NODES):
        raise ValueError("Mask file has incorrect shape: {}".format(mask_data.shape))
    
    # Create anatomical-contiguous version of the matrix
    # Multiply the original matrix by the mask to filter connections between non-contiguous regions
    matrix_ac = np.multiply(matrix, mask_data)

    return matrix_ac

def threshold_matrix(matrix:np.ndarray, thr:float=0.0, copy:bool=True):
    if copy:
        matrix_thr = np.copy(matrix)
    else:
        matrix_thr = matrix

    # Apply threshold
    matrix_thr[matrix_thr < thr] = 0

    return matrix_thr


def check_matrix(matrix_name:str, matrix:np.ndarray, min_value:float=0, max_value:float=1, data_type:str="SL") -> bool:
    # Check size
    if data_type == "SL":
        assert matrix.shape == (NUM_NODES,NUM_NODES), "Incorrect shape {} in matrix {}!".format(matrix.shape, matrix_name)
    else:
        assert matrix.shape == (NUM_NODES*2,NUM_NODES*2), "Incorrect shape {} in matrix {}!".format(matrix.shape, matrix_name)
    # Check values
    assert np.min(matrix) >= min_value, "Values < {} in matrix {}!".format(min_value, matrix_name)
    assert np.max(matrix) <= max_value, "Values > {} in matrix {}!".format(max_value, matrix_name)
    # Check diagonal
    assert np.all(matrix.diagonal() == 0) == True, "Diagonal values > 0 in matrix {}!".format(matrix_name)

    return True


def check_subject(subject:Subject, logger=logging) -> None:
    # Check all matrices (SL and ML)
    matrix_names = subject.get_all_matrices_keys()

    for matrix_name in matrix_names:
        matrix = subject.get_matrix(matrix_name)
        if matrix is not None:
            if "ML" in matrix_name:
                check_matrix(matrix_name=matrix_name, matrix=matrix, min_value=0, max_value=1, data_type="ML")
            else:
                check_matrix(matrix_name=matrix_name, matrix=matrix, min_value=0, max_value=1, data_type="SL")
        else:
            raise AssertionError("Matrix not found!")

    logger.info("   Subject {} checked!".format(subject.get_ID()))


def create_multilayer(FA, RS, GM) -> np.ndarray:
    # data structure
    ml = np.zeros((NUM_NODES*2, NUM_NODES*2), dtype=float)
    ml[:NUM_NODES,:NUM_NODES] = np.abs(RS[:,:])
    ml[NUM_NODES:,NUM_NODES:] = np.abs(GM[:,:])
    ml[NUM_NODES:,:NUM_NODES] = np.abs(FA[:,:])
    ml[:NUM_NODES,NUM_NODES:] = np.abs(FA[:,:])
        
    return ml


def compute_ml_metrics(subject: Subject, matrix_types: List[str], metric_list: List[str], logger=logging, debug=True) -> Subject:
    """
    Compute metrics for a subject
    :param subject: Subject object
    :param matrix_types: List of matrix types to process
    :param metric_list: List of metrics to compute
    :param logger: Logger object
    :return: Subject object with metrics
    """
    # get ID 
    ID = subject.get_ID()
    
    # get data type (i.e. matrix name)
    for data_type in matrix_types:
        # get data 
        A = subject.get_matrix(data_type)
        bool_reshape = True

        logger.info("+++ Computing {} metrics for subject {}".format(data_type, ID))
        
        for metric in metric_list:
            # compute the specific metric
            logger.debug("Computing {}...".format(metric))

            try:
                if metric=='Degree':
                    temp = np.count_nonzero(A > 0, axis=0)

                elif metric=='Strength':
                    temp = np.sum(A, axis=0)

                elif metric=='LocalEfficiency':
                    # Use 'distance'
                    A_inv = metrics.create_distance_A_from_A(A)

                    # Compute A min
                    A_min = metrics.compute_A_min(A_inv)

                    # create G
                    G = metrics.create_graph_from_AM(A_min)

                    # compute LE as a single layer
                    temp = metrics.compute_LE_SL(G)

                    # Do not reshape 
                    bool_reshape = False

                elif metric=='ClosenessCentrality':
                    # Use 'distance'
                    A_inv = metrics.create_distance_A_from_A(A)

                    # Compute A min
                    A_min = metrics.compute_A_min(A_inv)

                    # create G
                    G = metrics.create_graph_from_AM(A_min)

                    # compute LE as a single layer
                    temp = np.array(list(nx.closeness_centrality(G, distance='weight').values()))
                    
                    # Do not reshape 
                    bool_reshape = False

                elif metric=='BetweennessCentrality':
                    # Use 'distance'
                    A_inv = metrics.create_distance_A_from_A(A)

                    # Compute A min
                    A_min = metrics.compute_A_min(A_inv)

                    # create G
                    G = metrics.create_graph_from_AM(A_min)

                    # compute LE as a single layer
                    temp = np.array(list(nx.betweenness_centrality(G, k=None, normalized=True, weight='weight').values()))
                    
                    # Do not reshape 
                    bool_reshape = False
                
                else:
                    raise Exception("ERROR: Incorrect metric value! (METRIC is {})".format(metric))
            except Exception as e:
                    logger.error(e)
                    raise e
        
            # Folding results to get 76 nodes (instead of 152)
            if bool_reshape:
                # reshape and sum values
                temp2 = temp.reshape((NUM_NODES,2), order='F')
                temp3 = np.sum(temp2, axis=1)
            else:
                temp3 = temp
            
            # store the results
            key = data_type +"-"+ metric
            subject.set_metric(key, temp3)
            logger.debug("Results of metric {} on {} stored using key='{}'".format(metric, data_type, key))
        
    return subject


def compute_sl_metrics(subject: Subject, matrix_types: List[str], metric_list: List[str], apply_threshold: bool=False, logger=logging, debug=True) -> Subject:
    """
    Compute metrics for a subject
    :param subject: Subject object
    :param logger: Logger object
    :return: Subject object with metrics
    """
    ID = subject.get_ID()
    if debug:
        logger.info("+++ Computing metrics for subject {}".format(ID))

    # get data type (i.e. matrix name)
    for data_type in matrix_types:
        # get data 
        A = subject.get_matrix(data_type, raiseException=False)
        if A is None:
            logger.error("    Matrix {} not found for subject {}. Skipping...".format(data_type, ID))
            raise ValueError("Matrix {} not found for subject {}".format(data_type, ID))

        logger.debug("    Computing data_type = {} metrics for subject: {}".format(data_type, ID))

        if apply_threshold:
            # Apply threshold to the matrix
            A[A < THR[data_type]] = 0
            logger.debug("    Threshold {} applied to {} matrix for subject {}".format(THR[data_type], data_type, ID))
        
        for metric in metric_list:
            # compute the specific metric
            key = data_type +"-"+ metric
            if subject.get_metric(key, raiseException=False) is not None: # check if metric already exists
                logger.debug("    Metric {} on {} already exists. Skipping...".format(metric, data_type))
                continue

            logger.debug("    Computing metric {} on {} ".format(metric, data_type))

            # create graph
            G = metrics.create_graph_from_AM(A)

            # compute metric values
            try:
                if metric=='Degree':
                    temp = np.array(list(nx.degree_centrality(G).values()))

                elif metric=='Strength':
                    temp = np.sum(A, axis=0)

                elif metric=='Clustering':
                    temp = np.array(list(nx.clustering(G, weight='weight').values()))

                elif metric=='BetweennessCentrality':
                    # Use 'distance'
                    G = metrics.create_distance_graph_from_AM(A)

                    # debug only
                    metrics.report_graph_basics(G)

                    temp = np.array(list(nx.betweenness_centrality(G, k=None, normalized=True, weight='weight').values()))

                elif metric=='ClosenessCentrality':
                    # Use 'distance'
                    G = metrics.create_distance_graph_from_AM(A)

                    temp = np.array(list(nx.closeness_centrality(G, distance='weight').values()))

                elif metric=='EigenvectorCentrality':
                    temp = np.array(list(nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06, weight='weight').values()))

                elif metric=='PageRank':
                    temp = np.array(list(nx.pagerank(G, max_iter=1000, weight='weight').values()))

                elif metric=='LocalEfficiency':
                    # Use 'distance'
                    G = metrics.create_distance_graph_from_AM(A)

                    # Implementation of LE on single layer networks
                    temp = metrics.compute_LE_SL(G)
                else:
                    logger.error("ERROR: Incorrect metric value! (METRIC is {})".format(metric))
                    raise Exception("ERROR: Incorrect metric value! (METRIC is {})".format(metric))
            except Exception as e:
                    logger.error("Error computing metric {} on {} for subject {}: {}".format(metric, data_type, ID, e))
                    raise e

            # store the results
            subject.set_metric(key, temp)
            if debug:
                logger.debug("Results of metric {} on {} stored using key='{}'".format(metric, data_type, key))

    return subject


def compute_ph_metric(subject: Subject, matrix_types: List[str], logger=logging) -> Subject:
    """
    Compute PH for a subject
    :param subject: Subject object
    :param logger: Logger object
    :return: Subject object with metrics
    """
    # Params
    homology_dimensions_list = [[0], [0, 1], [0, 1, 2]] # 0; 0 and 1; and 0, 1 and 2 dimensions

    # get data type (i.e. matrix name)
    for data_type in matrix_types:
        # get data 
        A = subject.get_matrix(data_type)
        ID = subject.get_ID()

        for homology_dimensions in homology_dimensions_list:
            logger.debug("Computing {} PH metric for homology dimensions {} for subject {}...".format(data_type, homology_dimensions, ID))

            # compute metric values
            try:
                A = 1 - A
                np.fill_diagonal(A, 0)
                VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=homology_dimensions, n_jobs=-1)
                diagrams = VR.fit_transform([A])

                b = BettiCurve(n_bins=100) # Compute the Betti curve from the persistence diagrams
                temp = b.fit_transform(diagrams) # returns: ndarray of shape (n_samples, n_homology_dimensions, n_bins=100 (default))
                temp = temp.reshape(-1) # as a flat array
            except Exception as e:
                    print(e)
                    raise e

            # store the results
            key = data_type +"-PH_" + "-".join(map(str, homology_dimensions))
            subject.set_metric(key, temp)
            logger.debug("Results of {} PH metric for homology dimensions {} stored using key='{}'".format(data_type, homology_dimensions, key))

    return subject