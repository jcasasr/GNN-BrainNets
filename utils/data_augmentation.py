import os
import random
import numpy as np
import networkx as nx
from typing import List
import logging
from datetime import datetime

from Subject import Subject
from config import *


def augment_data_add_noise(subjects, num_to_add, index, logger=logging):
    """
    Augment data by adding Gaussian noise to the adjacency matrices of the subjects.

    Parameters:
    - subjects: List[Subject] - List of Subject objects to augment.
    - num_to_add: int - Number of augmented subjects to create.
    - DA_ADDNOISE_NOISE_LEVEL: float - Standard deviation of the Gaussian noise to add.
    - MATRIX_TYPES: List[str] - List of matrix types to augment.

    Returns:
    - List[Subject] - List of augmented Subject objects.
    """
    augmented_subjects = []
    for i in range(num_to_add):
        # Randomly select a subject to augment
        original_subject = np.random.choice(subjects)
        new_id = "SYN-ADN-" + "{:04}".format(index + i)

        new_subject = Subject(ID=new_id, 
                              cohort=original_subject.get_cohort(), 
                              ID_old=original_subject.get_ID(), 
                              mstype=original_subject._mstype)
        
        for key in original_subject.get_all_matrices_keys():
            if key in MATRIX_TYPES_ORIGINAL:
                original_matrix = original_subject.get_matrix(key)
                # Generate noise
                noise = np.random.normal(1, DA_ADDNOISE_NOISE_LEVEL, original_matrix.shape)
                # Add noise
                noisy_matrix = original_matrix * noise
                # Ensure the matrix remains valid (e.g., non-negative, symmetric if needed)
                noisy_matrix = np.clip(a=noisy_matrix, a_min=0, a_max=1)  # Ensure weights in range [0,1]
                noisy_matrix = (noisy_matrix + noisy_matrix.T) / 2  # Ensure symmetry
                np.fill_diagonal(noisy_matrix, 0)  # No self-loops

                new_subject.set_matrix(key, noisy_matrix)
            else:
                logger.warning(f"Matrix type {key} not in MATRIX_TYPES_ORIGINAL, skipping augmentation for this matrix.")
    
        augmented_subjects.append(new_subject)

    return augmented_subjects


def augment_data_interpolate(subjects, num_to_add, index, logger=logging):
    """
    Augment data by interpolating the adjacency matrices of two subjects.

    Parameters:
    - subjects: List[Subject] - List of Subject objects to augment.
    - num_to_add: int - Number of augmented subjects to create.
    - MATRIX_TYPES: List[str] - List of matrix types to augment.

    Returns:
    - List[Subject] - List of augmented Subject objects.
    """
    cohorts = list(set([subject.get_cohort() for subject in subjects]))

    augmented_subjects = []
    for i in range(num_to_add):
        # Randomly select two subjects to interpolate
        cohort = np.random.choice(cohorts)
        subjects_cohort = [subject for subject in subjects if subject.get_cohort()==cohort]
        original_subjects = np.random.choice(subjects_cohort, size=2, replace=True)
        new_id = "SYN-INT-" + "{:04}".format(index + i)

        new_subject = Subject(ID=new_id, 
                              cohort=cohort, 
                              ID_old=f"{original_subjects[0].get_ID()}_{original_subjects[1].get_ID()}",
                              mstype=original_subjects[0]._mstype)
        
        matrix_keys_0 = sorted(original_subjects[0].get_all_matrices_keys())
        matrix_keys_1 = sorted(original_subjects[1].get_all_matrices_keys())

        if matrix_keys_0 != matrix_keys_1:
            logger.warning(f"Selected subjects don't have same matrix types, skipping augmentation for this subjects.")
            continue

        for key in matrix_keys_0:
            if key in MATRIX_TYPES_ORIGINAL:
                original_matrix_0 = original_subjects[0].get_matrix(key)
                original_matrix_1 = original_subjects[1].get_matrix(key)

                t = np.random.uniform()
                interpolated_matrix = t * original_matrix_0 + (1 - t) * original_matrix_1

                new_subject.set_matrix(key, interpolated_matrix)
            else:
                logger.warning(f"Matrix type {key} not in MATRIX_TYPES_ORIGINAL, skipping augmentation for this matrix.")
    
        augmented_subjects.append(new_subject)

    return augmented_subjects


def augment_data_node_permutation(subjects, num_to_add, index, logger=logging):
    """
    Augment data by adding Gaussian noise to the adjacency matrices of the subjects.

    Parameters:
    - subjects: List[Subject] - List of Subject objects to augment.
    - num_to_add: int - Number of augmented subjects to create.
    - DA_ADDNOISE_NOISE_LEVEL: float - Standard deviation of the Gaussian noise to add.
    - MATRIX_TYPES: List[str] - List of matrix types to augment.

    Returns:
    - List[Subject] - List of augmented Subject objects.
    """
    augmented_subjects = []

    # Subjects to augment
    if num_to_add >= len(subjects):
        idxs = list(range(0, len(subjects)))
    else:
        idxs = random.sample(range(0, len(subjects)), num_to_add)

    for i in idxs:
        # Get subject to augment
        original_subject = subjects[i]
        # Create new subject
        new_id = "SYN-NOP-" + "{:04}".format(index + i)
        new_subject = Subject(ID=new_id, 
                              cohort=original_subject.get_cohort(), 
                              ID_old=original_subject.get_ID(), 
                              mstype=original_subject._mstype)
        
        for key in original_subject.get_all_matrices_keys():
            if key in MATRIX_TYPES_ORIGINAL:
                original_matrix = original_subject.get_matrix(key)
                noisy_matrix = node_permutation(original_matrix)
                new_subject.set_matrix(key, noisy_matrix)
            else:
                logger.warning(f"Matrix type {key} not in MATRIX_TYPES_ORIGINAL, skipping augmentation for this matrix.")
    
        augmented_subjects.append(new_subject)

    return augmented_subjects


def node_permutation(adj):
    """
    Apply a fixed node permutation to the adjacency matrix.

    Parameters:
    - adj: np.ndarray - Adjacency matrix to permute.
    
    Returns:
    - np.ndarray - Permuted adjacency matrix.
    """
    # Params
    permutations = [
        (0,45),(1,46),(2,47),(3,48),(4,49),(5,50),(6,51),(7,52),
        (8,53),(9,54),(10,55),(11,56),(12,57),(13,58),(14,59),
        (15,60),(16,61),(17,62),(18,63),(19,64),(20,65),(21,66),
        (22,67),(23,68),(24,69),(25,70),(26,71),(27,72),(28,73),
        (29,74),(30,75),(31,38),(32,39),(33,40),(34,41),(35,42),
        (36,43),(37,44)]
    # Vars
    A = adj.copy()

    # Apply permutations
    for p in permutations:
        A[p, :] = A[list(reversed(p)), :]
        A[:, p] = A[:, list(reversed(p))]

    return A


def augment_data(method, subjects, num_to_add, index, logger=logging):
    if method == "AddNoise":
        logger.info(f"  - Noise level: {DA_ADDNOISE_NOISE_LEVEL}")
        augmented_subjects = augment_data_add_noise(subjects, num_to_add, index, logger=logger)
    elif method == "Interpolate":
        augmented_subjects = augment_data_interpolate(subjects, num_to_add, index, logger=logger)
    elif method == "NodePermutation":
        augmented_subjects = augment_data_node_permutation(subjects, num_to_add, index, logger=logger)
    elif method == "MixGraphs":
        raise NotImplementedError("MixGraphs method is not implemented yet.")
    else:
        raise ValueError(f"Unknown data augmentation method: {method}")

    return augmented_subjects


def create_synthetic_subjects(subjects: List[Subject], method: str, da_target_sizes: List[int], logger=logging) -> List[Subject]:
    # load data
    subjects_hv = [subject for subject in subjects if subject.get_mstype(type="multiclass")==0]
    subjects_ms1 = [subject for subject in subjects if subject.get_mstype(type="multiclass")==1]
    subjects_ms2 = [subject for subject in subjects if subject.get_mstype(type="multiclass")==2]
    subjects_ms3 = [subject for subject in subjects if subject.get_mstype(type="multiclass")==3]

    logger.info(f"Total subjects.   : {len(subjects)}")
    logger.info(f"  - HV            : {len(subjects_hv)}")
    logger.info(f"  - pwMS all types: {len(subjects_ms1) + len(subjects_ms2) + len(subjects_ms3)}")
    logger.info(f"    + type 1 RRMS : {len(subjects_ms1)}")
    logger.info(f"    + type 2 SPMS : {len(subjects_ms2)}")
    logger.info(f"    + type 3 PPMS : {len(subjects_ms3)}")

    # Data augmentation
    logger.info(f"+++ Data augmentation method: {method}")
    new_subjects = []
    index = 0
    # HV
    num_to_add = da_target_sizes[0] - len(subjects_hv)
    if num_to_add > 0:
        subjects_hv_augmented = augment_data(method, subjects_hv, num_to_add, index, logger=logger)
        new_subjects.extend(subjects_hv_augmented)
        logger.info(f"    + HV: {len(subjects_hv)} -> {len(subjects_hv) + len(subjects_hv_augmented)}")
    else:
        logger.info(f"    + HV: No augmentation needed ({len(subjects_hv)} >= {DA_TARGET_SIZES[0]})")
    index += num_to_add
    # MS type 1
    num_to_add = da_target_sizes[1] - len(subjects_ms1)
    if num_to_add > 0:
        subjects_ms1_augmented = augment_data(method, subjects_ms1, num_to_add, index, logger=logger)
        new_subjects.extend(subjects_ms1_augmented)
        logger.info(f"    + MS type 1: {len(subjects_ms1)} -> {len(subjects_ms1) + len(subjects_ms1_augmented)}")
    else:
        logger.info(f"    + MS type 1: No augmentation needed ({len(subjects_ms1)} >= {DA_TARGET_SIZES[1]})")
    index += num_to_add
    # MS type 2
    num_to_add = da_target_sizes[2] - len(subjects_ms2)
    if num_to_add > 0:
        subjects_ms2_augmented = augment_data(method, subjects_ms2, num_to_add, index, logger=logger)
        new_subjects.extend(subjects_ms2_augmented)
        logger.info(f"    + MS type 2: {len(subjects_ms2)} -> {len(subjects_ms2) + len(subjects_ms2_augmented)}")
    else:
        logger.info(f"    + MS type 2: No augmentation needed ({len(subjects_ms2)} >= {DA_TARGET_SIZES[2]})")
    index += num_to_add
    # MS type 3
    num_to_add = da_target_sizes[3] - len(subjects_ms3)
    if num_to_add > 0:
        subjects_ms3_augmented = augment_data(method, subjects_ms3, num_to_add, index, logger=logger)
        new_subjects.extend(subjects_ms3_augmented)
        logger.info(f"    + MS type 3: {len(subjects_ms3)} -> {len(subjects_ms3) + len(subjects_ms3_augmented)}")
    else:
        logger.info(f"    + MS type 3: No augmentation needed ({len(subjects_ms3)} >= {DA_TARGET_SIZES[3]})")

    # Save new subjects
    if len(new_subjects) > 0:
        logger.info(f"  - New subjects created: {len(new_subjects)}")
    else:
        logger.info(f"  - No new subjects created")

    return new_subjects