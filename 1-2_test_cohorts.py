from Subject import Subject
from config import *
import utils.utils as utils
import os
import numpy as np


# Get info for each cohort
pattern_list = ["BCN-*.pkl", "NAP-*.pkl"]

for pattern in pattern_list:
    # load data
    subjects = utils.load_subjects(path_input=PATH, subject_pattern=pattern, debug=False)
    mstypes = [x.get_mstype(type="multiclass") for x in subjects]
    print("")
    print("*" * 40)
    print(f"Subjects with pattern '{pattern}':")
    print("*" * 40)
    print("   Total: {}".format(len(subjects)))
    print("   HV:    {}".format(mstypes.count(0)))
    print("   RRMS:  {}".format(mstypes.count(1)))
    print("   SPMS:  {}".format(mstypes.count(2)))
    print("   PPMS:  {}".format(mstypes.count(3)))
    print("")

    # Select the first subject for demonstration
    if subjects is None or len(subjects) == 0:
        print("No subjects found. Please check the input path and subject pattern.")
        exit(1)
    else:
        print("Found {} subjects.".format(len(subjects)))
        subject_zero = subjects[0]
    
    print("\n*** Analyzing the data of ALL subjects ***\n")

    # check all available attributes
    print("Attributes:")
    for key in subject_zero._attributes.keys():
        values = []
        for subject in subjects:
            values.append(subject.get_attribute(key))
        if isinstance(values[0], str):
            print("   {} [type: {}, value: {}]".format(key, type(values[0]), values[0]))
        else:
            print("   {} [type: {}, value range: min={:.4f}, max={:.4f}, mean={:.4f}+-{:.4f}]".format(
                key, type(values[0]), np.min(values), np.max(values), np.mean(values), np.std(values)))
    # check all available matrices
    print("Matrices  :")
    for key in subject_zero._matrices.keys():
        values = []
        mins = []
        maxs = []
        means = []
        edges = []
        for subject in subjects:
            values.append(subject.get_matrix(key))
            mins.append(np.min(subject.get_matrix(key)))
            maxs.append(np.max(subject.get_matrix(key)))
            means.append(np.mean(subject.get_matrix(key)))
            edges.append(np.sum(subject.get_matrix(key)>0) / np.size(subject.get_matrix(key)))
        print("   {} [shape: {}, range: min={:.4f}+-{:.4f}, max={:.4f}+-{:.4f}, mean={:.4f}+-{:.4f}, % edges={:.2f}+-{:.2f}]".format(
            key, values[0].shape, np.min(mins), np.std(mins), np.max(maxs), np.std(maxs), np.mean(means), np.std(means), np.mean(edges)*100, np.std(edges)*100))
        
    # check all available metrics
    print("Metrics   :")
    for key in subject_zero._metrics.keys():
        values = []
        mins = []
        maxs = []
        means = []
        non_zero = []
        for subject in subjects:
            values.append(subject.get_metric(key))
            mins.append(np.min(subject.get_metric(key)))
            maxs.append(np.max(subject.get_metric(key)))
            means.append(np.mean(subject.get_metric(key)))
        print("   {} [shape: {}, range: min={:.4f}+-{:.4f}, max={:.4f}+-{:.4f}, mean={:.4f}+-{:.4f}".format(
            key, values[0].shape, np.min(mins), np.std(mins), np.max(maxs), np.std(maxs), np.mean(means), np.std(means)))