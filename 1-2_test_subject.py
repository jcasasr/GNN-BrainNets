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
        subject = subjects[0]
        print("\n*** Analyzing the data of subject Nº 0 ***\n")

    # check all available attributes
    print("Attributes:")
    for key in subject._attributes.keys():
        value = subject.get_attribute(key)
        if value is not None:
            print("   {} [type: {}, value: {}]".format(key, type(value), value))
        else:
            # if the attribute is None, we can still print its name
            # but we don't know its type or value
            print("   {}".format(key))

    # check all available matrices
    print("Matrices  :")
    for key in subject._matrices.keys():
        value = subject.get_matrix(key)
        if value is not None:
            print("   {} [shape: {}, range: {:.4f} - {:.4f}, {:.2f} % edges]".format(key, value.shape, np.min(value), np.max(value), 100*np.sum(value>0)/np.size(value)))
        else:
            # if the matrix is None, we can still print its name
            # but we don't know its shape or range
            print("   {}".format(key))

    # check all available metrics
    print("Metrics   :")
    for key in subject._metrics.keys():
        value = subject.get_metric(key)
        if value is not None:
            print("   {} [shape: {}, range: {:.4f} - {:.4f}]".format(key, value.shape, np.min(value), np.max(value)))
        else:
            # if the metric is None, we can still print its name
            # but we don't know its shape or range
            print("   {}".format(key))
