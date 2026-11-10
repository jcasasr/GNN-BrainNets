from Subject import Subject
from config import *
import utils.utils as utils
import os
import numpy as np
import sys


def print_subject_info(subject:Subject=None):
    print("Subject ID: {}".format(subject.get_ID()))
    print("Cohort    : {}".format(subject.get_cohort()))
    print("Old ID    : {}".format(subject.get_ID_old()))
    print("MS type   : {} (binary: {})".format(subject.get_mstype(type="multiclass"), subject.get_mstype(type="binary")))

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
            print("   {} [shape: {}, range: {:.4f} - {:.4f}, mean: {:.4f} +- {:.4f}, {:.2f} % edges]".format(
                key, value.shape, np.min(value), np.max(value), np.mean(value), np.std(value), 100*np.sum(value>0)/np.size(value)))
        else:
            # if the matrix is None, we can still print its name
            # but we don't know its shape or range
            print("   {}".format(key))

    # check all available metrics
    print("Metrics   :")
    for key in subject._metrics.keys():
        value = subject.get_metric(key)
        if value is not None:
            print("   {} [shape: {}, range: {:.4f} - {:.4f}, mean: {:.4f} +- {:.4f}]".format(
                key, value.shape, np.min(value), np.max(value), np.mean(value), np.std(value)))
        else:
            # if the metric is None, we can still print its name
            # but we don't know its shape or range
            print("   {}".format(key))


def __main__():
    if sys.argv[1:]:
        pattern = sys.argv[1]
    else:
        print("No subject pattern provided. ")
        sys.exit(1)

    # Load the subject from the specified path and pattern
    subjects = utils.load_subjects(path_input=PATH, subject_pattern=pattern, debug=False)

    # Select the first subject for demonstration
    if subjects is None or len(subjects) == 0:
        print("No subjects found. Please check the input path and subject pattern.")
        exit(1)
    else:
        print("Found {} subjects.".format(len(subjects)))
        subject = subjects[0]
        print("\n*** Analyzing the data of subject {} ***\n".format(subject.get_ID()))

    # Print the information of the selected subject
    print_subject_info(subject)

if __name__ == "__main__":
    __main__()