from Subject import Subject
from config import *
import utils.utils as utils
import numpy as np
from scipy.stats import chisquare, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def t_test(dataset1, dataset1_label, dataset2, dataset2_label, label="", min_value=None, max_value=None):
    """
    Perform a t-test between two datasets, to see if they come from the same distribution.

    Parameters:
    dataset1 (list or array-like): First dataset.
    dataset2 (list or array-like): Second dataset.

    Returns:
    tuple: t-statistic and p-value of the t-test.
    """
    sum_dataset1 = [np.sum(x) for x in dataset1]
    sum_dataset2 = [np.sum(x) for x in dataset2]

    # set min and max values
    if min_value is None:
        min_value = min(min(sum_dataset1), min(sum_dataset2))
    if max_value is None:
        max_value = max(max(sum_dataset1), max(sum_dataset2))

    # set colors
    #colors = ['Blue', 'Green']
    colors = ["#5D83C3", "#D66060"]  # Blue and red
    colors_hard = ["#0095FF", "#FF0E0E"]  # Harder blue and orange

    t_statistic, p_value = ttest_ind(sum_dataset1, sum_dataset2)

    print(f"### Comparing {dataset1_label} and {dataset2_label} for {label}:")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.8f}")
    print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}\n")

    ######################################### 
    # PLOT 1
    # Plot the distributions
    ###########################################
    plt.figure(figsize=(12, 8))

    # Subplot 1: Histograms
    plt.subplot(2, 2, 1)
    plt.hist(sum_dataset1, alpha=0.7, label='Dataset 1 (BCN)', bins=20, color=colors[0])
    plt.hist(sum_dataset2, alpha=0.7, label='Dataset 2 (NAP)', bins=20, color=colors[1])
    plt.xlabel(f'{label} Sum Values')
    plt.ylabel('Frequency')
    plt.title('Distribution Comparison - Histograms')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Box plots
    plt.subplot(2, 2, 2)
    data_to_plot = [sum_dataset1, sum_dataset2]
    labels = ['BCN', 'NAP']
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel(f'{label} Sum Values')
    plt.title('Distribution Comparison - Box Plots')
    plt.grid(True, alpha=0.3)

    # Subplot 3: Violin plots
    plt.subplot(2, 2, 3)
    positions = [1, 2]
    plt.violinplot([sum_dataset1, sum_dataset2], positions=positions)
    plt.xticks(positions, labels)
    plt.ylabel(f'{label} Sum Values')
    plt.title('Distribution Comparison - Violin Plots')
    plt.grid(True, alpha=0.3)

    # Subplot 4: Statistical summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
    Statistical Test Results:

    T-statistic: {t_statistic:.4f}
    P-value: {p_value:.4f}
    Significance level: α = 0.05
    Result: {'Significant' if p_value < 0.05 else 'Not significant'}

    Dataset 1 (BCN):
    Mean: {np.mean(sum_dataset1):.4f}
    Std: {np.std(sum_dataset1):.4f}
    N: {len(sum_dataset1)}

    Dataset 2 (NAP):
    Mean: {np.mean(sum_dataset2):.4f}
    Std: {np.std(sum_dataset2):.4f}
    N: {len(sum_dataset2)}

    Mean Difference: {np.mean(sum_dataset1) - np.mean(sum_dataset2):.4f}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.suptitle(f'DTI Sum Comparison: BCN vs NAP (p = {p_value:.4f})', y=1.02)
    plt.savefig('figs/coherence_per_individual_'+ label +'_1.png', format='png', dpi=300, bbox_inches='tight')

    ######################################### 
    # PLOT 2 
    # Additional detailed plot with seaborn #
    ##########################################
    plt.figure(figsize=(10, 6))

    # Combine data for seaborn
    combined_data = np.concatenate([sum_dataset1, sum_dataset2])
    group_labels = ['BCN'] * len(sum_dataset1) + ['NAP'] * len(sum_dataset2)

    # Create DataFrame for seaborn
    df = pd.DataFrame({
        label +'_Sum': combined_data,
        'Dataset': group_labels
    })

    # Seaborn distribution plot
    sns.histplot(data=df, x=label+'_Sum', hue='Dataset', alpha=0.6, kde=True, palette=colors)

    #plt.title(f'{label} Sum Distribution by Dataset (t = {t_statistic:.3f}, p = {p_value:.4f})', fontsize=14)
    plt.title(f'(t-test = {t_statistic:.3f}, p = {p_value:.4f})', fontsize=16)
    #plt.xlabel(f'{label} Sum Values', fontsize=14)
    plt.xlabel('Sum Values', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xlim(min_value - 5, max_value + 5)

    # Add vertical lines for means
    mean1 = np.mean(sum_dataset1)
    mean2 = np.mean(sum_dataset2)
    plt.axvline(mean1, color=colors_hard[0], linestyle='--', alpha=0.8, label=f'{dataset1_label}')
    plt.axvline(mean2, color=colors_hard[1], linestyle=':', alpha=0.8, label=f'{dataset2_label}')
    plt.legend(fontsize=16, loc='upper left')
    plt.grid(True, alpha=0.4)
    plt.savefig('figs/coherence_per_individual_'+ label +'_2.png', format='png', dpi=300, bbox_inches='tight')

    ######################################### 
    # PLOT 3 
    # Plot the violin distributions
    ###########################################
    plt.figure(figsize=(10, 6))

    positions = [1, 2]
    plots = plt.violinplot([sum_dataset1, sum_dataset2], positions=positions)
    plt.xticks(positions, labels, fontsize=16)
    #plt.ylabel(f'{label} Sum Values', fontsize=14)
    plt.ylabel('Sum Values', fontsize=16)
    plt.xlabel('Cohort', fontsize=16)
    #plt.title('Distribution Comparison - Violin Plots', fontsize=14)
    plt.grid(True, alpha=0.4)
    # Set the color of the violin patches
    for pc, color in zip(plots['bodies'], colors):
        pc.set_facecolor(color)
    plt.savefig('figs/coherence_per_individual_'+ label +'_3.png', format='png', dpi=300, bbox_inches='tight')

    return t_statistic, p_value


# Get matrix list
matrix_list = ["FA", "GM", "RS"]

# load data
subjects_BCN = utils.load_subjects(path_input=PATH, subject_pattern="BCN-*.pkl", debug=False)
subjects_NAP = utils.load_subjects(path_input=PATH, subject_pattern="NAP-*.pkl", debug=False)

# For each matrix type
for matrix in matrix_list:
    # Compare all subjects of the two cohorts
    values_BCN = [x.get_matrix(name=matrix) for x in subjects_BCN]
    values_NAP = [x.get_matrix(name=matrix) for x in subjects_NAP]

    # Compute min and max values to set the same x-axis limits in all plots
    sum_dataset1 = [np.sum(x) for x in values_BCN]
    sum_dataset2 = [np.sum(x) for x in values_NAP]
    min_value = min(min(sum_dataset1), min(sum_dataset2))
    max_value = max(max(sum_dataset1), max(sum_dataset2))
    
    t_test(values_BCN, "BCN", values_NAP, "NAP", label=matrix +"_All", min_value=min_value, max_value=max_value)

    # Compare HV of the two cohorts
    values_BCN = [x.get_matrix(name=matrix) for x in subjects_BCN if x.get_mstype(type="binary") == 0]
    values_NAP = [x.get_matrix(name=matrix) for x in subjects_NAP if x.get_mstype(type="binary") == 0]
    
    t_test(values_BCN, "BCN", values_NAP, "NAP", label=matrix +"_HV", min_value=min_value, max_value=max_value)

    # Compare PwMS of the two cohorts
    values_BCN = [x.get_matrix(name=matrix) for x in subjects_BCN if x.get_mstype(type="binary") > 0]
    values_NAP = [x.get_matrix(name=matrix) for x in subjects_NAP if x.get_mstype(type="binary") > 0]
    
    t_test(values_BCN, "BCN", values_NAP, "NAP", label=matrix +"_PwMS", min_value=min_value, max_value=max_value)
