import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.distance import euclidean, pdist, cdist
from sklearn.manifold import TSNE

from Subject import Subject
import utils.utils as utils
from config import *

from utils.vae import *


def generate_synthetic_samples(model, original_samples, random_scale, num_samples, latent_dim):
    """
    """
    # Convert to tensor
    original_samples_tensor = torch.tensor(np.array(original_samples), dtype=torch.float32)

    # Set model to evaluation mode
    with torch.no_grad():
        mu, logvar = model.encode(original_samples_tensor)
        latent_vectors_original = model.reparameterize(mu, logvar)

    # Compute the mean latent vector for this feature
    mean_latent_vector_original = latent_vectors_original.mean(dim=0)
    
    # Number of new samples to generate
    latent_vectors_synthetic = mean_latent_vector_original + random_scale * torch.randn(num_samples, latent_dim)

    with torch.no_grad():
        # Generate samples
        synthetic_samples = model.decode(latent_vectors_synthetic)
    
    # Convert generated samples to numpy arrays
    synthetic_samples = synthetic_samples.numpy()

    return synthetic_samples, latent_vectors_original.cpu().numpy(), latent_vectors_synthetic.cpu().numpy()


def info_cohorts_per_mstype(subjects_per_mstype, ms_type:int, percentile=100):
    """
    Print information about the original cohorts for each MS_TYPE.
    """
    cohort = np.array(subjects_per_mstype[ms_type])
    print(f"\n*** MS_TYPE {ms_type}:")
    print(f"Number of samples: {len(cohort)}")
    print(f"Mean value       : {np.mean(cohort):.4f} +- {np.std(cohort):.4f}")
    print(f"Min value        : {np.min(cohort):.4f}")
    print(f"Max value        : {np.max(cohort):.4f}")

    # Compute diameter and radius
    diameter = compute_diameter(cohort, percentile=percentile)
    print(f"Diameter         : {diameter:.4f}")
    radius = compute_radius(cohort, percentile=percentile)
    print(f"Radius           : {radius:.4f}")

    return np.mean(cohort), np.min(cohort), np.max(cohort), diameter, radius


def compute_distance_between_centroids(cohort_1, cohort_2):
    """
    Compute the distance between the centroids of the original and synthetic cohorts.
    The centroid is defined as the mean of the samples in the cohort.
    """
    # Compute centroids
    cohort_1_centroid = np.mean(cohort_1, axis=0)
    cohort_2_centroid = np.mean(cohort_2, axis=0)

    # Compute distance between centroids
    distance = euclidean(cohort_1_centroid, cohort_2_centroid)

    return distance, cohort_1_centroid, cohort_2_centroid


def compute_diameter(cohort, percentile=100):
    """
    Compute the diameter of the cohort.
    The diameter is defined as the maximum distance between any two samples in the cohort.
    If percentile < 100, compute the diameter as the distance between the samples at the given percentile of the pairwise distances.
    """
    # Compute pairwise distances for original cohort
    distances = pdist(cohort, metric='euclidean')
    if percentile < 100:
        diameter = np.percentile(distances, percentile)
    else:
        diameter = np.max(distances)

    return diameter


def compute_radius(cohort, percentile=100):
    """
    Compute the radius of the cohort.
    The radius is defined as the maximum distance from the center to any sample in the cohort.
    If percentile < 100, compute the radius as the distance between the center and the sample at the given percentile of the pairwise distances.
    """
    # Compute the center of the original cohort
    center = np.mean(cohort, axis=0)

    # Compute distances from the center for original cohort
    distances = cdist([center], cohort, metric='euclidean')[0]
    if percentile < 100:
        radius = np.percentile(distances, percentile)
    else:
        radius = np.max(distances)

    return radius


def plot_tsne(dataset, cohort, matrix_type, space, data_type):
    """
    Plot the t-SNE of the original and synthetic latent spaces.
    """
    dataset_1 = dataset[0]  # MS_TYPE 0
    dataset_2 = dataset[1]  # MS_TYPE 1

    # Combine original and synthetic latent vectors
    combined_latent = np.vstack((dataset_1, dataset_2))
    labels = np.array([0] * len(dataset_1) + [1] * len(dataset_2))

    # Apply t-SNE
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(combined_latent)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[labels==0, 0], tsne_results[labels==0, 1], label=f'MS_TYPE 0 (n={len(dataset_1)})', alpha=0.5)
    plt.scatter(tsne_results[labels==1, 0], tsne_results[labels==1, 1], label=f'MS_TYPE 1 (n={len(dataset_2)})', alpha=0.5)

    plt.title(f't-SNE - Space: {space}, Data type: {data_type}, Cohort: {cohort}, Matrix: {matrix_type}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/TSNE_{cohort}-{matrix_type}-{space}-{data_type}.png')


def plot_mean_error(original_samples, synthetic_samples, cohort, matrix_type, ms_type):
    real_mean = np.mean(original_samples, axis=0)
    real_std = np.std(original_samples, axis=0)
    synth_mean = np.mean(synthetic_samples, axis=0)
    synth_std = np.std(synthetic_samples, axis=0)
    x = list(range(len(real_mean)))

    # Visualize results with subplots
    fig, axes = plt.subplots(3, 1, figsize=(18, 9))

    # First subplot: Real data
    axes[0].errorbar(x=x, y=real_mean, yerr=real_std, label="Real dataset", color='green', alpha=0.7, linewidth=1)
    axes[0].set_ylabel("Values", fontsize=12)
    axes[0].set_xlabel("Feature Index", fontsize=12)
    axes[0].set_title("Cohort={} Matrix={} MS_TYPE={} Real dataset".format(cohort, matrix_type, ms_type), fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Second subplot: Synthetic data
    axes[1].errorbar(x=x, y=synth_mean, yerr=synth_std, label="Synthetic dataset", color='blue', alpha=0.7, linewidth=1)
    axes[1].set_ylabel("Values", fontsize=12)
    axes[1].set_xlabel("Feature Index", fontsize=12)
    axes[1].set_title("Cohort={} Matrix={} MS_TYPE={} Synthetic dataset".format(cohort, matrix_type, ms_type), fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Third subplot: Error
    error_mean = np.sum(np.abs(real_mean - synth_mean))
    axes[2].plot(np.abs(real_mean - synth_mean), label="Error", color='red', alpha=0.7, linewidth=1)
    axes[2].set_ylabel("Error Values", fontsize=12)
    axes[2].set_xlabel("Feature Index", fontsize=12)
    axes[2].set_title("Cohort={} Matrix={} MS_TYPE={} ABS(Real - Synthetic) dataset (Error {:<.2f})".format(cohort, matrix_type, ms_type, error_mean), fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'figs/VAE_mean_error_{cohort}-{matrix_type}-{ms_type}.png')