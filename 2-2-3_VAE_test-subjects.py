import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.distance import euclidean, pdist, cdist
import pandas as pd
from datetime import datetime

from Subject import Subject
import utils.utils as utils
from config import *

from utils.vae import *
from utils.vae_latent_analysis import *

# Parameters
NUM_SAMPLES = 100  # Number of synthetic samples to generate
RANDOM_SCALE = 100.0  # Scale of the random noise added to the mean latent vector
COHORTS = ["BCN", "NAP", "ALL"]
COHORTS_PATTERN = ["BCN-*.pkl", "NAP-*.pkl", "*.pkl"]
MS_TYPES = [0, 1]  # MS_TYPE labels to process (0 and 1)


def compute_statistics(subjects_original_space, subjects_latent_space, cohort, matrix_type, data_type, df:pd.DataFrame, percentile:int=90):
    # Compute the distance between the centroids of the two MS_TYPE cohorts in both original and latent spaces
    distance_original, _, _ = compute_distance_between_centroids(subjects_original_space[0], subjects_original_space[1])
    distance_latent, _, _ = compute_distance_between_centroids(subjects_latent_space[0], subjects_latent_space[1])

    # For each ms_type, compute the mean, min, max, diameter, and radius of the cohorts in both original and latent spaces
    for ms_type in MS_TYPES:
        mean_val_original, min_val_original, max_val_original, diameter_original, radius_original = info_cohorts_per_mstype(subjects_original_space, ms_type, percentile=percentile)
        mean_val_latent, min_val_latent, max_val_latent, diameter_latent, radius_latent = info_cohorts_per_mstype(subjects_latent_space, ms_type, percentile=percentile)
        
        new_row = {"Cohort": cohort, "Matrix_Type": matrix_type, "Subject_Type": data_type, "MS_Type": ms_type, 
                   "O_Num_Samples": len(subjects_original_space[ms_type]), "O_Mean_Value": mean_val_original, "O_Min_Value": min_val_original, "O_Max_Value": max_val_original, "O_Diameter": diameter_original, "O_Radius": radius_original, "O_Distance": distance_original,
                   "L_Num_Samples": len(subjects_latent_space[ms_type]), "L_Mean_Value": mean_val_latent, "L_Min_Value": min_val_latent, "L_Max_Value": max_val_latent, "L_Diameter": diameter_latent, "L_Radius": radius_latent, "L_Distance": distance_latent}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Print TSNE visualization of the original and latent spaces
    plot_tsne(subjects_original_space, cohort, matrix_type, space="Original", data_type=data_type)
    plot_tsne(subjects_latent_space, cohort, matrix_type, space="Latent", data_type=data_type)

    return df
    

if __name__ == "__main__":
    # FOR EACH MATRIX TYPE
    for matrix_type in MATRIX_TYPES_ORIGINAL:
        label = f"##### Synthetic Data Generation with VAE for {matrix_type} matrices #####"
        print("\n"+"="*len(label))
        print(label)
        print("="*len(label))

        # create pandas dataframe to store the results
        df = pd.DataFrame(columns=["Cohort", "Matrix_Type", "Subject_Type", "MS_Type", 
                                    "O_Num_Samples", "O_Mean_Value", "O_Min_Value", "O_Max_Value", "O_Diameter", "O_Radius", "O_Distance",
                                    "L_Num_Samples", "L_Mean_Value", "L_Min_Value", "L_Max_Value", "L_Diameter", "L_Radius", "L_Distance"])
        
        # FOR EACH COHORT
        for cohort, subject_pattern in zip(COHORTS, COHORTS_PATTERN):
            label = f"=== Analysis for cohort {cohort} ==="
            print("\n"+"#"*len(label))
            print(label)
            print("#"*len(label))
            
            # Load model
            model_name = f'models/VAE_{cohort}_{matrix_type}.pth'  # Pre-trained model file
            latent_dim = PARAMS_VAE[cohort][matrix_type]["latent_dim"]
            hidden_dim = PARAMS_VAE[cohort][matrix_type]["hidden_dim"]

            # Get dataset as flattened adjacency matrices
            X_train, y_train = get_dataset_as_flattened_adj_matrices(matrix_type=matrix_type, 
                                                                              subject_pattern=subject_pattern, 
                                                                              ms_type="binary",
                                                                              logger=None)
            print("Train set size:", len(X_train))

            # Load the pre-trained VAE model
            print("Loading pre-trained VAE model from {}...".format(model_name))
            model = BasicVAE(input_dim=INPUT_DIM, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
            model.load_state_dict(torch.load(model_name))
            model.eval()

            ### PRINT INFORMATION ABOUT THE LATENT SPACE
            label = "+++ INFORMATION ABOUT THE ORIGINAL AND LATENT SPACE: +++"
            print("\n"+ "+" * len(label))
            print(label)
            print(f"NUM_SAMPLES: {NUM_SAMPLES}")
            print(f"RANDOM_SCALE: {RANDOM_SCALE}")
            print("+" * len(label))

            # Store centroids for each MS_TYPE
            centroids = {}
            original_per_mstype = {}
            synthetic_per_mstype = {}
            combined_per_mstype = {}
            latent_original = {}
            latent_synthetic = {}
            latent_combined = {}

            # Load original samples for each MS_TYPE
            for ms_type in MS_TYPES:
                # Select samples for this MS_TYPE
                original_per_mstype[ms_type] = []
                for adj, label in zip(X_train, y_train):
                    if label == ms_type:  # MS Type label
                        original_per_mstype[ms_type].append(adj)

                print(f"+++ MS_TYPE {ms_type}: {len(original_per_mstype[ms_type])} samples")
                print(f"    Params:\n\tRandom_scale: {RANDOM_SCALE}\n\tNum_samples: {NUM_SAMPLES}")

                # Generate synthetic samples for this MS_TYPE
                synthetic_per_mstype[ms_type], latent_original[ms_type], latent_synthetic[ms_type] = generate_synthetic_samples(model, original_per_mstype[ms_type], RANDOM_SCALE, NUM_SAMPLES, latent_dim)
                
                # Combine original and synthetic samples for this MS_TYPE
                combined_per_mstype[ms_type] = np.vstack((original_per_mstype[ms_type], synthetic_per_mstype[ms_type]))
                latent_combined[ms_type] = np.vstack((latent_original[ms_type], latent_synthetic[ms_type]))
            
            # Print information about the ORIGINAL cohorts
            df = compute_statistics(subjects_original_space=original_per_mstype, subjects_latent_space=latent_original, cohort=cohort, matrix_type=matrix_type, data_type="Original", df=df)

            # Print information about the SYNTHETIC cohorts
            df = compute_statistics(subjects_original_space=synthetic_per_mstype, subjects_latent_space=latent_synthetic, cohort=cohort, matrix_type=matrix_type, data_type="Synthetic", df=df)

            # Print information about the COMBINED cohorts
            df = compute_statistics(subjects_original_space=combined_per_mstype, subjects_latent_space=latent_combined, cohort=cohort, matrix_type=matrix_type, data_type="Combined", df=df)

            ### PRINT SUMMARY INFORMATION
            # Print error between original and synthetic cohorts
            for ms_type in MS_TYPES:
                plot_mean_error(original_per_mstype[ms_type], synthetic_per_mstype[ms_type], cohort, matrix_type, ms_type=ms_type)

        ### Export the results to a Excel file
        timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
        df.to_excel(f"results/{timestamp}_VAE_analysis_{matrix_type}_rs-{RANDOM_SCALE}.xlsx", index=False)