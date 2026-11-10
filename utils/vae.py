import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import logging

from Subject import Subject
import utils.utils as utils
from config import *


# Parameters
INPUT_DIM = 2850  # input dimension (number of features)
# Parameters obtained from hyperparameter optimization
PARAMS_VAE = {"BCN": {"FA": {'latent_dim': 14, 'hidden_dim': 20, 'batch_size': 16, 'num_epochs': 620, 'learning_rate': 0.0007169359426492317},
                      "GM": {'latent_dim': 29, 'hidden_dim': 64, 'batch_size': 32, 'num_epochs': 946, 'learning_rate': 0.0010138193832615822},
                      "RS": {'latent_dim': 25, 'hidden_dim': 37, 'batch_size': 32, 'num_epochs': 877, 'learning_rate': 0.00021181169288541805}},
              "NAP": {"FA": {'latent_dim': 17, 'hidden_dim': 37, 'batch_size': 8, 'num_epochs': 749, 'learning_rate': 0.003851024221698841},
                      "GM": {'latent_dim': 20, 'hidden_dim': 39, 'batch_size': 32, 'num_epochs': 543, 'learning_rate': 0.0004947671902720655},
                      "RS": {'latent_dim': 22, 'hidden_dim': 56, 'batch_size': 32, 'num_epochs': 876, 'learning_rate': 0.006279747548595029}},
              "ALL": {"FA": {'latent_dim': 19, 'hidden_dim': 51, 'batch_size': 32, 'num_epochs': 723, 'learning_rate': 0.0002979472569319667},
                      "GM": {'latent_dim': 19, 'hidden_dim': 45, 'batch_size': 32, 'num_epochs': 693, 'learning_rate': 0.0078512235616698},
                      "RS": {'latent_dim': 13, 'hidden_dim': 45, 'batch_size': 32, 'num_epochs': 612, 'learning_rate': 0.001792735250496199}},
              }
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BasicVAE(nn.Module):
    '''
    A basic Variational Autoencoder with single-layer encoder and decoder
    '''
    def __init__(self, input_dim, latent_dim, hidden_dim=8):
        super(BasicVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Outputs values in range [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
   

def vae_loss(recon_x, x, mu, logvar, tau=0.5, c=1.0):
    '''
    VAE loss function: Reconstruction + KL divergence losses summed over all elements and batch

    :param recon_x: Reconstructed input
    :param x: Original input
    :param mu: Mean of the latent distribution
    :param logvar: Log variance of the latent distribution
    :param tau: Temperature parameter (not used in this implementation)
    :param c: Regularization parameter (not used in this implementation)
    '''
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL Divergence Loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kld_loss / x.size(0)


def train_vae(model, data_loader, epochs, learning_rate, print_interval=100):
    '''
    Train VAE model

    :param model: VAE model
    :param data_loader: DataLoader for training data
    :param epochs: Number of training epochs
    :param learning_rate: Learning rate for optimizer
    :return: Lists of losses and reconstruction MSEs per epoch
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    losses = []
    reconstruction_mse = []

    for epoch in range(epochs):
        total_loss = 0
        total_mse = 0
        for batch in data_loader:
            batch_data = batch[0]
            batch_data = batch_data.float() # Convert to float tensor (float32)
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch_data)
            loss = vae_loss(reconstructed, batch_data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Compute batch-wise MSE for comparison
            mse = nn.MSELoss()(reconstructed, batch_data).item()
            total_mse += mse

        losses.append(total_loss / len(data_loader))
        reconstruction_mse.append(total_mse / len(data_loader))

        if epoch % print_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, MSE: {total_mse:.4f}")
    
    return losses, reconstruction_mse


def get_dataset_as_flattened_adj_matrices(matrix_type=None, subject_pattern:str="", ms_type:str="binary", logger:logging=None):
    '''
    Get dataset as flattened adjacency matrices

    :param matrix_type: Type of matrix to extract ("FA", "GM", "RS")
    :param ms_type: Multiple sclerosis type ("binary" or "multi-class")
    :return: Tuple of (dataset_graphs, dataset_y)
    '''
    dataset_graphs = []
    dataset_y = []
    subjects = utils.load_subjects(path_input=PATH, subject_pattern=subject_pattern, debug=False)
    for subject in subjects:
        A = subject.get_matrix(name=matrix_type)
        A = A[np.triu_indices_from(A, k=1)] # Get upper triangular part, K=1 to avoid diagonal
        dataset_graphs.append(A.flatten())
        dataset_y.append(subject.get_mstype(type=ms_type))

    if logger is not None:
        logger.debug("Loaded {} subjects from cohort '{}' and matrix type '{}'".format(len(dataset_graphs), subject_pattern, matrix_type))
        logger.debug("   dataset_graphs shape: {}".format(np.array(dataset_graphs).shape))
        logger.debug("   dataset_y shape: {}".format(np.array(dataset_y).shape))

    return dataset_graphs, dataset_y


def get_adj_matrix_from_flattened(falttened_matrix, num_nodes: int=NUM_NODES):
    # Set empty adjacency matrix
    A = np.zeros(shape=(num_nodes, num_nodes), dtype=np.float32)
    idx = np.triu_indices_from(A, k=1)
    # Set min and max values
    falttened_matrix[falttened_matrix < 0] = 0
    falttened_matrix[falttened_matrix > 1] = 1
    # Set values into A
    A[idx] = falttened_matrix
    A.T[idx] = A[idx]

    assert np.any(A > 0), "Error, values < 0 in A!"
    assert np.any(A < 1), "Error, values > 1 in A!"

    return A


def get_data_from_mstype(matrix_type, subject_pattern, ms_type, logger:logging=None):
    logger.info(f"=== Loading data from {subject_pattern}, {matrix_type} matrices and MS Type={ms_type} ===")
    # Get dataset as flattened adjacency matrices
    X_train, y_train = get_dataset_as_flattened_adj_matrices(matrix_type=matrix_type, 
                                                                      subject_pattern=subject_pattern, 
                                                                      ms_type="multiclass",
                                                                      logger=logger)
    logger.info("Train set size: {}".format(len(X_train)))
    
    # Select samples 
    subjects_per_mstype = []
    for adj, label in zip(X_train, y_train):
        if label == ms_type:  # label
            subjects_per_mstype.append(adj)

    logger.info("Train set size for MS Type = {}: {}".format(ms_type, len(subjects_per_mstype)))

    return subjects_per_mstype


def create_samples_from_vae(cohort:str, matrix_type:str, subjects_per_mstype, num_samples:int, random_scale:float=0.001, logger:logging=None):
    logger.info(f"=== Loading VAE model for {cohort} and {matrix_type} matrices ===")
    # Load model
    model_name = f'models/VAE_{cohort}_{matrix_type}.pth'  # Pre-trained model file
    latent_dim = PARAMS_VAE[cohort][matrix_type]["latent_dim"]
    hidden_dim = PARAMS_VAE[cohort][matrix_type]["hidden_dim"]

    logger.info("Loading pre-trained VAE model from {}...".format(model_name))
    model = BasicVAE(input_dim=INPUT_DIM, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    # Convert to tensor
    selected_samples_tensor = torch.tensor(np.array(subjects_per_mstype), dtype=torch.float32)

    # Set model to evaluation mode
    with torch.no_grad():
        mu, logvar = model.encode(selected_samples_tensor)
        latent_vectors = model.reparameterize(mu, logvar)

    # Compute the mean latent vector for this feature
    mean_latent_vector = latent_vectors.mean(dim=0)

    # Number of new samples to generate
    latent_samples = mean_latent_vector + random_scale * torch.randn(num_samples, latent_dim)

    with torch.no_grad():
        # Generate samples
        generated_samples = model.decode(latent_samples)
    
    # Convert generated samples to numpy arrays
    generated_samples = generated_samples.numpy()

    return generated_samples
