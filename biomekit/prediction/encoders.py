"""Autoencoder modules for dimensionality reduction."""
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Autoencoder(nn.Module):
    """Simple autoencoder for microbiome data."""

    def __init__(self, input_dim, hidden_dim=32, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class AutoencoderEncoder(BaseEstimator, TransformerMixin):
    """Autoencoder for dimensionality reduction.

    Parameters:
        latent_dim: Dimension of latent representation
        hidden_dim: Dimension of hidden layer
        epochs: Number of training epochs
        lr: Learning rate
    """

    def __init__(self, latent_dim=16, hidden_dim=32, epochs=100, lr=0.001):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.input_dim_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]

        # Don't encode if latent >= input
        if self.latent_dim >= self.input_dim_:
            return self

        self.model = Autoencoder(self.input_dim_, self.hidden_dim, self.latent_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        X_tensor = torch.from_numpy(X)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            _, reconstructed = self.model(X_tensor)
            loss = nn.MSELoss()(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()

        return self

    def transform(self, X):
        # Passthrough if not fitted or latent >= input
        if self.model is None or self.latent_dim >= self.input_dim_:
            return np.asarray(X, dtype=np.float32)

        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X)
        with torch.no_grad():
            latent, _ = self.model(X_tensor)
        return latent.numpy()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class VAEEncoder(BaseEstimator, TransformerMixin):
    """Variational Autoencoder for dimensionality reduction.

    Provides probabilistic latent representation.
    """

    def __init__(self, latent_dim=16, hidden_dim=32, epochs=100, lr=0.001):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.input_dim_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]

        if self.latent_dim >= self.input_dim_:
            return self

        # Simple VAE implementation
        class VAE(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc21 = nn.Linear(hidden_dim, latent_dim)
                self.fc22 = nn.Linear(hidden_dim, latent_dim)
                self.fc3 = nn.Linear(latent_dim, hidden_dim)
                self.fc4 = nn.Linear(hidden_dim, input_dim)

            def encode(self, x):
                h = torch.relu(self.fc1(x))
                return self.fc21(h), self.fc22(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                h = torch.relu(self.fc3(z))
                return self.fc4(h)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        self.model = VAE(self.input_dim_, self.hidden_dim, self.latent_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.from_numpy(X)
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon, mu, logvar = self.model(X_tensor)
            recon_loss = nn.MSELoss()(recon, X_tensor)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.001 * kl_loss
            loss.backward()
            optimizer.step()

        return self

    def transform(self, X):
        if self.model is None or self.latent_dim >= self.input_dim_:
            return np.asarray(X, dtype=np.float32)

        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X)
        with torch.no_grad():
            mu, _ = self.model.encode(X_tensor)
        return mu.numpy()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)