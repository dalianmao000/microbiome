import pytest
import numpy as np
import torch
from biomekit.prediction.encoders import AutoencoderEncoder, VAEEncoder

def test_autoencoder_dim_reduction():
    """Test autoencoder reduces dimensions correctly."""
    X = np.random.rand(50, 100).astype(np.float32) + 0.1
    encoder = AutoencoderEncoder(latent_dim=16, hidden_dim=32, epochs=10)
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (50, 16), f"Expected (50, 16), got {X_encoded.shape}"
    assert not np.any(np.isnan(X_encoded))

def test_autoencoder_passthrough():
    """Test autoencoder with None encoding (passthrough)."""
    X = np.random.rand(30, 50).astype(np.float32) + 0.1
    encoder = AutoencoderEncoder(latent_dim=50, hidden_dim=32, epochs=5)
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (30, 50)  # No reduction if latent >= input

def test_vae_dim_reduction():
    """Test VAE reduces dimensions correctly."""
    X = np.random.rand(50, 100).astype(np.float32) + 0.1
    encoder = VAEEncoder(latent_dim=16, hidden_dim=32, epochs=10)
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (50, 16), f"Expected (50, 16), got {X_encoded.shape}"
    assert not np.any(np.isnan(X_encoded))

def test_vae_passthrough():
    """Test VAE with latent >= input (passthrough)."""
    X = np.random.rand(30, 50).astype(np.float32) + 0.1
    encoder = VAEEncoder(latent_dim=50, hidden_dim=32, epochs=5)
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (30, 50)  # No reduction if latent >= input