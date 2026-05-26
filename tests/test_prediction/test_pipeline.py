import pytest
import numpy as np
from biomekit.prediction.pipeline import MicrobiomePipeline

def test_pipeline_fit_predict():
    """Test pipeline fit and predict."""
    X = np.random.rand(100, 50) + 0.1
    y = np.random.randint(0, 2, 100)

    pipeline = MicrobiomePipeline(
        preprocess='clr',
        encode=None,
        classifier='rf'
    )
    pipeline.fit(X, y)
    pred = pipeline.predict(X)
    assert len(pred) == 100

def test_pipeline_evaluate():
    """Test pipeline evaluation with cross-validation."""
    X = np.random.rand(100, 50) + 0.1
    y = np.random.randint(0, 2, 100)

    pipeline = MicrobiomePipeline(
        preprocess='clr',
        classifier='rf'
    )
    results = pipeline.evaluate(X, y, cv=3)
    assert 'accuracy' in results
    assert 'auc_roc' in results
    assert results['accuracy'][0] >= 0  # Mean should be valid

def test_pipeline_with_autoencoder():
    """Test pipeline with autoencoder encoding."""
    X = np.random.rand(80, 100).astype(np.float32) + 0.1
    y = np.random.randint(0, 2, 80)

    pipeline = MicrobiomePipeline(
        preprocess='clr',
        encode='autoencoder',
        classifier='rf',
        latent_dim=16
    )
    pipeline.fit(X, y)
    pred = pipeline.predict(X)
    assert len(pred) == 80

def test_get_feature_importance():
    """Test feature importance extraction."""
    X = np.random.rand(60, 40) + 0.1
    y = np.random.randint(0, 2, 60)

    pipeline = MicrobiomePipeline(classifier='rf')
    pipeline.fit(X, y)
    importance = pipeline.get_feature_importance()
    assert importance is not None
    assert len(importance) > 0