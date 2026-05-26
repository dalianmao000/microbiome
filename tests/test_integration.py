import numpy as np
from biomekit.prediction import MicrobiomePipeline

def test_prediction_pipeline_integration():
    """Test full prediction pipeline with simulated data."""
    # Generate simulated data using numpy directly (simulating microbiome abundance data)
    np.random.seed(42)
    n_samples = 100
    n_features = 200

    # Simulate abundance-like data (non-negative)
    X = np.random.rand(n_samples, n_features) + 0.1
    # Binary classification target
    y = np.random.randint(0, 2, n_samples)

    # Run full pipeline
    pipeline = MicrobiomePipeline(preprocess='clr', classifier='rf', encode=None)
    pipeline.fit(X, y)

    results = pipeline.evaluate(X, y, cv=3)
    assert results['accuracy'][0] >= 0  # Mean should be valid
    assert results['accuracy'][0] <= 1.0

    # Test get_feature_importance
    importance = pipeline.get_feature_importance()
    assert importance is not None

    print(f"Pipeline test passed. Accuracy: {results['accuracy'][0]:.3f} ± {results['accuracy'][1]:.3f}")