from hyperspectral_soil.features import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline(
    use_indices=True,
    use_derivatives=True,
    derivative_order=1,
    use_pca=True,
    n_components=10
)

X_features = pipeline.transform(X_processed, wavelengths)
