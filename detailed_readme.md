# Ensemble AIRR Predictor - Implementation Guide

This implementation adapts your ensemble training pipeline to fit the AIRR-ML-25 Challenge template requirements.

## Overview

The predictor combines three complementary approaches:
1. **K-mer Features**: Sparse features from 3-mer and 4-mer sequences
2. **Public Clone Model**: Fisher's exact test-based sequence selection
3. **ESM2 Embeddings**: Deep learning protein embeddings (optional, if available)

These models are combined using a meta-learner (Logistic Regression) trained via stacking.

The implementation includes automatic feature generation capabilities - if ESM2 embeddings are not pre-computed, the system can generate them on-the-fly using the ESM2 model.

## Architecture

### Training Pipeline (`fit()`)
1. Load metadata and labels from training directory
2. Auto-detect feature file paths (K-mer, ESM2)
3. Run ESM grid search (if ESM features available):
   - Tests 8 ESM variants (2 BERT pooling × 4 row pooling strategies)
   - Tests 2 ML models (ExtraTrees, SVM)
   - Selects best combination via 5-fold CV
4. Train ensemble:
   - K-mer model (Lasso Logistic Regression)
   - Public Clone model (Fisher + Lasso)
   - Best ESM model (from grid search)
   - Meta-learner on stacked predictions
5. Identify important sequences (public clones)

### Prediction Pipeline (`predict_proba()`)
1. Load test repertoire IDs
2. Extract features (K-mer, sequences, ESM if available)
3. Generate predictions from each base model
4. Combine with meta-learner
5. Return probabilities in required format

## File Structure

```
predict-airr-main/
├── submission/
│   ├── __init__.py
│   ├── main.py              # Entry point (unchanged)
│   ├── predictor.py         # YOUR ENSEMBLE IMPLEMENTATION HERE
│   └── utils.py             # Utility functions (unchanged)
├── requirements.txt         # Updated with ML dependencies
├── test_interface.py        # Interface verification script
└── README_IMPLEMENTATION.md # This file
```

## Configuration

The predictor auto-detects feature paths based on your directory structure:

### Expected Directory Structure
```
data/
├── train_datasets/
│   └── train_dataset_X/
│       ├── metadata.csv
│       ├── {ID}.tsv files
├── kmer/
│   └── k3_k4_train_dataset_X_features.pkl
└── test_datasets/
    └── test_dataset_X_Y/
        ├── {ID}.tsv files
workingfolder/
└── aggregates/
    ├── aggregated_esm2_t6_8M_max/  (BERT max pooling)
    │   ├── esm2_train_dataset_X_aggregated_mean.pkl
    │   ├── esm2_train_dataset_X_aggregated_max.pkl
    │   ├── esm2_train_dataset_X_aggregated_std.pkl
    │   └── esm2_train_dataset_X_aggregated_mean_std.pkl
    └── aggregated_esm2_t6_8M_mean/  (BERT mean pooling)
        ├── esm2_train_dataset_X_aggregated_mean.pkl
        ├── esm2_train_dataset_X_aggregated_max.pkl
        ├── esm2_train_dataset_X_aggregated_std.pkl
        └── esm2_train_dataset_X_aggregated_mean_std.pkl
```

### Custom Paths
You can override default paths when instantiating:

```python
predictor = ImmuneStatePredictor(
    n_jobs=4,
    device='cpu',
    out_dir='/custom/output/',
    kmer_path='/custom/kmer/features.pkl',
    esm_path='/custom/esm/aggregates/',
    auto_generate_esm=True,  # Enable ESM feature generation
    model_selection_method='hybrid'  # 'cv', 'weights', or 'hybrid' (default)
)
```

### Model Selection Strategies

The predictor supports three strategies for selecting which single model to use from the ensemble:

1. **`hybrid` (default)**: Intelligent adaptive selection
   - If any model has weight > 0.5: Uses weight-based selection (strong consensus)
   - If all weights ≤ 0.5: Falls back to CV-based selection (no consensus)
   - **Recommended for most use cases**

2. **`cv`**: Always selects the model with highest individual cross-validation AUC
   - Best for scenarios where standalone model performance is most important
   - Ignores ensemble complementarity

3. **`weights`**: Always selects the model with highest meta-learner weight
   - Best when you trust the ensemble learning to identify the best contributor
   - May select a model with lower individual CV but better complementarity

#### Selection Examples

**Example 1: Strong consensus (hybrid uses weights)**
```
Model Weights: K-mer=0.72, Public=0.01, ESM=0.27
Selection Method: Hybrid (Weight > 0.5 threshold)
Selected Model: K-mer (weight=0.72, CV AUC=0.68500)
```

**Example 2: No consensus (hybrid uses CV)**
```
Model Weights: K-mer=0.04, Public=0.47, ESM=0.49
Selection Method: Hybrid (No weight > 0.5, using CV)
Selected Model: Public (CV AUC=0.71234, weight=0.47)
```

## Usage

### Basic Usage

```bash
python3 -m submission.main \
    --train_dir /path/to/train_dataset_X \
    --test_dirs /path/to/test_dataset_X_1 /path/to/test_dataset_X_2 \
    --out_dir /path/to/output \
    --n_jobs 4
```

### Advanced Usage with ESM Auto-Generation

```bash
python3 -m submission.main \
    --train_dir /path/to/train_dataset_X \
    --test_dirs /path/to/test_dataset_X_1 \
    --out_dir /path/to/output \
    --n_jobs 4 \
    --device cuda
```

When `auto_generate_esm=True` is set in the predictor initialization, the system will automatically generate ESM2 embeddings if pre-computed features are not found. This requires the `save_representation.py` and `aggregate_allpooling.py` scripts to be present in the project root.

### Programmatic Usage

```python
from submission.predictor import ImmuneStatePredictor

# Initialize with default hybrid selection
predictor = ImmuneStatePredictor(
    n_jobs=4, 
    device='cpu',
    out_dir='/path/to/output/',
    model_selection_method='hybrid'  # default, can also be 'cv' or 'weights'
)

# Train
predictor.fit('/path/to/train_dataset_X')

# Predict
predictions_df = predictor.predict_proba('/path/to/test_dataset_X')

# Get important sequences
important_seqs = predictor.important_sequences_
```

#### Using Different Selection Methods

```python
# Force CV-based selection
predictor_cv = ImmuneStatePredictor(
    model_selection_method='cv',
    out_dir='/path/to/output/'
)

# Force weight-based selection
predictor_weights = ImmuneStatePredictor(
    model_selection_method='weights',
    out_dir='/path/to/output/'
)

# Use hybrid (default - no need to specify)
predictor_hybrid = ImmuneStatePredictor(
    out_dir='/path/to/output/'
)
```

## Key Features

### Auto-Detection
- Automatically detects K-mer and ESM feature files based on dataset naming conventions
- Falls back gracefully when features are missing
- Supports both pre-computed and on-the-fly feature generation

### Ensemble Components
- **K-mer Model**: Lasso-regularized logistic regression on sparse k-mer counts (k=3,4). C parameter auto-optimized during training.
- **Public Clone Model**: Fisher's exact test (p<0.05) for sequence selection + logistic regression. Identifies statistically significant public sequences.
- **ESM Model**: Grid search over 8 ESM2 variants:
  - **BERT Pooling**: max vs mean (2 options)
  - **Row Pooling**: mean, max, std, mean_std (4 options)
  - **ML Classifiers**: ExtraTrees (300 trees, depth=6) vs SVM (linear kernel)
  - Selects best variant/classifier combination via 5-fold CV
- **Meta-Learner**: Logistic regression combining base model predictions via stacking
- **Model Selection**: After meta-learning, selects a single model to use:
  - **Hybrid (default)**: Weight-based if any weight > 0.5, otherwise CV-based
  - Computes individual CV AUCs for all models
  - Final predictions use only the selected model (weight=1.0, others=0.0)

### Reproducibility
- Fixed random seeds for all operations
- Deterministic feature ordering
- Single-threaded BLAS operations during training

### Performance Optimization
- Parallel processing support (`n_jobs`)
- GPU acceleration for ESM feature generation (`device='cuda'`)
- Memory-efficient sparse matrices for sequence features

## Dependencies

Key packages (see `requirements.txt`):
- scikit-learn: ML algorithms and utilities
- scipy: Statistical tests and sparse matrices
- pandas/numpy: Data manipulation
- tqdm: Progress bars
- transformers: ESM model loading (if using auto-generation)
- torch: Deep learning framework (if using auto-generation)

## Output Files

The implementation generates three output files per training dataset:

1. `{dataset}_test_predictions.tsv`: Prediction probabilities for test repertoires
2. `{dataset}_important_sequences.tsv`: Top sequences identified by the model
3. `{dataset}_ensemble_model.pkl`: Serialized trained model for reuse

These are automatically concatenated into `submissions.csv` for challenge submission.

### Test Predictions
File: `{dataset_name}_test_predictions.tsv`

| ID | dataset | label_positive_probability | junction_aa | v_call | j_call |
|----|---------|---------------------------|-------------|---------|---------|
| rep001 | test_dataset_8 | 0.742 | -999.0 | -999.0 | -999.0 |
| rep002 | test_dataset_8 | 0.123 | -999.0 | -999.0 | -999.0 |

### Important Sequences
File: `{dataset_name}_important_sequences.tsv`

| ID | dataset | label_positive_probability | junction_aa | v_call | j_call |
|----|---------|---------------------------|-------------|---------|---------|
| train_dataset_8_seq_top_1 | train_dataset_8 | -999.0 | CASSLAPGATNEKLFF | TRBV27*01 | TRBJ1-4*01 |
| train_dataset_8_seq_top_2 | train_dataset_8 | -999.0 | CASSLGQAYEQYF | TRBV6-4*01 | TRBJ2-7*01 |

## Installation

```bash
cd predict-airr-main

# Install dependencies
pip install -r requirements.txt

# Test interface
python test_interface.py
```

## Testing

### Verify Interface
```bash
python test_interface.py
```

### Run Full Pipeline (Small Test)
```bash
# Make sure your feature files exist first!
python3 -m submission.main \
    --train_dir /path/to/train_dataset_X \
    --test_dirs /path/to/train_dataset_X \
    --out_dir ./test_output \
    --n_jobs 2 \
    --device cpu
```

## Performance Notes

### Memory Usage
- K-mer features: ~1GB per dataset
- ESM features: ~2-5GB per dataset
- Public sequences: Minimal (sparse matrices)

### Computation Time
- Training (5-fold CV): ~10-30 minutes (depends on ESM grid search)
- Prediction: ~1-5 minutes per test dataset

### Parallelization
- Uses `n_jobs` for:
  - ExtraTrees training
  - Feature extraction (when computing on-the-fly)
- Set `n_jobs=-1` to use all CPU cores

## Troubleshooting

### Issue: "Metadata not found"
**Solution**: Ensure `metadata.csv` exists in train directory with columns: `filename`, `repertoire_id`, `label_positive`

### Issue: "K-mer features not available"
**Solution**: The code will compute K-mers on-the-fly, but this is slower. Pre-compute using your existing pipeline for better performance.

### Issue: "ESM features not found"
**Solution**: The code will skip ESM grid search and train a 2-model ensemble (K-mer + Public). This is fine but may have slightly lower performance.

### Issue: ImportError for sklearn, scipy, etc.
**Solution**: 
```bash
pip install -r requirements.txt
```

## Key Changes from Original Code

1. **Structure**: Moved from standalone script to class-based predictor
2. **Paths**: Auto-detection instead of hardcoded paths
3. **Interface**: Added standard `fit()` and `predict_proba()` methods
4. **Flexibility**: Handles missing features gracefully
5. **Output**: Formatted according to challenge requirements
6. **Model Selection**: Hybrid strategy that intelligently selects single best model from ensemble

## Model Parameters

All parameters are set based on the implementation:
- **Random Seed**: 42 (for reproducibility)
- **CV Folds**: 5 (for validation and meta-learner training)
- **K-mer Model**: 
  - Penalty: L1 (Lasso)
  - Solver: liblinear
  - Class weight: balanced
  - C values tested: [1.0, 0.2, 0.1, 0.05, 0.03] (auto-selected via grid search)
- **Public Clone Model**: 
  - Fisher's exact test p-value threshold: < 0.05
  - Minimum positive clones: 3
  - No negative clones allowed (cn > 0 filters them out)
- **ExtraTrees Classifier**:
  - n_estimators: 300
  - max_depth: 6
  - min_samples_leaf: 5
  - n_jobs: 1 (for reproducibility)
- **SVM**: 
  - kernel: linear
  - C: 1.0
  - probability: True
- **Meta-Learner**: 
  - Penalty: None (no regularization)
  - Solver: lbfgs
  - max_iter: 1000
  - tolerance: 1e-6

## Contact & Support

This implementation follows the AIRR-ML-25 Challenge template: 
https://github.com/uio-bmi/predict-airr

For questions about the challenge or template, refer to the official Kaggle page:
https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025
