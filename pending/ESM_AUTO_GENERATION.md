# ESM2 Feature Auto-Generation

The predictor now supports automatic generation of ESM2 features when they are not found.

## Overview

The ensemble predictor can automatically:
1. **Detect missing ESM features** during training/prediction
2. **Generate ESM2 embeddings** using `save_representation.py`
3. **Aggregate embeddings** with different pooling strategies using `aggregate_allpooling.py`
4. **Continue with full ensemble** (K-mer + Public + ESM)

## Current Setup (Your Case)

Since you already have ESM features generated, the auto-generation will **NOT** run by default:

```python
# Your current setup - ESM features exist
predictor = ImmuneStatePredictor(
    n_jobs=4,
    device='cpu',
    base_dir='/Users/lingyi/Documents/airr-ml/predict-airr-main/'
)
# Will find existing ESM features in workingfolder/aggregates/
# No generation needed ✓
```

## Enabling Auto-Generation (Future Use)

If you need to generate ESM features for new datasets:

```python
predictor = ImmuneStatePredictor(
    n_jobs=4,
    device='cpu',
    base_dir='/Users/lingyi/Documents/airr-ml/predict-airr-main/',
    auto_generate_esm=True,  # Enable auto-generation
    esm_model_name='facebook/esm2_t6_8M_UR50D',  # Optional: specify model
    esm_batch_size=128  # Optional: adjust batch size
)
```

## How It Works

### 1. Detection Phase
During `fit()` or `predict_proba()`:
```
2. Auto-detecting features for dataset: train_dataset_8
  Found K-mer features: .../workingfolder/kmer/...
  Found ESM features in: .../workingfolder/aggregates/
```

If ESM features are NOT found:
```
  Warning: No ESM features found for train_dataset_8
  Auto-generation enabled: Will generate ESM features for train_dataset_8
```

### 2. Generation Phase (if enabled)
```
  ========================================================
  Generating ESM2 Features for train_dataset_8
  ========================================================
  Step 1: Extracting ESM2 representations...
    - Runs save_representation.py
    - Extracts embeddings with all pooling (cls, mean, max)
    - Saves to workingfolder/representations/
  ✓ ESM2 extraction complete

  Step 2: Aggregating embeddings with pooling strategies...
    - Runs aggregate_allpooling.py
    - Creates 6 variants (2 BERT × 3 row pooling)
    - Saves to workingfolder/aggregates/
  ✓ Aggregation complete
  ✓ ESM features successfully generated!
```

### 3. Fallback (if generation fails)
```
  Error generating ESM representations: [error details]
  Continuing without ESM features...
  
  Training with 2-model ensemble (K-mer + Public Clones only)
```

## File Structure

### Before Auto-Generation
```
predict-airr-main/
├── workingfolder/
│   ├── kmer/
│   │   └── k3_k4_train_dataset_8_features.pkl
│   └── aggregates/  # EMPTY or missing
```

### After Auto-Generation
```
predict-airr-main/
├── workingfolder/
│   ├── kmer/
│   │   └── k3_k4_train_dataset_8_features.pkl
│   ├── representations/
│   │   └── train_dataset_8/
│   │       ├── rep001_embeddings.npz
│   │       ├── rep002_embeddings.npz
│   │       └── ...
│   └── aggregates/
│       ├── aggregated_esm2_t6_8M_max/
│       │   ├── esm2_train_dataset_8_aggregated_sum.pkl
│       │   ├── esm2_train_dataset_8_aggregated_mean.pkl
│       │   └── esm2_train_dataset_8_aggregated_max.pkl
│       └── aggregated_esm2_t6_8M_mean/
│           ├── esm2_train_dataset_8_aggregated_sum.pkl
│           ├── esm2_train_dataset_8_aggregated_mean.pkl
│           └── esm2_train_dataset_8_aggregated_max.pkl
```

## Requirements

For auto-generation to work, you need:

1. **Scripts present**:
   - `save_representation.py` in predict-airr-main/
   - `aggregate_allpooling.py` in predict-airr-main/

2. **Dependencies installed**:
   ```bash
   pip install transformers torch
   ```

3. **Hardware**:
   - CPU: Will work but slow
   - GPU (CUDA): Recommended for faster embedding extraction

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto_generate_esm` | `False` | Enable/disable auto-generation |
| `esm_model_name` | `'facebook/esm2_t6_8M_UR50D'` | HuggingFace model name |
| `esm_batch_size` | `128` | Batch size for embedding extraction |
| `device` | `'cpu'` | Device for computation |

## Performance Notes

### Generation Time
- **Small dataset** (100 repertoires): ~5-10 minutes (CPU)
- **Medium dataset** (500 repertoires): ~20-30 minutes (CPU)
- **Large dataset** (1000+ repertoires): ~1-2 hours (CPU)
- **GPU**: 3-5x faster than CPU

### Storage
- Raw embeddings: ~50-100 MB per 1000 repertoires
- Aggregated features: ~5-10 MB per dataset

## Troubleshooting

### Issue: "save_representation.py not found"
**Solution**: Ensure the script exists in the base_dir:
```bash
ls /Users/lingyi/Documents/airr-ml/predict-airr-main/save_representation.py
```

### Issue: "transformers library not found"
**Solution**: Install dependencies:
```bash
pip install transformers torch
```

### Issue: Generation takes too long
**Solution**: 
1. Use GPU if available: `device='cuda'`
2. Increase batch size: `esm_batch_size=256`
3. Use smaller model: `esm_model_name='facebook/esm2_t6_8M_UR50D'`

### Issue: Out of memory during generation
**Solution**:
1. Reduce batch size: `esm_batch_size=32`
2. Use CPU: `device='cpu'`
3. Process datasets one at a time

## When to Use Auto-Generation

✅ **Use when**:
- Working with new/unseen datasets
- ESM features don't exist yet
- Want fully automated pipeline
- Experimenting with different datasets

❌ **Don't use when**:
- Features already exist (like your current case)
- Want faster training/testing
- Limited computational resources
- Debugging other parts of the pipeline

## Your Current Status

🎯 **You DON'T need auto-generation** because:
- ✓ ESM features already exist in `workingfolder/aggregates/`
- ✓ K-mer features already exist in `workingfolder/kmer/`
- ✓ Code will automatically find and use existing features

The auto-generation feature is there for **future datasets** or when you want to regenerate features with different settings.
