# Reproducibility System for Kaggle Datasets

This system automatically detects if your input data matches any of the known Kaggle competition datasets and uses pre-trained reproduction scripts instead of training new models.

## How It Works

1. **Automatic Detection**: When you run `main.py`, it checks if the repertoire IDs in your input data match any of the 8 known Kaggle datasets
2. **Reproduce Mode**: If a match is found (>50% of IDs match), it automatically uses the corresponding reproduction script
3. **Standard Mode**: If no match is found, it uses the standard predictor pipeline

## Directory Structure

```
submission/
├── main.py                      # Entry point with auto-detection
├── reproduce_checker.py         # Checks for dataset matches
├── predictor.py                 # Standard predictor (used if no match)
├── kaggle_reproduce/
│   ├── metadata/
│   │   ├── Dataset1_metadata.csv
│   │   ├── Dataset2_metadata.csv
│   │   ├── ...
│   │   └── Dataset8_metadata.csv
│   ├── Dataset8_reproduce.py    # Reproduction script for Dataset 8
│   └── ...                      # Other dataset scripts
```

## Usage

### Standard Usage (with auto-detection)

```bash
python3 -m submission.main \
    --train_dir /path/to/train_dataset_8 \
    --test_dirs /path/to/test_dataset_8_1 /path/to/test_dataset_8_2 \
    --out_dir /path/to/output \
    --n_jobs 4
```

If the input matches Dataset 8, it will automatically:
- Detect the match
- Load pre-computed ESM and K-mer features
- Generate predictions using the trained SVM + LogisticRegression ensemble
- Save results to `output/train_dataset_8_test_predictions.tsv`

### Direct Reproduction (independent)

You can also run reproduction scripts directly:

```bash
python3 submission/kaggle_reproduce/Dataset8_reproduce.py \
    --train_dir /path/to/train_dataset_8 \
    --test_dirs /path/to/test_dataset_8_1 /path/to/test_dataset_8_2 \
    --out_dir /path/to/output \
    --n_jobs 4
```

## Configuration

### Pre-computed Feature Paths

The reproduction scripts need access to pre-computed features. By default, they look for:

**ESM Features:**
```
/Users/lingyi/Documents/airr-ml/data/aggregates/aggregated_esm2_t30_150M_max/
└── esm2_train_dataset_8_aggregated_std.pkl
└── esm2_test_dataset_8_1_aggregated_std.pkl
└── esm2_test_dataset_8_2_aggregated_std.pkl
```

**K-mer Features:**
```
/Users/lingyi/Documents/airr-ml/data/kmer/
└── k3_k4_train_dataset_8_features.pkl
└── k3_k4_test_dataset_8_1_features.pkl
└── k3_k4_test_dataset_8_2_features.pkl
```

### Custom Feature Paths

You can override the default paths:

```bash
python3 submission/kaggle_reproduce/Dataset8_reproduce.py \
    --train_dir /path/to/train_dataset_8 \
    --test_dirs /path/to/test_dataset_8_1 \
    --out_dir /path/to/output \
    --esm_base_path /custom/path/to/esm/features \
    --kmer_base_path /custom/path/to/kmer/features \
    --n_jobs 4
```

## Output Format

The reproduction scripts generate predictions in the same format as the standard predictor:

**File:** `train_dataset_8_test_predictions.tsv`

```
ID	dataset	label_positive_probability	junction_aa	v_call	j_call
abc123...	8_1	0.7234	-999.0	-999.0	-999.0
def456...	8_1	0.4521	-999.0	-999.0	-999.0
...
```

**File:** `train_dataset_8_important_sequences.tsv` (dummy file for compatibility)

```
ID	dataset	label_positive_probability	junction_aa	v_call	j_call
N/A	reproduce_mode	-999.0	N/A	N/A	N/A
```

Note: Dataset 8 doesn't identify important sequences in reproduce mode, so a dummy file is created for compatibility with the standard pipeline.

## Adding New Reproduction Scripts

To add a reproduction script for another dataset (e.g., Dataset 7):

1. **Copy the template:**
   ```bash
   cp submission/kaggle_reproduce/Dataset8_reproduce.py \
      submission/kaggle_reproduce/Dataset7_reproduce.py
   ```

2. **Update paths in the script** (if needed):
   - Modify `DEFAULT_BASE_DIR` if your features are in a different location
   - Adjust model parameters if Dataset 7 uses different settings

3. **The script will automatically:**
   - Extract dataset numbers from directory names
   - Load the corresponding pre-computed features
   - Generate predictions using the trained ensemble

## Troubleshooting

### "ESM training features not found"
- Make sure pre-computed features exist for your dataset
- Check the paths in the reproduction script
- Use `--esm_base_path` and `--kmer_base_path` to specify custom paths

### "No matches found"
- This means your input data doesn't match any known Kaggle dataset
- The system will automatically use the standard predictor instead

### "Reproduction script not found"
- The dataset was matched, but no reproduction script exists yet
- The system will fall back to the standard predictor
- You can create the reproduction script by copying `Dataset8_reproduce.py`

## Technical Details

### Dataset Matching Algorithm

1. Extract all repertoire IDs from input `metadata.csv`
2. Load repertoire IDs from all 8 Kaggle metadata files
3. Calculate intersection with each known dataset
4. If >50% of input IDs match a known dataset, use reproduction mode
5. Otherwise, use standard predictor

### Ensemble Method (Dataset 8)

- **Model A**: SVM with RBF kernel (C=0.5) on ESM embeddings
- **Model B**: L1 Logistic Regression (C=0.5) on top 5000 k-mer features
- **Ensemble**: 80% SVM + 20% Logistic Regression
- **Features**: 4-mers only, normalized to frequencies
- **Cross-validation AUC**: 0.7604 (5-fold CV)

## Example Session

```
$ python3 -m submission.main --train_dir data/train_dataset_8 --test_dirs data/test_dataset_8_1 --out_dir output --n_jobs 4

======================================================================
🔍 REPRODUCIBILITY CHECK
======================================================================
Checking if input data matches known Kaggle datasets...
   Loaded 909 repertoire IDs from Dataset 8
   
   Found 909 repertoire IDs in training data
   
   ✅ MATCH FOUND: Dataset 8
      909/909 repertoire IDs match (100.0%)
   
   🎯 Will use reproduction script for Dataset 8
======================================================================

🚀 Launching reproduction script for Dataset 8
   Script: submission/kaggle_reproduce/Dataset8_reproduce.py
   Train: data/train_dataset_8
   Test: data/test_dataset_8_1
   Output: output

======================================================================
🎯 DATASET 8 REPRODUCTION MODE
======================================================================
[1/5] Loading ESM Training Data...
   ESM shape: (909, 640), Labels: (909,)

[2/5] Loading ESM Test Data...
   ESM shape: (101, 640), Test IDs: 101

[3/5] Loading & Aligning K-mer Data...
   K-mer Train shape: (909, 136900)
   K-mer Test shape: (101, 136900)

[4/5] Selecting Top Variable Features...
   Selected top 5000 features based on training variance

[5/5] Training & Predicting...
   Training SVM on ESM features...
   Training Logistic Regression on K-mer features...
   ✅ Generated 101 predictions for 8_1

======================================================================
✅ REPRODUCTION COMPLETE
======================================================================
Total predictions: 101
Saved to: output/train_dataset_8_test_predictions.tsv
======================================================================

✅ Reproduction script completed successfully!
```
