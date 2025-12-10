import pandas as pd
import numpy as np
import os
import pickle
import sys

# ==============================================================================
# CONFIGURATION (EDIT THESE)
# ==============================================================================

DATASET_NUM = "8_1"

# Set dataset type: "train" or "test"
DATASET_TYPE = "test"  # Change to "train" or "test"

# Choose which BERT pooling methods to extract: ["cls", "mean", "max"] or any subset
BERT_POOLING_METHODS = ["cls", "mean", "max"]  # Will process all methods in this list

# Choose which row pooling methods to use: ["mean", "max", "sum"] or any subset
ROW_POOLING_METHODS = ["mean", "max", "sum"]  # Will process all methods in this list

# Base paths
if DATASET_TYPE == "train":
    METADATA_PATH = f"/Users/lingyi/Documents/airr-ml/data/train_datasets/train_dataset_{DATASET_NUM}/metadata.csv"
    ESM2_FEATURES_DIR = f"/Users/lingyi/Documents/airr-ml/predict-airr-main/workingfolder/representations/train_dataset_{DATASET_NUM}/"
elif DATASET_TYPE == "test":
    SAMPLE_SUBMISSIONS_PATH = "/Users/lingyi/Documents/airr-ml/data/sample_submissions.csv"
    ESM2_FEATURES_DIR = f"/Users/lingyi/Documents/airr-ml/predict-airr-main/workingfolder/representations/test_dataset_{DATASET_NUM}/"
else:
    raise ValueError(f"DATASET_TYPE must be 'train' or 'test', got: {DATASET_TYPE}")

# Map BERT pooling method to the key in the npz file
BERT_POOLING_KEY_MAP = {
    "cls": "cls",
    "mean": "mean",
    "max": "max"
}

print(f"=== Running ESM2 Aggregation Utility for Dataset {DATASET_NUM} ===")
print(f"Dataset Type: {DATASET_TYPE.upper()}")
print(f"BERT Pooling Methods to process: {BERT_POOLING_METHODS}")
print(f"Row Pooling Methods to process: {ROW_POOLING_METHODS}")
print(f"Input Directory: {ESM2_FEATURES_DIR}")
print()


# ==============================================================================
# POOLING FUNCTIONS
# ==============================================================================

def apply_pooling(embeddings, method):
    """Apply pooling across rows of embeddings."""
    if method == "mean":
        return embeddings.mean(axis=0)
    elif method == "max":
        return embeddings.max(axis=0)
    elif method == "min":
        return embeddings.min(axis=0)
    elif method == "sum":
        return embeddings.sum(axis=0)
    elif method == "mean_std":
        return np.concatenate([embeddings.mean(axis=0),
                               embeddings.std(axis=0)])
    else:
        raise ValueError(f"Unknown pooling method: {method}")


# ==============================================================================
# TRAIN DATASET: AGGREGATION BY REPERTOIRE (with metadata)
# ==============================================================================

def load_and_aggregate_train_dataset(metadata_path, features_dir, embedding_key, pooling_method):
    print(f"  -> Loading metadata from {metadata_path}...")
    try:
        df_metadata = pd.read_csv(metadata_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
    
    df_labels = (
        df_metadata
        .rename(columns={'ID': 'repertoire_id'})
        .set_index('repertoire_id')
    )
    df_labels['label'] = df_labels['label_positive'].astype(int)
    
    repertoire_features = []
    valid_ids = []

    print(f"  -> Aggregating ESM2 embeddings (row pooling: {pooling_method})...")

    for rep_id in df_labels.index:
        filepath = os.path.join(features_dir, f"{rep_id}_embeddings.npz")

        if not os.path.exists(filepath):
            print(f"  Warning: File not found: {filepath}")
            continue
        
        try:
            with np.load(filepath) as data:
                embeddings = data[embedding_key]
                # Apply row-pooling to aggregate the embeddings
                aggregated_vector = apply_pooling(embeddings, pooling_method)

                repertoire_features.append(aggregated_vector)
                valid_ids.append(rep_id)

        except Exception as e:
            print(f"  Warning: Could not process {filepath}: {e}")

    if len(repertoire_features) == 0:
        raise RuntimeError("No ESM2 feature files loaded. Check your directory and filenames.")

    X = np.stack(repertoire_features)
    y = df_labels.loc[valid_ids, "label"].values

    print(f"  -> Aggregation complete: {X.shape[0]} samples")
    print(f"  -> Feature matrix shape: {X.shape}")

    return X, y


# ==============================================================================
# TEST DATASET: SAMPLE-LEVEL EMBEDDINGS (following sample_submissions.csv order)
# ==============================================================================

def load_test_dataset_embeddings(sample_submissions_path, dataset_num, features_dir, embedding_key, pooling_method):
    print(f"  -> Loading sample submissions from {sample_submissions_path}...")
    df_submissions = pd.read_csv(sample_submissions_path)
    
    # Filter for this specific test dataset
    df_test = df_submissions[df_submissions['dataset'] == f'test_dataset_{dataset_num}'].copy()
    print(f"  -> Found {len(df_test)} samples for test_dataset_{dataset_num}")
    
    sample_embeddings = []
    valid_ids = []
    
    print(f"  -> Loading ESM2 embeddings for each sample (row pooling: {pooling_method})...")
    
    for idx, row in df_test.iterrows():
        sample_id = row['ID']
        filepath = os.path.join(features_dir, f"{sample_id}_embeddings.npz")
        
        if not os.path.exists(filepath):
            print(f"  Warning: File not found: {filepath}")
            continue
        
        try:
            with np.load(filepath) as data:
                embeddings = data[embedding_key]
                # Apply row-pooling to test embeddings
                aggregated_vector = apply_pooling(embeddings, pooling_method)
                sample_embeddings.append(aggregated_vector)
                valid_ids.append(sample_id)
        
        except Exception as e:
            print(f"  Warning: Could not process {filepath}: {e}")
    
    if len(sample_embeddings) == 0:
        raise RuntimeError("No ESM2 feature files loaded. Check your directory and filenames.")
    
    # Stack into a matrix for consistency
    X = np.stack(sample_embeddings)
    
    print(f"  -> Loading complete: {X.shape[0]} samples")
    print(f"  -> Feature matrix shape: {X.shape}")
    
    return X, valid_ids


# ==============================================================================
# MAIN LOOP: Process each BERT pooling method and each row pooling method
# ==============================================================================

for bert_method in BERT_POOLING_METHODS:
    print(f"\n{'='*70}")
    print(f"Processing BERT Pooling: {bert_method.upper()}")
    print(f"{'='*70}")
    
    # Get the embedding key for this BERT pooling method
    EMBEDDING_KEY = BERT_POOLING_KEY_MAP[bert_method]
    
    for row_pooling_method in ROW_POOLING_METHODS:
        print(f"\n  {'-'*66}")
        print(f"  Row Pooling Method: {row_pooling_method.upper()}")
        print(f"  {'-'*66}")
        
        # Build output path based on BERT pooling method and row pooling method
        if DATASET_TYPE == "train":
            OUTPUT_FILE_PATH = (
                f"/Users/lingyi/Documents/airr-ml/predict-airr-main/workingfolder/aggregates/aggregated_esm2_t6_8M_{bert_method}/"
                f"esm2_train_dataset_{DATASET_NUM}_aggregated_{row_pooling_method}.pkl"
            )
        else:  # test
            OUTPUT_FILE_PATH = (
                f"/Users/lingyi/Documents/airr-ml/predict-airr-main/workingfolder/aggregates/aggregated_esm2_t6_8M_{bert_method}/"
                f"esm2_test_dataset_{DATASET_NUM}_aggregated_{row_pooling_method}.pkl"
            )
        
        print(f"  BERT Pooling Key: {EMBEDDING_KEY}")
        print(f"  Output File: {OUTPUT_FILE_PATH}")
        print()
        
        # ==============================================================================
        # RUN
        # ==============================================================================
        
        if DATASET_TYPE == "train":
            X_features, y_labels = load_and_aggregate_train_dataset(
                METADATA_PATH,
                ESM2_FEATURES_DIR,
                EMBEDDING_KEY,
                row_pooling_method
            )
            
            print(f"    X_features shape: {X_features.shape}")
            print(f"    y_labels shape: {y_labels.shape}")
            
            output_data = (X_features, y_labels)
            
        else:  # test
            # Test dataset - no labels, with pooling aggregation
            X_features, sample_ids = load_test_dataset_embeddings(
                SAMPLE_SUBMISSIONS_PATH,
                DATASET_NUM,
                ESM2_FEATURES_DIR,
                EMBEDDING_KEY,
                row_pooling_method
            )
            
            print(f"    X_features shape: {X_features.shape}")
            print(f"    Sample IDs: {len(sample_ids)}")
            
            output_data = (X_features, sample_ids)
        
        
        # ==============================================================================
        # SAVE THE OUTPUT
        # ==============================================================================
        
        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        
        if output_dir and not os.path.exists(output_dir):
            print(f"    -> Creating output directory: {output_dir}")
            os.makedirs(output_dir)
        
        print(f"    -> Saving data to {OUTPUT_FILE_PATH}...")
        
        with open(OUTPUT_FILE_PATH, "wb") as f:
            pickle.dump(output_data, f)
        
        print(f"    -> Saving complete!")
        
        if DATASET_TYPE == "train":
            print(f"    Saved: (X_features, y_labels) with shapes {X_features.shape}, {y_labels.shape}")
        else:
            print(f"    Saved: X_features, ids with shape {X_features.shape}")

print(f"\n{'='*70}")
print(f"All BERT pooling methods and row pooling methods processed successfully!")
print(f"{'='*70}")
