"""
Aggregate ESM2 representations for AIRR datasets.
This script loads per-sequence representations and aggregates them into repertoire-level features.
After aggregation, it automatically removes the raw representation npz files to save space.
"""

import pandas as pd
import numpy as np
import os
import pickle
import glob
from typing import Tuple, List, Optional


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class AggregateConfig:
    """Configuration for representation aggregation."""
    
    # Base paths - should be set by the calling code
    representation_dir = None  # Set via arguments
    aggregate_out_dir = None  # Set via arguments
    train_data_dir = None  # Set via arguments
    test_data_dir = None  # Set via arguments
    sample_submissions_path = None  # Set via arguments
    
    # Pooling methods
    bert_pooling_methods = ["mean", "max"]  # BERT/ESM pooling (excluding cls)
    row_pooling_methods = ["mean", "max", "std", "mean_std"]  # Aggregation across sequences
    
    # Map BERT pooling method to the key in the npz file
    bert_pooling_key_map = {
        "cls": "cls",
        "mean": "mean",
        "max": "max"
    }


# ==============================================================================
# POOLING FUNCTIONS
# ==============================================================================

def apply_pooling(embeddings: np.ndarray, method: str) -> np.ndarray:
    """
    Apply pooling across rows of embeddings.
    
    Args:
        embeddings: Array of shape (n_sequences, embedding_dim)
        method: Pooling method ('mean', 'max', 'min', 'sum', 'std', 'mean_std')
        
    Returns:
        Pooled embedding vector
    """
    if method == "mean":
        return embeddings.mean(axis=0)
    elif method == "max":
        return embeddings.max(axis=0)
    elif method == "min":
        return embeddings.min(axis=0)
    elif method == "sum":
        return embeddings.sum(axis=0)
    elif method == "std":
        return np.std(embeddings, axis=0)
    elif method == "mean_std":
        return np.concatenate([embeddings.mean(axis=0), embeddings.std(axis=0)])
    else:
        raise ValueError(f"Unknown pooling method: {method}")


# ==============================================================================
# TRAIN DATASET AGGREGATION
# ==============================================================================

def load_and_aggregate_train_dataset(
    metadata_path: str,
    features_dir: str,
    embedding_key: str,
    pooling_method: str,
    collect_npz_files: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load and aggregate representations for a training dataset.
    
    Args:
        metadata_path: Path to metadata CSV
        features_dir: Directory containing .npz representation files
        embedding_key: Key to extract from npz file ('cls', 'mean', 'max')
        pooling_method: Row pooling method to aggregate sequences
        collect_npz_files: Whether to collect npz file paths for later removal
        
    Returns:
        Tuple of (X_features, y_labels, npz_files_list, repertoire_ids)
    """
    print(f"  -> Loading metadata from {metadata_path}...")
    try:
        df_metadata = pd.read_csv(metadata_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
    
    # Determine ID column
    if 'repertoire_id' in df_metadata.columns:
        id_col = 'repertoire_id'
    elif 'ID' in df_metadata.columns:
        id_col = 'ID'
        df_metadata['repertoire_id'] = df_metadata['ID']
    elif 'filename' in df_metadata.columns:
        df_metadata['repertoire_id'] = df_metadata['filename'].str.replace('.tsv', '', regex=False)
        id_col = 'repertoire_id'
    else:
        raise ValueError(f"Cannot find ID column in metadata. Columns: {list(df_metadata.columns)}")
    
    df_labels = (
        df_metadata
        .set_index(id_col)
    )
    df_labels['label'] = df_labels['label_positive'].astype(int)
    
    repertoire_features = []
    valid_ids = []
    npz_files_to_remove = []

    print(f"  -> Aggregating ESM2 embeddings (BERT pooling: {embedding_key}, row pooling: {pooling_method})...")

    for rep_id in df_labels.index:
        filepath = os.path.join(features_dir, f"{rep_id}_embeddings.npz")

        if not os.path.exists(filepath):
            print(f"  Warning: File not found: {filepath}")
            continue
        
        try:
            with np.load(filepath) as data:
                embeddings = data[embedding_key]
                aggregated_vector = apply_pooling(embeddings, pooling_method)

                repertoire_features.append(aggregated_vector)
                valid_ids.append(rep_id)
                
                if collect_npz_files:
                    npz_files_to_remove.append(filepath)

        except Exception as e:
            print(f"  Warning: Could not process {filepath}: {e}")

    if len(repertoire_features) == 0:
        raise RuntimeError("No ESM2 feature files loaded. Check your directory and filenames.")

    X = np.stack(repertoire_features)
    y = df_labels.loc[valid_ids, "label"].values

    print(f"  -> Aggregation complete: {X.shape[0]} samples")
    print(f"  -> Feature matrix shape: {X.shape}")
    print(f"  -> Repertoire ID order: {valid_ids[:3]}... (first 3)")

    return X, y, npz_files_to_remove, valid_ids


# ==============================================================================
# TEST DATASET AGGREGATION
# ==============================================================================

def load_test_dataset_embeddings(
    sample_submissions_path: str,
    dataset_num: str,
    features_dir: str,
    embedding_key: str,
    pooling_method: str,
    collect_npz_files: bool = False
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load and aggregate test dataset embeddings.
    
    Args:
        sample_submissions_path: Path to sample_submissions.csv
        dataset_num: Dataset number (e.g., "7_1")
        features_dir: Directory containing .npz representation files
        embedding_key: Key to extract from npz file
        pooling_method: Row pooling method
        collect_npz_files: Whether to collect npz file paths for later removal
        
    Returns:
        Tuple of (X_features, sample_ids, npz_files_list)
    """
    print(f"  -> Loading sample submissions from {sample_submissions_path}...")
    df_submissions = pd.read_csv(sample_submissions_path)
    
    df_test = df_submissions[df_submissions['dataset'] == f'test_dataset_{dataset_num}'].copy()
    print(f"  -> Found {len(df_test)} samples for test_dataset_{dataset_num}")
    
    sample_embeddings = []
    valid_ids = []
    npz_files_to_remove = []
    
    print(f"  -> Loading ESM2 embeddings (BERT pooling: {embedding_key}, row pooling: {pooling_method})...")
    
    for idx, row in df_test.iterrows():
        sample_id = row['ID']
        filepath = os.path.join(features_dir, f"{sample_id}_embeddings.npz")
        
        if not os.path.exists(filepath):
            print(f"  Warning: File not found: {filepath}")
            continue
        
        try:
            with np.load(filepath) as data:
                embeddings = data[embedding_key]
                aggregated_vector = apply_pooling(embeddings, pooling_method)
                sample_embeddings.append(aggregated_vector)
                valid_ids.append(sample_id)
                
                if collect_npz_files:
                    npz_files_to_remove.append(filepath)
        
        except Exception as e:
            print(f"  Warning: Could not process {filepath}: {e}")
    
    if len(sample_embeddings) == 0:
        raise RuntimeError("No ESM2 feature files loaded. Check your directory and filenames.")
    
    X = np.stack(sample_embeddings)
    
    print(f"  -> Loading complete: {X.shape[0]} samples")
    print(f"  -> Feature matrix shape: {X.shape}")
    
    return X, valid_ids, npz_files_to_remove


# ==============================================================================
# MAIN AGGREGATION FUNCTION
# ==============================================================================

def aggregate_for_dataset(
    dataset_num: str,
    dataset_type: str = "train",
    config: Optional[AggregateConfig] = None,
    bert_pooling_methods: Optional[List[str]] = None,
    row_pooling_methods: Optional[List[str]] = None,
    remove_npz: bool = True
):
    """
    Aggregate representations for a specific dataset.
    
    Args:
        dataset_num: Dataset number (e.g., "1", "7_1")
        dataset_type: "train" or "test"
        config: AggregateConfig instance
        bert_pooling_methods: List of BERT pooling methods to process
        row_pooling_methods: List of row pooling methods to process
        remove_npz: Whether to remove npz files after aggregation
    """
    if config is None:
        config = AggregateConfig()
    
    if bert_pooling_methods is None:
        bert_pooling_methods = config.bert_pooling_methods
    
    if row_pooling_methods is None:
        row_pooling_methods = config.row_pooling_methods
    
    dataset_name = f"{dataset_type}_dataset_{dataset_num}"
    features_dir = os.path.join(config.representation_dir, dataset_name)
    
    print(f"\n{'='*70}")
    print(f"=== Aggregating {dataset_name} ===")
    print(f"{'='*70}")
    print(f"Features directory: {features_dir}")
    print(f"BERT Pooling Methods: {bert_pooling_methods}")
    print(f"Row Pooling Methods: {row_pooling_methods}")
    print()
    
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # Collect all npz files to remove (only after ALL aggregations are done)
    all_npz_files_to_remove = set()
    
    # Process each combination of BERT pooling and row pooling
    for bert_method in bert_pooling_methods:
        print(f"\n{'='*70}")
        print(f"Processing BERT Pooling: {bert_method.upper()}")
        print(f"{'='*70}")
        
        embedding_key = config.bert_pooling_key_map[bert_method]
        
        for row_pooling_method in row_pooling_methods:
            print(f"\n  {'-'*66}")
            print(f"  Row Pooling Method: {row_pooling_method.upper()}")
            print(f"  {'-'*66}")
            
            # Construct output path
            output_dir = os.path.join(
                config.aggregate_out_dir,
                f"aggregated_esm2_t6_8M_{bert_method}"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = f"esm2_{dataset_type}_dataset_{dataset_num}_aggregated_{row_pooling_method}.pkl"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"  Output File: {output_path}")
            print()
            
            # Check if already exists
            if os.path.exists(output_path):
                print(f"  ⏭️  Skipping: Output file already exists")
                continue
            
            # Aggregate
            try:
                if dataset_type == "train":
                    metadata_path = os.path.join(
                        config.train_data_dir,
                        dataset_name,
                        "metadata.csv"
                    )
                    X_features, y_labels, npz_files, rep_ids = load_and_aggregate_train_dataset(
                        metadata_path,
                        features_dir,
                        embedding_key,
                        row_pooling_method,
                        collect_npz_files=remove_npz
                    )
                    
                    # Add to set for removal after all aggregations
                    all_npz_files_to_remove.update(npz_files)
                    
                    print(f"    X_features shape: {X_features.shape}")
                    print(f"    y_labels shape: {y_labels.shape}")
                    print(f"    Repertoire order maintained: {rep_ids[:3]}...")
                    
                    output_data = (X_features, y_labels)
                    
                else:  # test
                    X_features, sample_ids = load_test_dataset_embeddings(
                        config.sample_submissions_path,
                        dataset_num,
                        features_dir,
                        embedding_key,
                        row_pooling_method,
                        remove_npz=remove_npz
                    )
                    
                    print(f"    X_features shape: {X_features.shape}")
                    print(f"    Sample IDs: {len(sample_ids)}")
                    
                    output_data = (X_features, sample_ids)
                
                # Save
                print(f"    -> Saving data to {output_path}...")
                with open(output_path, "wb") as f:
                    pickle.dump(output_data, f)
                
                print(f"    -> Saving complete!")
                
                if dataset_type == "train":
                    print(f"    Saved: (X_features, y_labels) with shapes {X_features.shape}, {y_labels.shape}")
                else:
                    print(f"    Saved: (X_features, ids) with shape {X_features.shape}")
                    
            except Exception as e:
                print(f"    ❌ Error during aggregation: {e}")
                raise
    
    # Now remove all npz files after ALL aggregations are complete
    if remove_npz and all_npz_files_to_remove:
        print(f"\n{'='*70}")
        print(f"Cleaning up raw representation files...")
        print(f"{'='*70}")
        print(f"Removing {len(all_npz_files_to_remove)} raw npz files...")
        removed_count = 0
        for npz_file in all_npz_files_to_remove:
            try:
                if os.path.exists(npz_file):
                    os.remove(npz_file)
                    removed_count += 1
            except Exception as e:
                print(f"  Warning: Could not remove {npz_file}: {e}")
        print(f"✅ Successfully removed {removed_count} files")
    
    print(f"\n{'='*70}")
    print(f"All aggregations complete for {dataset_name}!")
    print(f"{'='*70}")


# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

def aggregate_all_datasets(
    config: Optional[AggregateConfig] = None,
    target_dataset_nums: Optional[List[str]] = None,
    remove_npz: bool = True
):
    """
    Aggregate all available datasets.
    
    Args:
        config: AggregateConfig instance
        target_dataset_nums: List of dataset numbers to process (e.g., ["1", "2", "7_1"])
        remove_npz: Whether to remove npz files after aggregation
    """
    if config is None:
        config = AggregateConfig()
    
    # Find all representation directories
    all_rep_dirs = glob.glob(os.path.join(config.representation_dir, "*_dataset_*"))
    
    for rep_dir in all_rep_dirs:
        dataset_full_name = os.path.basename(rep_dir)
        
        # Parse dataset type and number
        if dataset_full_name.startswith("train_dataset_"):
            dataset_type = "train"
            dataset_num = dataset_full_name.replace("train_dataset_", "")
        elif dataset_full_name.startswith("test_dataset_"):
            dataset_type = "test"
            dataset_num = dataset_full_name.replace("test_dataset_", "")
        else:
            print(f"Skipping unknown directory: {dataset_full_name}")
            continue
        
        # Filter by target dataset numbers if specified
        if target_dataset_nums and dataset_num not in target_dataset_nums:
            continue
        
        try:
            aggregate_for_dataset(
                dataset_num=dataset_num,
                dataset_type=dataset_type,
                config=config,
                remove_npz=remove_npz
            )
        except Exception as e:
            print(f"\n❌ Error processing {dataset_full_name}: {e}\n")
            continue


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    config = AggregateConfig()
    
    # Check if specific dataset provided via command line
    if len(sys.argv) >= 3:
        dataset_num = sys.argv[1]
        dataset_type = sys.argv[2]
        
        print(f"Processing single dataset: {dataset_type}_dataset_{dataset_num}")
        aggregate_for_dataset(
            dataset_num=dataset_num,
            dataset_type=dataset_type,
            config=config,
            remove_npz=True
        )
    else:
        # Process all available datasets
        print("Processing all available datasets...")
        aggregate_all_datasets(config=config, remove_npz=True)
    
    print("\n✅ All aggregations complete!")
