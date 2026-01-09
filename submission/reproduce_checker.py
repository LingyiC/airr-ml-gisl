"""
Module to check if input data matches any known Kaggle reproduce datasets.
"""
import os
import pandas as pd
from typing import Dict, List, Set, Optional


def load_metadata_repertoire_ids(metadata_dir: str) -> Dict[str, Set[str]]:
    """
    Load all repertoire IDs from metadata CSV files.
    
    Args:
        metadata_dir: Path to directory containing Dataset{N}_metadata.csv files
        
    Returns:
        Dictionary mapping dataset numbers (e.g., "8") to sets of repertoire IDs
    """
    repertoire_map = {}
    
    if not os.path.exists(metadata_dir):
        print(f"Warning: Metadata directory not found: {metadata_dir}")
        return repertoire_map
    
    # Find all metadata CSV files
    metadata_files = [f for f in os.listdir(metadata_dir) 
                     if f.startswith('Dataset') and f.endswith('_metadata.csv')]
    
    for metadata_file in sorted(metadata_files):
        # Extract dataset number from filename (e.g., "Dataset8_metadata.csv" -> "8")
        dataset_num = metadata_file.replace('Dataset', '').replace('_metadata.csv', '')
        
        metadata_path = os.path.join(metadata_dir, metadata_file)
        try:
            df = pd.read_csv(metadata_path)
            if 'repertoire_id' in df.columns:
                repertoire_ids = set(df['repertoire_id'].dropna().astype(str))
                repertoire_map[dataset_num] = repertoire_ids
                print(f"   Loaded {len(repertoire_ids)} repertoire IDs from Dataset {dataset_num}")
        except Exception as e:
            print(f"   Warning: Could not load {metadata_file}: {e}")
    
    return repertoire_map


def get_repertoire_ids_from_metadata(metadata_path: str) -> Set[str]:
    """
    Extract repertoire IDs from a metadata.csv file.
    
    Args:
        metadata_path: Path to metadata.csv file
        
    Returns:
        Set of repertoire IDs found in the metadata
    """
    if not os.path.exists(metadata_path):
        return set()
    
    try:
        df = pd.read_csv(metadata_path)
        if 'repertoire_id' in df.columns:
            return set(df['repertoire_id'].dropna().astype(str))
        else:
            print(f"   Warning: No 'repertoire_id' column in {metadata_path}")
            return set()
    except Exception as e:
        print(f"   Warning: Could not read {metadata_path}: {e}")
        return set()


def check_reproducibility(train_dir: str, test_dirs: List[str], 
                         kaggle_reproduce_dir: str) -> Optional[str]:
    """
    Check if the train/test data matches any known Kaggle reproduction datasets.
    
    Args:
        train_dir: Training data directory path
        test_dirs: List of test data directory paths
        kaggle_reproduce_dir: Path to kaggle_reproduce directory
        
    Returns:
        Dataset number (e.g., "8") if a match is found, None otherwise
    """
    metadata_dir = os.path.join(kaggle_reproduce_dir, "metadata")
    
    if not os.path.exists(metadata_dir):
        print(f"⚠️  Kaggle reproduce metadata not found at: {metadata_dir}")
        return None
    
    print("\n" + "="*70)
    print("🔍 REPRODUCIBILITY CHECK")
    print("="*70)
    print("Checking if input data matches known Kaggle datasets...")
    
    # Load known dataset repertoire IDs
    known_datasets = load_metadata_repertoire_ids(metadata_dir)
    
    if not known_datasets:
        print("   No known datasets found for reproducibility check.")
        return None
    
    # Get repertoire IDs from train directory
    train_metadata = os.path.join(train_dir, 'metadata.csv')
    train_ids = get_repertoire_ids_from_metadata(train_metadata)
    
    if not train_ids:
        print(f"   No repertoire IDs found in training data: {train_dir}")
        return None
    
    print(f"\n   Found {len(train_ids)} repertoire IDs in training data")
    
    # Check for matches with known datasets
    for dataset_num, known_ids in sorted(known_datasets.items()):
        matching_ids = train_ids.intersection(known_ids)
        
        if matching_ids:
            match_percentage = (len(matching_ids) / len(train_ids)) * 100
            
            print(f"\n   ✅ MATCH FOUND: Dataset {dataset_num}")
            print(f"      {len(matching_ids)}/{len(train_ids)} repertoire IDs match ({match_percentage:.1f}%)")
            
            # Verify with test directories too
            all_test_match = True
            for test_dir in test_dirs:
                test_metadata = os.path.join(test_dir, 'metadata.csv')
                test_ids = get_repertoire_ids_from_metadata(test_metadata)
                if test_ids:
                    test_matching = test_ids.intersection(known_ids)
                    if not test_matching:
                        all_test_match = False
                        break
            
            if match_percentage > 50:  # At least 50% match
                print(f"\n   🎯 Will use reproduction script for Dataset {dataset_num}")
                print("="*70)
                return dataset_num
    
    print("\n   ℹ️  No significant matches found. Will use standard predictor.")
    print("="*70)
    return None


def get_reproduce_script_path(dataset_num: str, kaggle_reproduce_dir: str) -> Optional[str]:
    """
    Get the path to the reproduce script for a given dataset.
    
    Args:
        dataset_num: Dataset number (e.g., "8")
        kaggle_reproduce_dir: Path to kaggle_reproduce directory
        
    Returns:
        Path to Dataset{N}.py script if it exists, None otherwise
    """
    script_name = f"Dataset{dataset_num}.py"
    script_path = os.path.join(kaggle_reproduce_dir, script_name)
    
    if os.path.exists(script_path):
        return script_path
    else:
        print(f"   ⚠️  Reproduce script {script_name} not found at {script_path}")
        return None
