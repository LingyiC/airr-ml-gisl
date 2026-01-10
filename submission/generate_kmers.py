"""
Generate k-mer features for AIRR datasets.
This script extracts k3 and k4 k-mer features from junction_aa sequences.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm


# ==========================================
# CONFIGURATION
# ==========================================

class KmerConfig:
    """Configuration for k-mer feature generation."""
    
    # Paths - should be set by the calling code
    train_dir = None  # Set via arguments
    test_dirs = None  # Set via arguments
    kmer_out_dir = None  # Set via arguments
    
    # K-mer settings
    k_values = [3, 4]  # k3 and k4
    
    # Data selection
    target_ids = [1, 2, 3, 4, 5, 7, 8]


# ==========================================
# K-MER EXTRACTION FUNCTIONS
# ==========================================

def extract_kmers(sequence: str, k: int) -> List[str]:
    """
    Extract all k-mers from a sequence.
    
    Args:
        sequence: Protein sequence string
        k: K-mer length
        
    Returns:
        List of k-mers
    """
    if len(sequence) < k:
        return []
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def build_kmer_vocabulary(sequences: List[str], k_values: List[int]) -> Dict[str, int]:
    """
    Build a vocabulary of all k-mers found in sequences.
    
    Args:
        sequences: List of protein sequences
        k_values: List of k values (e.g., [3, 4])
        
    Returns:
        Dictionary mapping k-mer to index
    """
    kmer_set = set()
    
    for seq in sequences:
        if pd.isna(seq) or not isinstance(seq, str):
            continue
        for k in k_values:
            kmers = extract_kmers(seq, k)
            kmer_set.update(kmers)
    
    # Sort for deterministic ordering
    sorted_kmers = sorted(kmer_set)
    kmer_to_idx = {kmer: idx for idx, kmer in enumerate(sorted_kmers)}
    
    return kmer_to_idx


def sequences_to_kmer_vector(sequences: List[str], kmer_to_idx: Dict[str, int], k_values: List[int]) -> np.ndarray:
    """
    Convert a list of sequences to a k-mer count vector.
    
    Args:
        sequences: List of sequences for one repertoire
        kmer_to_idx: Vocabulary mapping k-mer to index
        k_values: List of k values
        
    Returns:
        Dense numpy array of k-mer counts
    """
    kmer_counts = defaultdict(int)
    
    for seq in sequences:
        if pd.isna(seq) or not isinstance(seq, str):
            continue
        for k in k_values:
            kmers = extract_kmers(seq, k)
            for kmer in kmers:
                if kmer in kmer_to_idx:
                    kmer_counts[kmer] += 1
    
    # Create dense vector
    vector = np.zeros(len(kmer_to_idx), dtype=np.int32)
    for kmer, count in kmer_counts.items():
        if kmer in kmer_to_idx:
            vector[kmer_to_idx[kmer]] = count
    
    return vector


# ==========================================
# DATA LOADING
# ==========================================

def load_dataset_sequences(data_dir: str, is_test: bool = False) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Load sequences from a dataset directory.
    
    Args:
        data_dir: Path to dataset directory
        is_test: Whether this is a test dataset
        
    Returns:
        Tuple of (repertoire_ids, sequences_dict)
    """
    metadata_path = os.path.join(data_dir, "metadata.csv")
    repertoire_ids = []
    sequences_dict = {}
    
    if os.path.exists(metadata_path) and not is_test:
        # Training dataset with metadata
        df_metadata = pd.read_csv(metadata_path)
        
        # Get repertoire IDs in order
        if 'repertoire_id' in df_metadata.columns:
            repertoire_ids = df_metadata['repertoire_id'].tolist()
        elif 'ID' in df_metadata.columns:
            repertoire_ids = df_metadata['ID'].tolist()
        elif 'filename' in df_metadata.columns:
            repertoire_ids = df_metadata['filename'].str.replace('.tsv', '', regex=False).tolist()
        else:
            raise ValueError(f"Cannot find ID column in metadata: {list(df_metadata.columns)}")
        
        # Load sequences for each repertoire
        for rep_id in tqdm(repertoire_ids, desc="Loading sequences"):
            # Try to find the TSV file
            tsv_file = os.path.join(data_dir, f"{rep_id}.tsv")
            if not os.path.exists(tsv_file):
                # Try with filename from metadata
                for _, row in df_metadata.iterrows():
                    if 'repertoire_id' in df_metadata.columns and row['repertoire_id'] == rep_id:
                        tsv_file = os.path.join(data_dir, row['filename'])
                        break
                    elif 'ID' in df_metadata.columns and row['ID'] == rep_id:
                        tsv_file = os.path.join(data_dir, row['filename'])
                        break
            
            if os.path.exists(tsv_file):
                try:
                    df = pd.read_csv(tsv_file, sep='\t')
                    if 'junction_aa' in df.columns:
                        sequences_dict[rep_id] = df['junction_aa'].dropna().tolist()
                    else:
                        sequences_dict[rep_id] = []
                except Exception as e:
                    print(f"Warning: Could not load {tsv_file}: {e}")
                    sequences_dict[rep_id] = []
            else:
                print(f"Warning: File not found for {rep_id}")
                sequences_dict[rep_id] = []
    else:
        # Test dataset without metadata - load all TSV files
        tsv_files = sorted(glob.glob(os.path.join(data_dir, "*.tsv")))
        
        for tsv_file in tqdm(tsv_files, desc="Loading sequences"):
            rep_id = os.path.basename(tsv_file).replace('.tsv', '')
            repertoire_ids.append(rep_id)
            
            try:
                df = pd.read_csv(tsv_file, sep='\t')
                if 'junction_aa' in df.columns:
                    sequences_dict[rep_id] = df['junction_aa'].dropna().tolist()
                else:
                    sequences_dict[rep_id] = []
            except Exception as e:
                print(f"Warning: Could not load {tsv_file}: {e}")
                sequences_dict[rep_id] = []
    
    return repertoire_ids, sequences_dict


# ==========================================
# MAIN K-MER GENERATION
# ==========================================

def generate_kmer_features(data_dir: str, dataset_type: str, config: KmerConfig = None):
    """
    Generate k-mer features for a dataset.
    
    Args:
        data_dir: Path to dataset directory
        dataset_type: "train" or "test"
        config: KmerConfig instance
    """
    if config is None:
        config = KmerConfig()
    
    dataset_name = os.path.basename(data_dir)
    
    # Check if features already exist
    output_filename = f"k3_k4_{dataset_name}_features.pkl"
    output_path = os.path.join(config.kmer_out_dir, output_filename)
    
    if os.path.exists(output_path):
        print(f"✅ K-mer features already exist: {output_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Generating K-mer Features for {dataset_name}")
    print(f"{'='*70}")
    
    # Load sequences
    print("\n1. Loading sequences...")
    repertoire_ids, sequences_dict = load_dataset_sequences(
        data_dir, 
        is_test=(dataset_type == "test")
    )
    
    print(f"   Loaded {len(repertoire_ids)} repertoires")
    
    # Build vocabulary from all sequences
    print("\n2. Building k-mer vocabulary...")
    all_sequences = []
    for rep_id in repertoire_ids:
        all_sequences.extend(sequences_dict.get(rep_id, []))
    
    kmer_to_idx = build_kmer_vocabulary(all_sequences, config.k_values)
    print(f"   Vocabulary size: {len(kmer_to_idx)} unique k-mers")
    
    # Generate k-mer features for each repertoire
    print("\n3. Generating k-mer features...")
    kmer_matrix_data = []
    
    for rep_id in tqdm(repertoire_ids, desc="Processing repertoires"):
        seqs = sequences_dict.get(rep_id, [])
        kmer_vector = sequences_to_kmer_vector(seqs, kmer_to_idx, config.k_values)
        kmer_matrix_data.append(kmer_vector)
    
    # Stack into matrix
    X_kmer = np.vstack(kmer_matrix_data)
    
    print(f"   Feature matrix shape: {X_kmer.shape}")
    print(f"   Repertoire IDs: {len(repertoire_ids)}")
    
    # Save features
    print("\n4. Saving k-mer features...")
    os.makedirs(config.kmer_out_dir, exist_ok=True)
    
    # Save as tuple: (X_kmer, repertoire_ids, kmer_to_idx)
    output_data = (X_kmer, repertoire_ids, kmer_to_idx)
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"   ✅ Saved to: {output_path}")
    print(f"   Data format: (X_kmer: {X_kmer.shape}, repertoire_ids: {len(repertoire_ids)}, vocab: {len(kmer_to_idx)} k-mers)")


def ensure_kmer_features_exist(data_dir: str, dataset_type: str, out_dir: str):
    """
    Ensure k-mer features exist for a dataset, generate if missing.
    
    Args:
        data_dir: Path to dataset directory
        dataset_type: "train" or "test"
        out_dir: Output directory
    """
    dataset_name = os.path.basename(data_dir)
    
    config = KmerConfig()
    config.kmer_out_dir = os.path.join(out_dir, "kmer")
    
    output_filename = f"k3_k4_{dataset_name}_features.pkl"
    output_path = os.path.join(config.kmer_out_dir, output_filename)
    
    if os.path.exists(output_path):
        return
    
    print(f"\n⚠️  K-mer features not found for {dataset_name}")
    print(f"   Generating k-mer features...")
    
    generate_kmer_features(data_dir, dataset_type, config)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    import sys
    
    config = KmerConfig()
    os.makedirs(config.kmer_out_dir, exist_ok=True)
    
    # Process training datasets
    print("="*70)
    print("Processing Training Datasets")
    print("="*70)
    
    train_folders = sorted(glob.glob(os.path.join(config.train_dir, "train_dataset_*")))
    
    filtered_train_folders = []
    for f in train_folders:
        try:
            curr_id = int(os.path.basename(f).split('_')[-1])
            if curr_id in config.target_ids:
                filtered_train_folders.append(f)
        except ValueError:
            continue
    
    for train_path in filtered_train_folders:
        generate_kmer_features(train_path, "train", config)
    
    # Process test datasets
    print("\n" + "="*70)
    print("Processing Test Datasets")
    print("="*70)
    
    test_folders = []
    for target_id in config.target_ids:
        suffix = str(target_id)
        dataset_base_name = f"test_dataset_{suffix}"
        
        for td in config.test_dirs:
            potential_sub_dirs = glob.glob(os.path.join(td, f"{dataset_base_name}_*"))
            base_dir = os.path.join(td, dataset_base_name)
            if os.path.isdir(base_dir):
                test_folders.append(base_dir)
            test_folders.extend(potential_sub_dirs)
    
    for test_path in test_folders:
        generate_kmer_features(test_path, "test", config)
    
    print("\n" + "="*70)
    print("✅ K-mer Feature Generation Complete!")
    print("="*70)
