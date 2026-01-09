#!/usr/bin/env python3
"""
Dataset 8 Reproduction Script
This script reproduces the exact predictions for Dataset 8 using pre-trained ESM and K-mer features.

Usage:
    python Dataset8_reproduce.py --train_dir /path/to/train --test_dirs /path/to/test1 /path/to/test2 --out_dir /path/to/output --n_jobs 4
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_VERSION = "esm2_t30_150M"
BERT_POOLING = "max"
ROW_POOLING = "std"  
SEED = 42
TOP_K_KMERS = 5000 

# Default paths for pre-computed features (MUST BE CONFIGURED)
DEFAULT_BASE_DIR = "/Users/lingyi/Documents/airr-ml"
DEFAULT_ESM_BASE_PATH = os.path.join(DEFAULT_BASE_DIR, "data/aggregates")
DEFAULT_KMER_BASE_PATH = os.path.join(DEFAULT_BASE_DIR, "data/kmer")

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_esm(path):
    """Load ESM embeddings from pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[0], data[1]


def load_and_align_kmers(train_path, test_path):
    """
    Loads both Train and Test K-mers.
    Ensures Test columns EXACTLY match Train columns (order and presence).
    """
    print(f"   Loading Train K-mers: {train_path}")
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    df_train = train_data[0]
    
    print(f"   Loading Test K-mers:  {test_path}")
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    df_test = test_data[0]

    # 1. Identify 4-mer columns in TRAINING data
    train_k4_cols = [c for c in df_train.columns if c.startswith('k4__')]
    
    if not train_k4_cols:
        print("⚠️ No 4-mers found in Train! Using all columns.")
        train_k4_cols = df_train.columns.tolist()
    
    # 2. Select these columns from Train
    X_train = df_train[train_k4_cols].values.astype(np.float32)
    
    # 3. ALIGN TEST TO TRAIN
    print("   Aligning Test columns to match Train columns...")
    df_test_aligned = df_test.reindex(columns=train_k4_cols, fill_value=0)
    X_test = df_test_aligned.values.astype(np.float32)

    # 4. Normalize (Frequencies)
    print("   Normalizing counts to frequencies...")
    # Train
    row_sums_tr = X_train.sum(axis=1, keepdims=True)
    row_sums_tr[row_sums_tr == 0] = 1.0
    X_train = X_train / row_sums_tr
    
    # Test
    row_sums_te = X_test.sum(axis=1, keepdims=True)
    row_sums_te[row_sums_te == 0] = 1.0
    X_test = X_test / row_sums_te
    
    return X_train, X_test


# ==============================================================================
# MAIN PREDICTION FUNCTION
# ==============================================================================

def run_reproduce_prediction(train_dir: str,
                            test_dirs: List[str],
                            out_dir: str,
                            n_jobs: int = 1,
                            esm_base_path: str = None,
                            kmer_base_path: str = None) -> pd.DataFrame:
    """
    Run reproduction prediction using pre-computed features.
    
    Args:
        train_dir: Training data directory (used to extract dataset number)
        test_dirs: List of test data directories
        out_dir: Output directory for predictions
        n_jobs: Number of jobs (not used in this version but kept for compatibility)
        esm_base_path: Base path to ESM features
        kmer_base_path: Base path to K-mer features
        
    Returns:
        DataFrame with predictions
    """
    
    if esm_base_path is None:
        esm_base_path = DEFAULT_ESM_BASE_PATH
    if kmer_base_path is None:
        kmer_base_path = DEFAULT_KMER_BASE_PATH
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Extract dataset numbers from directory names
    train_dataset_num = os.path.basename(train_dir).replace('train_dataset_', '')
    
    print("\n" + "="*70)
    print(f"🎯 DATASET {train_dataset_num} REPRODUCTION MODE")
    print("="*70)
    print(f"Train Dataset: {train_dataset_num}")
    print(f"ESM Base Path: {esm_base_path}")
    print(f"K-mer Base Path: {kmer_base_path}")
    print(f"Output Directory: {out_dir}")
    
    # Construct file paths for pre-computed features
    ESM_FOLDER = f"aggregated_{MODEL_VERSION}_{BERT_POOLING}"
    ESM_TRAIN_FILE = os.path.join(esm_base_path, ESM_FOLDER, 
                                   f"esm2_train_dataset_{train_dataset_num}_aggregated_{ROW_POOLING}.pkl")
    KMER_TRAIN_FILE = os.path.join(kmer_base_path, 
                                    f"k3_k4_train_dataset_{train_dataset_num}_features.pkl")
    
    # Check if pre-computed features exist
    if not os.path.exists(ESM_TRAIN_FILE):
        raise FileNotFoundError(f"ESM training features not found: {ESM_TRAIN_FILE}")
    if not os.path.exists(KMER_TRAIN_FILE):
        raise FileNotFoundError(f"K-mer training features not found: {KMER_TRAIN_FILE}")
    
    # Load training data
    print("\n[1/5] Loading ESM Training Data...")
    X_esm_train, y_train = load_esm(ESM_TRAIN_FILE)
    print(f"   ESM shape: {X_esm_train.shape}, Labels: {y_train.shape}")
    
    # Process each test directory
    all_predictions = []
    
    for test_dir in test_dirs:
        test_dataset_num = os.path.basename(test_dir).replace('test_dataset_', '')
        print(f"\n{'='*70}")
        print(f"Processing Test Dataset: {test_dataset_num}")
        print(f"{'='*70}")
        
        # Construct test file paths
        ESM_TEST_FILE = os.path.join(esm_base_path, ESM_FOLDER, 
                                      f"esm2_test_dataset_{test_dataset_num}_aggregated_{ROW_POOLING}.pkl")
        KMER_TEST_FILE = os.path.join(kmer_base_path, 
                                       f"k3_k4_test_dataset_{test_dataset_num}_features.pkl")
        
        if not os.path.exists(ESM_TEST_FILE):
            print(f"   ⚠️ ESM test features not found: {ESM_TEST_FILE}")
            print(f"   Skipping {test_dataset_num}")
            continue
        if not os.path.exists(KMER_TEST_FILE):
            print(f"   ⚠️ K-mer test features not found: {KMER_TEST_FILE}")
            print(f"   Skipping {test_dataset_num}")
            continue
        
        # Load test data
        print("\n[2/5] Loading ESM Test Data...")
        X_esm_test, test_ids = load_esm(ESM_TEST_FILE)
        print(f"   ESM shape: {X_esm_test.shape}, Test IDs: {len(test_ids)}")
        
        # Load & Align K-mers
        print("\n[3/5] Loading & Aligning K-mer Data...")
        X_kmer_train, X_kmer_test = load_and_align_kmers(KMER_TRAIN_FILE, KMER_TEST_FILE)
        print(f"   K-mer Train shape: {X_kmer_train.shape}")
        print(f"   K-mer Test shape: {X_kmer_test.shape}")
        
        # Feature Selection (Using TRAIN Variance only)
        print("\n[4/5] Selecting Top Variable Features...")
        variances = np.var(X_kmer_train, axis=0)
        
        if TOP_K_KMERS < X_kmer_train.shape[1]:
            top_indices = np.argsort(variances)[-TOP_K_KMERS:]
            X_kmer_train_selected = X_kmer_train[:, top_indices]
            X_kmer_test_selected = X_kmer_test[:, top_indices]
            print(f"   Selected top {TOP_K_KMERS} features based on training variance")
        else:
            X_kmer_train_selected = X_kmer_train
            X_kmer_test_selected = X_kmer_test
            print("   Keeping all features")
        
        # Scale
        scaler = StandardScaler()
        X_kmer_train_scaled = scaler.fit_transform(X_kmer_train_selected)
        X_kmer_test_scaled = scaler.transform(X_kmer_test_selected)
        
        # Train models
        print("\n[5/5] Training & Predicting...")
        y_train_int = y_train.astype(int)
        
        # Model A: SVM on ESM features
        print("   Training SVM on ESM features...")
        model_A = SVC(kernel="rbf", C=0.5, gamma="scale", probability=True, random_state=SEED)
        model_A.fit(X_esm_train, y_train_int)
        probs_A = model_A.predict_proba(X_esm_test)[:, 1]
        
        # Model B: Logistic Regression on K-mer features
        print("   Training Logistic Regression on K-mer features...")
        model_B = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000, random_state=SEED)
        model_B.fit(X_kmer_train_scaled, y_train_int)
        probs_B = model_B.predict_proba(X_kmer_test_scaled)[:, 1]
        
        # Ensemble (80% ESM, 20% K-mer)
        final_probs = (probs_A * 0.8) + (probs_B * 0.2)
        
        # Create predictions DataFrame matching predictor.py format
        test_predictions = pd.DataFrame({
            'ID': test_ids,
            'dataset': [test_dataset_num] * len(test_ids),
            'label_positive_probability': final_probs
        })
        
        # Add compatibility columns (same as predictor.py)
        test_predictions['junction_aa'] = -999.0
        test_predictions['v_call'] = -999.0
        test_predictions['j_call'] = -999.0
        
        # Reorder columns to match predictor.py output
        test_predictions = test_predictions[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
        
        all_predictions.append(test_predictions)
        print(f"   ✅ Generated {len(test_predictions)} predictions for {test_dataset_num}")
    
    # Combine all predictions
    if not all_predictions:
        raise ValueError("No predictions were generated. Check if pre-computed features exist.")
    
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Save predictions (matching predictor.py format)
    output_filename = f"{os.path.basename(train_dir)}_test_predictions.tsv"
    output_path = os.path.join(out_dir, output_filename)
    combined_predictions.to_csv(output_path, sep='\t', index=False)
    
    # Create dummy important_sequences file (Dataset 8 doesn't have this)
    important_sequences_filename = f"{os.path.basename(train_dir)}_important_sequences.tsv"
    important_sequences_path = os.path.join(out_dir, important_sequences_filename)
    
    # Create empty DataFrame with same structure as predictor.py would output
    dummy_sequences = pd.DataFrame({
        'ID': ['N/A'],
        'dataset': ['reproduce_mode'],
        'label_positive_probability': [-999.0],
        'junction_aa': ['N/A'],
        'v_call': ['N/A'],
        'j_call': ['N/A']
    })
    dummy_sequences.to_csv(important_sequences_path, sep='\t', index=False)
    
    print("\n" + "="*70)
    print(f"✅ REPRODUCTION COMPLETE")
    print("="*70)
    print(f"Total predictions: {len(combined_predictions)}")
    print(f"Predictions saved to: {output_path}")
    print(f"Important sequences (dummy) saved to: {important_sequences_path}")
    print("="*70)
    
    return combined_predictions


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dataset 8 Reproduction Script - Generate predictions using pre-computed features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python Dataset8_reproduce.py \\
        --train_dir /path/to/train_dataset_8 \\
        --test_dirs /path/to/test_dataset_8_1 /path/to/test_dataset_8_2 \\
        --out_dir /path/to/output \\
        --n_jobs 4
        
Configuration:
    Pre-computed feature paths can be set with environment variables:
        ESM_BASE_PATH: Base path to ESM features (default: /Users/lingyi/Documents/airr-ml/data/aggregates)
        KMER_BASE_PATH: Base path to K-mer features (default: /Users/lingyi/Documents/airr-ml/data/kmer)
    """
    )
    
    parser.add_argument("--train_dir", required=True,
                       help="Path to training data directory")
    parser.add_argument("--test_dirs", required=True, nargs="+",
                       help="Path(s) to test data director(ies)")
    parser.add_argument("--out_dir", required=True,
                       help="Path to output directory")
    parser.add_argument("--n_jobs", type=int, default=1,
                       help="Number of CPU cores to use (for compatibility, not used in this version)")
    parser.add_argument("--esm_base_path", type=str, default=None,
                       help=f"Base path to ESM features (default: {DEFAULT_ESM_BASE_PATH})")
    parser.add_argument("--kmer_base_path", type=str, default=None,
                       help=f"Base path to K-mer features (default: {DEFAULT_KMER_BASE_PATH})")
    
    args = parser.parse_args()
    
    try:
        run_reproduce_prediction(
            train_dir=args.train_dir,
            test_dirs=args.test_dirs,
            out_dir=args.out_dir,
            n_jobs=args.n_jobs,
            esm_base_path=args.esm_base_path,
            kmer_base_path=args.kmer_base_path
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
