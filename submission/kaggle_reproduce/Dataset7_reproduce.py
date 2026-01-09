#!/usr/bin/env python3
"""
Dataset 7 Reproduction Script
This script reproduces exact predictions for Dataset 7 by retraining the ensemble model.

Usage:
    PYTHONHASHSEED=0 python Dataset7_reproduce.py --train_dir /path/to/train --test_dirs /path/to/test1 /path/to/test2 --out_dir /path/to/output --n_jobs 4

Note: PYTHONHASHSEED=0 must be set as an environment variable before running the script for reproducibility.
"""

import os
import sys

# Add parent directory to path to allow importing from submission module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
import pickle
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List
from pathlib import Path
from tqdm import tqdm
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEED = 42
N_FOLDS = 5

# Default paths for pre-computed features
DEFAULT_BASE_DIR = "/Users/lingyi/Documents/airr-ml"
DEFAULT_OUT_DIR = "/Users/lingyi/Documents/airr-ml/workingFolder/output"


# ==============================================================================
# HELPER CLASSES AND FUNCTIONS
# ==============================================================================

class PublicCloneModel:
    """Fisher's exact test based public clone model."""
    def __init__(self, p_val=0.05, min_pos=3):
        self.p_val = p_val
        self.min_pos = min_pos
        self.selected = []
        self.clf = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', 
                                      class_weight='balanced', random_state=SEED)
        
    def fit(self, seqs_dict, y, train_ids):
        pos_ids = [pid for pid, lbl in zip(train_ids, y) if lbl == 1]
        neg_ids = [pid for pid, lbl in zip(train_ids, y) if lbl == 0]
        
        counts = {}
        for pid in pos_ids:
            for seq in seqs_dict[pid]:
                if seq not in counts: counts[seq] = [0, 0]
                counts[seq][0] += 1
        for pid in neg_ids:
            for seq in seqs_dict[pid]:
                if seq in counts: counts[seq][1] += 1
                
        n_pos, n_neg = len(pos_ids), len(neg_ids)
        self.selected = []
        
        # # # Sort counts to ensure deterministic feature order
        # sorted_counts = sorted(counts.items())
        
        for seq, (cp, cn) in counts.items():
            if cp < self.min_pos: continue
            if cn > 0: continue
            _, p = stats.fisher_exact([[cp, cn], [n_pos-cp, n_neg-cn]], alternative='greater')
            if p < self.p_val: self.selected.append(seq)
            
        if self.selected:
            X = self._make_matrix(seqs_dict, train_ids)
            self.clf.fit(X, y)
        return self

    def predict_proba(self, seqs_dict, test_ids):
        if not self.selected: return np.full(len(test_ids), 0.5)
        X = self._make_matrix(seqs_dict, test_ids)
        return self.clf.predict_proba(X)[:, 1]

    def _make_matrix(self, seqs_dict, ids):
        mapper = {s: i for i, s in enumerate(self.selected)}
        X = sparse.lil_matrix((len(ids), len(mapper)), dtype=np.int8)
        for r, pid in enumerate(ids):
            for seq in seqs_dict[pid]:
                if seq in mapper: X[r, mapper[seq]] = 1
        return X.tocsr()


def get_repertoire_ids(data_dir_path):
    """Get list of repertoire IDs from TSV files in directory."""
    files = sorted([f.replace('.tsv', '') for f in os.listdir(data_dir_path) if f.endswith('.tsv')])
    return files


def load_metadata_and_labels(train_dir_path):
    """Loads ground truth labels from metadata."""
    meta_path = os.path.join(train_dir_path, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}")
    
    df_meta = pd.read_csv(meta_path)
    if 'filename' in df_meta.columns:
        df_meta['ID'] = df_meta['filename'].str.replace('.tsv', '', regex=False)
    
    return df_meta.set_index('ID')['label_positive'].astype(int)


def load_public_sequences(data_dir_path, ids):
    """Load public clone sequences."""
    seqs_dict = {}
    for pid in tqdm(ids, desc="   Loading sequences", leave=False):
        try:
            fpath = os.path.join(data_dir_path, f"{pid}.tsv")
            df = pd.read_csv(fpath, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
            seqs_dict[pid] = set(zip(df['junction_aa'], df['v_call'], df['j_call']))
        except:
            seqs_dict[pid] = set()
    return seqs_dict

def load_kmer_features(kmer_path, ids):
    """Load and align K-mer features."""
    if kmer_path is None or not os.path.exists(kmer_path):
        print("   Warning: K-mer features not available")
        return None, None
    
    with open(kmer_path, "rb") as f:
        kmer_raw = pickle.load(f)
    
    result_df = None
    
    # Handle tuple format (features, ids)
    if isinstance(kmer_raw, tuple) and len(kmer_raw) >= 2:
        features = kmer_raw[0]
        stored_ids = kmer_raw[1]
        
        # Convert stored_ids to list if needed
        if isinstance(stored_ids, pd.Index):
            stored_ids = stored_ids.tolist()
        elif isinstance(stored_ids, np.ndarray):
            stored_ids = stored_ids.tolist()
        
        # If features is a DataFrame
        if isinstance(features, pd.DataFrame):
            result_df = features.loc[ids]
        # If features is array-like, create DataFrame with IDs
        elif isinstance(features, (np.ndarray, list)):
            df = pd.DataFrame(features, index=stored_ids)
            result_df = df.loc[ids]
        else:
            raise TypeError(f"Unexpected feature type in tuple: {type(features)}")
    
    # Single element tuple
    elif isinstance(kmer_raw, tuple) and len(kmer_raw) == 1:
        kmer_raw = kmer_raw[0]
        if isinstance(kmer_raw, pd.DataFrame):
            result_df = kmer_raw.loc[ids]
        elif isinstance(kmer_raw, np.ndarray):
            result_df = pd.DataFrame(kmer_raw, index=ids)
        else:
            raise TypeError(f"Unexpected data type in single-element tuple: {type(kmer_raw)}")
    
    # Direct DataFrame
    elif isinstance(kmer_raw, pd.DataFrame):
        result_df = kmer_raw.loc[ids]
    
    # Direct array (assume same order as ids)
    elif isinstance(kmer_raw, np.ndarray):
        result_df = pd.DataFrame(kmer_raw, index=ids)
    
    # List - check if it's a [features, ids] format (like tuple)
    elif isinstance(kmer_raw, list):
        if len(kmer_raw) == 2 and hasattr(kmer_raw[0], '__len__'):
            # Treat like tuple format
            features = kmer_raw[0]
            stored_ids = kmer_raw[1]
            
            # Convert stored_ids to list if needed
            if isinstance(stored_ids, pd.Index):
                stored_ids = stored_ids.tolist()
            elif isinstance(stored_ids, np.ndarray):
                stored_ids = stored_ids.tolist()
            
            # Handle features based on type
            if isinstance(features, pd.DataFrame):
                result_df = features.loc[ids]
            elif isinstance(features, (np.ndarray, list)):
                df = pd.DataFrame(features, index=stored_ids)
                result_df = df.loc[ids]
            else:
                raise TypeError(f"Unexpected feature type in list: {type(features)}")
        else:
            # Try converting to array
            try:
                arr = np.array(kmer_raw)
                result_df = pd.DataFrame(arr, index=ids)
            except ValueError:
                # If conversion fails, features might be variable length
                # Create DataFrame assuming ids match order
                result_df = pd.DataFrame(kmer_raw, index=ids)
    
    else:
        raise TypeError(f"Unexpected K-mer data type: {type(kmer_raw)}")
    
    feature_names = result_df.columns.tolist() if hasattr(result_df, 'columns') else list(range(result_df.shape[1]))
    return result_df.values, feature_names


def load_esm_data(path, master_labels):
    """Loads ESM pickle and aligns it to master labels."""
    if not os.path.exists(path):
        return None, None, None
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, (tuple, list)) and len(data) >= 2:
        try:
            ids = data[2] if len(data) > 2 else master_labels.index
            df = pd.DataFrame(data[0], index=ids)
        except:
            df = pd.DataFrame(data[0], index=master_labels.index)
    else:
        df = pd.DataFrame(data, index=master_labels.index)
    
    common_ids = master_labels.index.intersection(df.index)
    X = df.loc[common_ids].values
    y = master_labels.loc[common_ids].values
    
    return common_ids, X, y


def get_ml_model(model_name, n_jobs=1):
    """Returns the ML classifier configuration."""
    if model_name == "ExtraTrees_shallow":
        return ExtraTreesClassifier(
            n_estimators=300, 
            max_depth=6, 
            min_samples_leaf=5, 
            n_jobs=n_jobs,
            random_state=SEED
        )
    elif model_name == "SVM_Linear":
        return SVC(
            kernel="linear", 
            C=1.0, 
            probability=True, 
            random_state=SEED
        )
    return None


def run_esm_grid_search(esm_path, dataset_name, master_labels, n_jobs=1):
    """Phase 1: Grid search over ESM variants."""
    agg_dir = esm_path
    
    # ESM variants - match submission/predictor.py exactly
    esm_variants = {
        "BertMax_RowMean": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_max/esm2_{dataset_name}_aggregated_mean.pkl"),
        "BertMax_RowMax": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_max/esm2_{dataset_name}_aggregated_max.pkl"),
        "BertMean_RowMean": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_mean/esm2_{dataset_name}_aggregated_mean.pkl"),
        "BertMean_RowMax": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_mean/esm2_{dataset_name}_aggregated_max.pkl")
    }
    
    results = []
    for variant_name, path in esm_variants.items():
        ids, X, y = load_esm_data(path, master_labels)
        if X is None:
            continue
        
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        
        for model_name in ["ExtraTrees_shallow", "SVM_Linear"]:
            print(f"   Testing [{variant_name}] + [{model_name}] ... ", end="")
            fold_aucs = []
            
            for train_idx, val_idx in kf.split(X, y):
                clf = get_ml_model(model_name, n_jobs)
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X[train_idx])
                X_val_sc = scaler.transform(X[val_idx])
                clf.fit(X_train_sc, y[train_idx])
                preds = clf.predict_proba(X_val_sc)[:, 1]
                fold_aucs.append(roc_auc_score(y[val_idx], preds))
            
            avg_auc = np.mean(fold_aucs)
            print(f"AUC: {avg_auc:.4f}")
            
            results.append({
                "variant": variant_name,
                "model": model_name,
                "auc": avg_auc,
                "ids": ids,
                "X": X,
                "y": y
            })
    
    if len(results) == 0:
        return None
    
    best = sorted(results, key=lambda x: x['auc'], reverse=True)[0]
    print(f"\n   >>> WINNER: {best['variant']} using {best['model']} (AUC: {best['auc']:.4f})")
    return best


# ==============================================================================
# MAIN PREDICTION FUNCTION
# ==============================================================================

def run_reproduce_prediction(train_dir: str,
                            test_dirs: List[str],
                            out_dir: str,
                            n_jobs: int = 1,
                            kmer_path: str = None,
                            esm_path: str = None,
                            model_path: str = None) -> pd.DataFrame:
    """
    Run reproduction prediction using saved model or by retraining the ensemble model.
    
    Args:
        train_dir: Training data directory
        test_dirs: List of test data directories
        out_dir: Output directory for predictions
        n_jobs: Number of jobs for parallel processing
        kmer_path: Path to K-mer features (optional, will auto-detect)
        esm_path: Path to ESM features directory (optional, will auto-detect)
        model_path: Path to saved model pickle (optional, will auto-detect)
        
    Returns:
        DataFrame with predictions
    """
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Set random seed
    np.random.seed(SEED)
    
    # Extract dataset number
    train_dataset_num = os.path.basename(train_dir).replace('train_dataset_', '')
    dataset_name = os.path.basename(train_dir)
    
    # Check for saved model (in script directory, not output directory)
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "models")
        model_path = os.path.join(model_dir, f"{dataset_name}_ensemble_model.pkl")
    
    use_saved_model = os.path.exists(model_path)
    
    print("\n" + "="*70)
    if use_saved_model:
        print(f"🎯 DATASET {train_dataset_num} REPRODUCTION MODE (Using Saved Model)")
        print(f"Model Path: {model_path}")
    else:
        print(f"🎯 DATASET {train_dataset_num} REPRODUCTION MODE (Retrain Ensemble)")
        print(f"Model not found at: {model_path}")
    print("="*70)
    print(f"Train Dataset: {train_dataset_num}")
    print(f"Random Seed: {SEED}")
    print(f"N-Folds CV: {N_FOLDS}")
    print(f"Output Directory: {out_dir}")
    
    # Load saved model if it exists
    if use_saved_model:
        print("\n[1/2] Loading Saved Model...")
        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)
        
        # Check if it's an ImmuneStatePredictor object with meta_model
        if hasattr(saved_data, 'predict_proba') and hasattr(saved_data, 'meta_model') and saved_data.meta_model is not None:
            # It's a trained ImmuneStatePredictor with meta_model
            # Extract components to avoid calling predictor.predict_proba (which may use simplified_weights)
            print(f"   ✓ Loaded ImmuneStatePredictor with meta_model")
            predictor = saved_data
            
            # Extract model components
            kmer_model = predictor.kmer_model
            kmer_scaler = predictor.kmer_scaler
            public_model = predictor.public_model
            meta_model = predictor.meta_model
            training_kmer_features = predictor.kmer_feature_names_
            best_esm_variant = predictor.best_esm_variant if hasattr(predictor, 'best_esm_variant') else None
            esm_model = predictor.esm_model if hasattr(predictor, 'esm_model') else None
            esm_scaler = predictor.esm_scaler if hasattr(predictor, 'esm_scaler') else None
            important_sequences_df = predictor.important_sequences_ if hasattr(predictor, 'important_sequences_') else None
            
            # Update out_dir for feature detection
            if hasattr(predictor, 'out_dir'):
                predictor.out_dir = out_dir
            
            # Auto-detect feature paths if not provided
            if kmer_path is None:
                kmer_dir = os.path.join(out_dir, "kmer")
                kmer_pattern = os.path.join(kmer_dir, f"*{dataset_name}*.pkl")
                kmer_files = glob.glob(kmer_pattern)
                if kmer_files:
                    kmer_path = kmer_files[0]
            
            if esm_path is None:
                esm_path = os.path.join(out_dir, "aggregates")
            
            # Generate predictions on test sets
            print("\n[2/2] Generating Predictions on Test Sets...")
            all_predictions = []
            
            for test_dir in test_dirs:
                test_dataset_num = os.path.basename(test_dir).replace('test_dataset_', '')
                print(f"\n   Processing test dataset: {test_dataset_num}")
                
                # Get test IDs
                test_files = get_repertoire_ids(test_dir)
                if len(test_files) == 0:
                    print(f"   ⚠️  No test files found")
                    continue
                
                print(f"   Found {len(test_files)} test samples")
                
                # Load K-mer features for test
                test_kmer_path = kmer_path.replace(dataset_name, f"test_dataset_{test_dataset_num}")
                X_kmer_test, test_kmer_feature_names = load_kmer_features(test_kmer_path, test_files)
                if X_kmer_test is None:
                    print(f"   ⚠️  K-mer features not found: {test_kmer_path}")
                    continue
                
                # Align test features to match training features
                test_df = pd.DataFrame(X_kmer_test, columns=test_kmer_feature_names, index=test_files)
                
                # Add missing columns (features present in train but not in test)
                missing_cols = set(training_kmer_features) - set(test_kmer_feature_names)
                if missing_cols:
                    missing_df = pd.DataFrame(0, index=test_df.index, columns=list(missing_cols))
                    test_df = pd.concat([test_df, missing_df], axis=1)
                
                # Reorder to match training features (also removes extra columns)
                test_df = test_df[training_kmer_features]
                X_kmer_test_aligned = test_df.values
                
                # Load public sequences for test
                seqs_dict_test = load_public_sequences(test_dir, test_files)
                
                # Generate predictions using extracted components (not calling predictor.predict_proba)
                Xk_scaled = kmer_scaler.transform(X_kmer_test_aligned)
                pred_kmer = kmer_model.predict_proba(Xk_scaled)[:, 1]
                pred_public = public_model.predict_proba(seqs_dict_test, test_files)
                
                if best_esm_variant is not None and esm_model is not None:
                    # Load ESM for test - determine row aggregation type
                    if 'RowMean' in best_esm_variant:
                        row_agg = 'mean'
                    else:  # RowMax
                        row_agg = 'max'
                    
                    test_esm_path = os.path.join(
                        esm_path,
                        f"aggregated_esm2_t6_8M_{'max' if 'BertMax' in best_esm_variant else 'mean'}",
                        f"esm2_test_dataset_{test_dataset_num}_aggregated_{row_agg}.pkl"
                    )
                    _, X_esm_test, _ = load_esm_data(test_esm_path, pd.Series(index=test_files))
                    if X_esm_test is not None:
                        Xe_scaled = esm_scaler.transform(X_esm_test)
                        pred_esm = esm_model.predict_proba(Xe_scaled)[:, 1]
                        stacked_preds = np.column_stack([pred_kmer, pred_public, pred_esm])
                    else:
                        stacked_preds = np.column_stack([pred_kmer, pred_public])
                else:
                    stacked_preds = np.column_stack([pred_kmer, pred_public])
                
                # Final prediction using meta_model
                probabilities = meta_model.predict_proba(stacked_preds)[:, 1]
                
                test_predictions = pd.DataFrame({
                    'ID': test_files,
                    'dataset': [test_dataset_num] * len(test_files),
                    'label_positive_probability': probabilities
                })
                test_predictions['junction_aa'] = -999.0
                test_predictions['v_call'] = -999.0
                test_predictions['j_call'] = -999.0
                test_predictions = test_predictions[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
                
                print(f"   ✓ Generated {len(test_predictions)} predictions")
                all_predictions.append(test_predictions)
            
            # Combine all predictions
            if not all_predictions:
                raise ValueError("No predictions were generated.")
            
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            
            # Save predictions
            output_filename = f"{dataset_name}_test_predictions.tsv"
            output_path = os.path.join(out_dir, output_filename)
            combined_predictions.to_csv(output_path, sep='\t', index=False)
            
            # Save important sequences
            if important_sequences_df is not None:
                important_sequences_filename = f"{dataset_name}_important_sequences.tsv"
                important_sequences_path = os.path.join(out_dir, important_sequences_filename)
                important_sequences_df.to_csv(important_sequences_path, sep='\t', index=False)
                print(f"\n   Saved {len(important_sequences_df)} important sequences to: {important_sequences_path}")
            
            print("\n" + "="*70)
            print(f"✅ REPRODUCTION COMPLETE")
            print("="*70)
            print(f"Total predictions: {len(combined_predictions)}")
            print(f"Predictions saved to: {output_path}")
            print(f"Method: ImmuneStatePredictor with meta_model (manual prediction)")
            print("="*70)
            
            return combined_predictions
        
        elif isinstance(saved_data, dict) and 'kmer_model' in saved_data and 'meta_model' in saved_data:
            # It's a dictionary with model components - use them directly
            print(f"   ✓ Loaded dictionary-based ensemble model")
            
            kmer_model = saved_data['kmer_model']
            kmer_scaler = saved_data['kmer_scaler']
            public_model = saved_data['public_model']
            meta_model = saved_data['meta_model']
            training_kmer_features = saved_data['training_kmer_features']
            best_esm_config = saved_data.get('best_esm_config')
            esm_model = saved_data.get('esm_model')
            esm_scaler = saved_data.get('esm_scaler')
            important_sequences_df = saved_data.get('important_sequences_df')
            
            # Auto-detect feature paths if not provided
            if kmer_path is None:
                kmer_dir = os.path.join(out_dir, "kmer")
                kmer_pattern = os.path.join(kmer_dir, f"*{dataset_name}*.pkl")
                kmer_files = glob.glob(kmer_pattern)
                if kmer_files:
                    kmer_path = kmer_files[0]
            
            if esm_path is None:
                esm_path = os.path.join(out_dir, "aggregates")
            
            # Generate predictions on test sets
            print("\n[2/2] Generating Predictions on Test Sets...")
            all_predictions = []
            
            for test_dir in test_dirs:
                test_dataset_num = os.path.basename(test_dir).replace('test_dataset_', '')
                print(f"\n   Processing test dataset: {test_dataset_num}")
                
                # Get test IDs
                test_files = sorted([f.replace('.tsv', '') for f in os.listdir(test_dir) if f.endswith('.tsv')])
                if len(test_files) == 0:
                    print(f"   ⚠️  No test files found")
                    continue
                
                print(f"   Found {len(test_files)} test samples")
                
                # Load K-mer features for test
                test_kmer_path = kmer_path.replace(dataset_name, f"test_dataset_{test_dataset_num}")
                X_kmer_test, test_kmer_feature_names = load_kmer_features(test_kmer_path, test_files)
                if X_kmer_test is None:
                    print(f"   ⚠️  K-mer features not found: {test_kmer_path}")
                    continue
                
                # Align test features to match training features
                test_df = pd.DataFrame(X_kmer_test, columns=test_kmer_feature_names, index=test_files)
                
                # Add missing columns (features present in train but not in test)
                missing_cols = set(training_kmer_features) - set(test_kmer_feature_names)
                if missing_cols:
                    missing_df = pd.DataFrame(0, index=test_df.index, columns=list(missing_cols))
                    test_df = pd.concat([test_df, missing_df], axis=1)
                
                # Reorder to match training features (also removes extra columns)
                test_df = test_df[training_kmer_features]
                X_kmer_test_aligned = test_df.values
                
                # Load public sequences for test
                seqs_dict_test = load_public_sequences(test_dir, test_files)
                
                # Generate predictions
                Xk_scaled = kmer_scaler.transform(X_kmer_test_aligned)
                pred_kmer = kmer_model.predict_proba(Xk_scaled)[:, 1]
                pred_public = public_model.predict_proba(seqs_dict_test, test_files)
                
                if best_esm_config is not None and esm_model is not None:
                    # Load ESM for test - determine row aggregation type
                    if 'RowMean' in best_esm_config['variant']:
                        row_agg = 'mean'
                    else:  # RowMax
                        row_agg = 'max'
                    
                    test_esm_path = os.path.join(
                        esm_path,
                        f"aggregated_esm2_t6_8M_{'max' if 'BertMax' in best_esm_config['variant'] else 'mean'}",
                        f"esm2_test_dataset_{test_dataset_num}_aggregated_{row_agg}.pkl"
                    )
                    _, X_esm_test, _ = load_esm_data(test_esm_path, pd.Series(index=test_files))
                    if X_esm_test is not None:
                        Xe_scaled = esm_scaler.transform(X_esm_test)
                        pred_esm = esm_model.predict_proba(Xe_scaled)[:, 1]
                        stacked_preds = np.column_stack([pred_kmer, pred_public, pred_esm])
                    else:
                        stacked_preds = np.column_stack([pred_kmer, pred_public])
                else:
                    stacked_preds = np.column_stack([pred_kmer, pred_public])
                
                # Final prediction
                probabilities = meta_model.predict_proba(stacked_preds)[:, 1]
                
                test_predictions = pd.DataFrame({
                    'ID': test_files,
                    'dataset': [test_dataset_num] * len(test_files),
                    'label_positive_probability': probabilities
                })
                test_predictions['junction_aa'] = -999.0
                test_predictions['v_call'] = -999.0
                test_predictions['j_call'] = -999.0
                test_predictions = test_predictions[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
                
                all_predictions.append(test_predictions)
                print(f"   ✓ Generated {len(test_predictions)} predictions")
            
            # Combine all predictions
            if not all_predictions:
                raise ValueError("No predictions were generated.")
            
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            
            # Save predictions
            output_filename = f"{dataset_name}_test_predictions.tsv"
            output_path = os.path.join(out_dir, output_filename)
            combined_predictions.to_csv(output_path, sep='\t', index=False)
            
            # Save important sequences
            if important_sequences_df is not None:
                important_sequences_filename = f"{dataset_name}_important_sequences.tsv"
                important_sequences_path = os.path.join(out_dir, important_sequences_filename)
                important_sequences_df.to_csv(important_sequences_path, sep='\t', index=False)
                print(f"\n   Saved {len(important_sequences_df)} important sequences to: {important_sequences_path}")
            
            print("\n" + "="*70)
            print(f"✅ REPRODUCTION COMPLETE")
            print("="*70)
            print(f"Total predictions: {len(combined_predictions)}")
            print(f"Predictions saved to: {output_path}")
            print(f"Method: Loaded saved ensemble model (dictionary format)")
            print("="*70)
            
            return combined_predictions
        else:
            # Model format not supported
            print(f"   ⚠️  Saved model format not supported (type: {type(saved_data).__name__})")
            if hasattr(saved_data, 'predict_proba'):
                if not hasattr(saved_data, 'meta_model') or saved_data.meta_model is None:
                    print(f"   Note: ImmuneStatePredictor is missing meta_model (needs retraining)")
                else:
                    print(f"   Note: ImmuneStatePredictor format not recognized")
            else:
                print(f"   Note: Expected ImmuneStatePredictor with meta_model or dictionary format")
            print(f"   Falling back to retraining...")
            use_saved_model = False
    
    if not use_saved_model:
        # Auto-detect feature paths if not provided
        if kmer_path is None:
            kmer_dir = os.path.join(out_dir, "kmer")
            kmer_pattern = os.path.join(kmer_dir, f"*{dataset_name}*.pkl")
            kmer_files = glob.glob(kmer_pattern)
            if kmer_files:
                kmer_path = kmer_files[0]
                print(f"K-mer Path: {kmer_path}")
        
        if esm_path is None:
            esm_path = os.path.join(out_dir, "aggregates")
            if os.path.exists(esm_path):
                print(f"ESM Path: {esm_path}")
        
        # 1. Load metadata and labels
        print("\n[1/7] Loading Training Metadata...")
        master_labels = load_metadata_and_labels(train_dir)
        print(f"   Loaded {len(master_labels)} samples")
        
        # 2. ESM Grid Search (if ESM features available)
        best_esm_config = None
        X_esm = None
        esm_ids = None
        y = master_labels.values
        ids_array = np.array(master_labels.index)
        
        if esm_path and os.path.exists(esm_path):
            print("\n[2/7] Running ESM Grid Search...")
            best_esm_config = run_esm_grid_search(esm_path, dataset_name, master_labels, n_jobs)
            if best_esm_config:
                esm_ids = best_esm_config['ids']
                X_esm = best_esm_config['X']
                y = best_esm_config['y']
                ids_array = np.array(esm_ids)
        else:
            print("\n[2/7] Skipping ESM Grid Search (no ESM features found)")
        
        # 3. Load K-mer features
        print("\n[3/7] Loading K-mer Features...")
        X_kmer, kmer_feature_names = load_kmer_features(kmer_path, ids_array)
        if X_kmer is None:
            raise ValueError("K-mer features are required but not found")
        print(f"   K-mer shape: {X_kmer.shape}")
        
        # Store feature names for test alignment
        training_kmer_features = kmer_feature_names
        
        # 4. Load Public Clone sequences
        print("\n[4/7] Loading Public Clone Sequences...")
        seqs_dict = load_public_sequences(train_dir, ids_array)
        print(f"   Loaded sequences for {len(seqs_dict)} samples")
        
        # 5. Train ensemble with cross-validation
        print("\n[5/7] Training Ensemble Models with CV...")
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        
        n_models = 2 if X_esm is None else 3
        oof_preds = np.zeros((len(ids_array), n_models))
        
        kmer_model = None
        public_model = None
        esm_model = None
        kmer_scaler = None
        esm_scaler = None
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_kmer, y)):
            print(f"   Fold {fold+1}/{N_FOLDS}")
            
            # A. K-mer Model
            kmer_model = LogisticRegression(
                penalty='l1', C=0.1, solver='liblinear',
                class_weight='balanced', random_state=SEED
            )
            kmer_scaler = StandardScaler()
            Xk_tr = kmer_scaler.fit_transform(X_kmer[train_idx])
            Xk_val = kmer_scaler.transform(X_kmer[val_idx])
            kmer_model.fit(Xk_tr, y[train_idx])
            oof_preds[val_idx, 0] = kmer_model.predict_proba(Xk_val)[:, 1]
            
            # B. Public Clone Model
            public_model = PublicCloneModel()
            public_model.fit(seqs_dict, y[train_idx], ids_array[train_idx])
            oof_preds[val_idx, 1] = public_model.predict_proba(seqs_dict, ids_array[val_idx])
            
            # C. ESM Model (if available)
            if X_esm is not None:
                esm_model = get_ml_model(best_esm_config['model'], n_jobs)
                esm_scaler = StandardScaler()
                Xe_tr = esm_scaler.fit_transform(X_esm[train_idx])
                Xe_val = esm_scaler.transform(X_esm[val_idx])
                esm_model.fit(Xe_tr, y[train_idx])
                oof_preds[val_idx, 2] = esm_model.predict_proba(Xe_val)[:, 1]
        
        # Train final models on full data
        print("   Training final models on full data...")
        kmer_scaler = StandardScaler()
        Xk_full = kmer_scaler.fit_transform(X_kmer)
        kmer_model = LogisticRegression(
            penalty='l1', C=0.1, solver='liblinear',
            class_weight='balanced', random_state=SEED
        )
        kmer_model.fit(Xk_full, y)
        
        public_model = PublicCloneModel()
        public_model.fit(seqs_dict, y, ids_array)
        
        if X_esm is not None:
            esm_scaler = StandardScaler()
            Xe_full = esm_scaler.fit_transform(X_esm)
            esm_model = get_ml_model(best_esm_config['model'], n_jobs)
            esm_model.fit(Xe_full, y)
        
        # Train meta-learner
        meta_model = LogisticRegression(penalty=None, solver='lbfgs')
        meta_model.fit(oof_preds, y)
        
        final_auc = roc_auc_score(y, meta_model.predict_proba(oof_preds)[:, 1])
        weights = meta_model.coef_[0]
        norm_w = weights / np.abs(weights).sum()
        
        print(f"\n   Final Ensemble CV AUC: {final_auc:.5f}")
        print(f"   Model Weights: K-mer={norm_w[0]:.2f}, Public={norm_w[1]:.2f}", end="")
        if X_esm is not None:
            print(f", ESM={norm_w[2]:.2f}")
        else:
            print()
        
        # 6. Save important sequences
        print("\n[6/7] Identifying Important Sequences...")
        if public_model and len(public_model.selected) > 0:
            selected_seqs = public_model.selected[:50000]
            sequences_data = []
            for seq_tuple in selected_seqs:
                if isinstance(seq_tuple, tuple) and len(seq_tuple) >= 3:
                    sequences_data.append({
                        'junction_aa': seq_tuple[0],
                        'v_call': seq_tuple[1],
                        'j_call': seq_tuple[2]
                    })
            
            important_sequences_df = pd.DataFrame(sequences_data)
            important_sequences_df['ID'] = [f'{dataset_name}_seq_top_{i+1}' for i in range(len(important_sequences_df))]
            important_sequences_df['dataset'] = dataset_name
            important_sequences_df['label_positive_probability'] = -999.0
            important_sequences_df = important_sequences_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
        else:
            important_sequences_df = pd.DataFrame({
                'ID': ['N/A'],
                'dataset': [dataset_name],
                'label_positive_probability': [-999.0],
                'junction_aa': ['N/A'],
                'v_call': ['N/A'],
                'j_call': ['N/A']
            })
        
        important_sequences_filename = f"{os.path.basename(train_dir)}_important_sequences.tsv"
        important_sequences_path = os.path.join(out_dir, important_sequences_filename)
        important_sequences_df.to_csv(important_sequences_path, sep='\t', index=False)
        print(f"   Saved {len(important_sequences_df)} important sequences")
        
        # 7. Predict on test sets
        print("\n[7/7] Generating Predictions on Test Sets...")
        all_predictions = []
        
        for test_dir in test_dirs:
            test_dataset_num = os.path.basename(test_dir).replace('test_dataset_', '')
            print(f"\n   Processing test dataset: {test_dataset_num}")
            
            # Get test IDs
            test_files = sorted([f.replace('.tsv', '') for f in os.listdir(test_dir) if f.endswith('.tsv')])
            if len(test_files) == 0:
                print(f"   ⚠️  No test files found")
                continue
            
            print(f"   Found {len(test_files)} test samples")
            
            # Load K-mer features for test
            test_kmer_path = kmer_path.replace(dataset_name, f"test_dataset_{test_dataset_num}")
            X_kmer_test, test_kmer_feature_names = load_kmer_features(test_kmer_path, test_files)
            if X_kmer_test is None:
                print(f"   ⚠️  K-mer features not found: {test_kmer_path}")
                continue
            
            # Align test features to match training features
            test_df = pd.DataFrame(X_kmer_test, columns=test_kmer_feature_names, index=test_files)
            
            # Add missing columns (features present in train but not in test)
            missing_cols = set(training_kmer_features) - set(test_kmer_feature_names)
            if missing_cols:
                missing_df = pd.DataFrame(0, index=test_df.index, columns=list(missing_cols))
                test_df = pd.concat([test_df, missing_df], axis=1)
            
            # Reorder to match training features (also removes extra columns)
            test_df = test_df[training_kmer_features]
            X_kmer_test_aligned = test_df.values
            
            # Load public sequences for test
            seqs_dict_test = load_public_sequences(test_dir, test_files)
            
            # Generate predictions
            Xk_scaled = kmer_scaler.transform(X_kmer_test_aligned)
            pred_kmer = kmer_model.predict_proba(Xk_scaled)[:, 1]
            pred_public = public_model.predict_proba(seqs_dict_test, test_files)
            
            if best_esm_config is not None and esm_model is not None:
                # Load ESM for test - determine row aggregation type
                if 'RowMean' in best_esm_config['variant']:
                    row_agg = 'mean'
                else:  # RowMax
                    row_agg = 'max'
                
                test_esm_path = os.path.join(
                    esm_path,
                    f"aggregated_esm2_t6_8M_{'max' if 'BertMax' in best_esm_config['variant'] else 'mean'}",
                    f"esm2_test_dataset_{test_dataset_num}_aggregated_{row_agg}.pkl"
                )
                _, X_esm_test, _ = load_esm_data(test_esm_path, pd.Series(index=test_files))
                if X_esm_test is not None:
                    Xe_scaled = esm_scaler.transform(X_esm_test)
                    pred_esm = esm_model.predict_proba(Xe_scaled)[:, 1]
                    stacked_preds = np.column_stack([pred_kmer, pred_public, pred_esm])
                else:
                    stacked_preds = np.column_stack([pred_kmer, pred_public])
            else:
                stacked_preds = np.column_stack([pred_kmer, pred_public])
            
            # Final prediction
            probabilities = meta_model.predict_proba(stacked_preds)[:, 1]
            
            test_predictions = pd.DataFrame({
                'ID': test_files,
                'dataset': [test_dataset_num] * len(test_files),
                'label_positive_probability': probabilities
            })
            test_predictions['junction_aa'] = -999.0
            test_predictions['v_call'] = -999.0
            test_predictions['j_call'] = -999.0
            test_predictions = test_predictions[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
            
            all_predictions.append(test_predictions)
            print(f"   ✓ Generated {len(test_predictions)} predictions")
        
        # Combine all predictions
        if not all_predictions:
            raise ValueError("No predictions were generated.")
        
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Save predictions
    output_filename = f"{os.path.basename(train_dir)}_test_predictions.tsv"
    output_path = os.path.join(out_dir, output_filename)
    combined_predictions.to_csv(output_path, sep='\t', index=False)
    
    print("\n" + "="*70)
    print(f"✅ REPRODUCTION COMPLETE")
    print("="*70)
    print(f"Total predictions: {len(combined_predictions)}")
    print(f"Predictions saved to: {output_path}")
    print(f"Important sequences saved to: {important_sequences_path}")
    print(f"Method: Retrained ensemble (K-mer + PublicClone + ESM)")
    print("="*70)
    
    return combined_predictions


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dataset 7 Reproduction Script - Retrain ensemble model from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python Dataset7_reproduce.py \\
        --train_dir /path/to/train_dataset_7 \\
        --test_dirs /path/to/test_dataset_7_1 /path/to/test_dataset_7_2 \\
        --out_dir /path/to/output \\
        --n_jobs 4

Method:
    Retrains the ensemble model (K-mer + PublicClone + ESM2) from scratch using the original
    training pipeline. Uses pre-computed K-mer and ESM features.

Model Components:
    - K-mer Model: L1 Logistic Regression (C=0.1)
    - Public Clone Model: Fisher's exact test + Logistic Regression (p<0.05, min_pos=3)
    - ESM Model: ExtraTrees or SVM on ESM2 t6_8M embeddings (selected via CV)
    - Meta Model: Logistic Regression stacking

Features Required:
    - K-mer features: {out_dir}/kmer/*train_dataset_7*.pkl
    - ESM features: {out_dir}/aggregates/aggregated_esm2_t6_8M_*/esm2_train_dataset_7_*.pkl
    """
    )
    
    parser.add_argument("--train_dir", required=True,
                       help="Path to training data directory")
    parser.add_argument("--test_dirs", required=True, nargs="+",
                       help="Path(s) to test data director(ies)")
    parser.add_argument("--out_dir", required=True,
                       help="Path to output directory")
    parser.add_argument("--n_jobs", type=int, default=1,
                       help="Number of CPU cores to use")
    parser.add_argument("--kmer_path", type=str, default=None,
                       help="Path to K-mer features pickle (optional, will auto-detect)")
    parser.add_argument("--esm_path", type=str, default=None,
                       help="Path to ESM features directory (optional, will auto-detect)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to saved model pickle (default: submission/kaggle_reproduce/models/train_dataset_7_ensemble_model.pkl)")
    
    args = parser.parse_args()
    
    try:
        run_reproduce_prediction(
            train_dir=args.train_dir,
            test_dirs=args.test_dirs,
            out_dir=args.out_dir,
            n_jobs=args.n_jobs,
            kmer_path=args.kmer_path,
            esm_path=args.esm_path,
            model_path=args.model_path
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during reproduction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
