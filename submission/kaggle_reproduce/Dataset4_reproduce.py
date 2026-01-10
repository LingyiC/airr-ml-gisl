import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import glob
import sys
import argparse
from collections import defaultdict, Counter
from typing import Iterator, Tuple, Union, List
#import psutil
import gc
import pickle

## imports that are additionally used by this notebook

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline

# Same as basline code template provided
def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """
    A generator to load immune repertoire data.

    This function operates in two modes:
    1.  If metadata is found, it yields data based on the metadata file.
    2.  If metadata is NOT found, it uses glob to find and yield all '.tsv'
        files in the directory.

    Args:
        data_dir (str): The path to the directory containing the data.

    Yields:
        An iterator of tuples. The format depends on the mode:
        - With metadata: (repertoire_id, pd.DataFrame, label_positive)
        - Without metadata: (filename, pd.DataFrame)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                print(f"Warning: File '{row.filename}' listed in metadata not found. Skipping.")
                continue
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        for file_path in sorted(tsv_files):
            try:
                filename = os.path.basename(file_path)
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield filename, repertoire_df
            except Exception as e:
                print(f"Warning: Could not read file '{file_path}'. Error: {e}. Skipping.")
                continue

# Same as basline code template provided
def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """
    Loads all TSV files from a directory and concatenates them into a single DataFrame.

    This function handles two scenarios:
    1. If metadata.csv exists, it loads data based on the metadata and adds
       'repertoire_id' and 'label_positive' columns.
    2. If metadata.csv does not exist, it loads all .tsv files and adds
       a 'filename' column as an identifier.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        pd.DataFrame: A single, concatenated DataFrame containing all the data.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total_files = len(metadata_df)
        for rep_id, data_df, label in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        total_files = len(glob.glob(search_pattern))
        for filename, data_df in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = os.path.basename(filename).replace(".tsv", "")
            df_list.append(data_df)

    if not df_list:
        print("Warning: No data files were loaded.")
        return pd.DataFrame()

    full_dataset_df = pd.concat(df_list, ignore_index=True)
    return full_dataset_df


# LC: Major change made here 
def load_and_encode_kmers_combined(data_dir: str, k_list: List[int] = [3, 4], 
                                   min_kmer_count: int = 2,
                                   batch_size: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loading and combined k-mer encoding of repertoire data.
    
    Memory-aware approach:
    - Process files in batches
    - Filter k-mers that appear less than min_kmer_count times (reduces feature space)
    - Combine features from multiple k values
    - Explicit garbage collection to manage memory

    Args:
        data_dir: Path to data directory
        k_list: List of k-mer lengths to use (e.g., [3, 4])
        min_kmer_count: Minimum count threshold for k-mers (filters rare k-mers)
        batch_size: Number of files to process before saving intermediate results

    Returns:
        Tuple of (encoded_features_df, metadata_df)
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    repertoire_features = []
    metadata_records = []
    
    # Global k-mer counter to track which k-mers are actually used
    global_kmer_counts = Counter()

    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    print(f"\n[K-mer Encoding] Processing {total_files} files with k-values: {k_list}")
    print(f"[K-mer Encoding] Min k-mer count threshold: {min_kmer_count}")
    
    # First pass: collect all k-mers to identify frequent ones
    print("\n[K-mer Encoding] First pass: collecting k-mer frequencies...")
    data_loader = load_data_generator(data_dir=data_dir)
    
    file_count = 0
    for item in tqdm(data_loader, total=total_files, desc="Collecting k-mer frequencies"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
            label = None

        for seq in data_df['junction_aa'].dropna():
            for k in k_list:
                for i in range(len(seq) - k + 1):
                    global_kmer_counts[seq[i:i + k]] += 1

        del data_df
        file_count += 1
        
        # Periodic garbage collection
        #if file_count % batch_size == 0:
        #    gc.collect()
        #    MemoryMonitor.log_memory(f"After {file_count} files")

    # Filter to keep only frequent k-mers
    frequent_kmers = {kmer for kmer, count in global_kmer_counts.items() if count >= min_kmer_count}
    print(f"\n[K-mer Encoding] Total unique k-mers: {len(global_kmer_counts)}")
    print(f"[K-mer Encoding] Frequent k-mers (count >= {min_kmer_count}): {len(frequent_kmers)}")
    print(f"[K-mer Encoding] Memory reduction: {(1 - len(frequent_kmers)/len(global_kmer_counts))*100:.1f}%")

    # Second pass: encode using only frequent k-mers
    print("\n[K-mer Encoding] Second pass: encoding features...")
    data_loader = load_data_generator(data_dir=data_dir)
    file_count = 0
    
    for item in tqdm(data_loader, total=total_files, desc=f"Encoding {k_list}-mers"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
            label = None

        # Count only frequent k-mers
        kmer_counts = {}
        for seq in data_df['junction_aa'].dropna():
            for k in k_list:
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i + k]
                    if kmer in frequent_kmers:
                        kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1

        repertoire_features.append({
            'ID': rep_id,
            **kmer_counts
        })

        metadata_record = {'ID': rep_id}
        # Always add label_positive (even if None) to ensure consistency
        metadata_record['label_positive'] = label
        metadata_records.append(metadata_record)

        del data_df, kmer_counts
        file_count += 1
        
        # Periodic garbage collection and memory logging
        #if file_count % batch_size == 0:
        #    gc.collect()
        #    MemoryMonitor.log_memory(f"After encoding {file_count} files")

    features_df = pd.DataFrame(repertoire_features).fillna(0).set_index('ID')
    metadata_df = pd.DataFrame(metadata_records)
    
    print(f"\n[K-mer Encoding] Final feature matrix shape: {features_df.shape}")
    #MemoryMonitor.log_memory("After encoding complete")

    return features_df, metadata_df    


# This chunck is the same as basline code template provided


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def get_repertoire_ids(data_dir: str) -> list:
    """
    Retrieves repertoire IDs from the metadata file or filenames in the directory.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        list: A list of repertoire IDs.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        repertoire_ids = metadata_df['repertoire_id'].tolist()
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        repertoire_ids = [os.path.basename(f).replace('.tsv', '') for f in sorted(tsv_files)]

    return repertoire_ids


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train directory `{train_dir}` does not exist."
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs, f"No .tsv files found in train directory `{train_dir}`."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` not found in train directory `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs, f"No .tsv files found in test directory `{test_dir}`."

    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, "test_write_permission.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Failed to create or write to output directory `{out_dir}`: {e}")
        sys.exit(1)


def concatenate_output_files(out_dir: str) -> None:
    """
    Concatenates all test predictions and important sequences TSV files from the output directory.

    This function finds all files matching the patterns:
    - *_test_predictions.tsv
    - *_important_sequences.tsv

    and concatenates them to match the expected output format of submissions.csv.

    Args:
        out_dir (str): Path to the output directory containing the TSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame with predictions followed by important sequences.
                     Columns: ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    """
    predictions_pattern = os.path.join(out_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(out_dir, '*_important_sequences.tsv')

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []

    for pred_file in predictions_files:
        try:
            df = pd.read_csv(pred_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read predictions file '{pred_file}'. Error: {e}. Skipping.")
            continue

    for seq_file in sequences_files:
        try:
            df = pd.read_csv(seq_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read sequences file '{seq_file}'. Error: {e}. Skipping.")
            continue

    if not df_list:
        print("Warning: No output files were found to concatenate.")
        concatenated_df = pd.DataFrame(
            columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    
    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output written to `{submissions_file}`.")


def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    """Returns list of (train_path, [test_paths]) tuples for dataset pairs."""
    test_groups = defaultdict(list)
    for test_name in sorted(os.listdir(test_dir)):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, test_name))

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, train_name)
            pairs.append((train_path, test_groups.get(train_id, [])))

    return pairs

class KmerClassifier:
    """L1-regularized logistic regression for combined k-mer count data."""

    def __init__(self, c_values=None, cv_folds=5,
                 opt_metric='balanced_accuracy', random_state=None, n_jobs=1):
        if c_values is None:
            c_values = [1, 0.1, 0.05, 0.03]
        self.c_values = c_values
        self.cv_folds = cv_folds
        self.opt_metric = opt_metric
        self.random_state = random_state if random_state is not None else 42
        self.n_jobs = n_jobs
        self.best_C_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.model_ = None
        self.feature_names_ = None
        self.val_score_ = None

    def _make_pipeline(self, C):
        """Create standardization + L1 logistic regression pipeline."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l1', C=C, solver='liblinear',
                random_state=self.random_state, max_iter=1000
            ))
        ])

    def _get_scorer(self):
        """Get scoring function for optimization."""
        if self.opt_metric == 'balanced_accuracy':
            return 'balanced_accuracy'
        elif self.opt_metric == 'roc_auc':
            return 'roc_auc'
        else:
            raise ValueError(f"Unknown metric: {self.opt_metric}")

    def tune_and_fit(self, X, y, val_size=0.0):
        """Perform CV tuning on all data (no validation split) to use all available training data."""

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state, stratify=y)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        scorer = self._get_scorer()

        results = []
        for C in self.c_values:
            pipeline = self._make_pipeline(C)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer,
                                     n_jobs=self.n_jobs)
            results.append({
                'C': C,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            })

        self.cv_results_ = pd.DataFrame(results)
        best_idx = self.cv_results_['mean_score'].idxmax()
        self.best_C_ = self.cv_results_.loc[best_idx, 'C']
        self.best_score_ = self.cv_results_.loc[best_idx, 'mean_score']

        print(f"Best C: {self.best_C_} (CV {self.opt_metric}: {self.best_score_:.4f})")

        # Fit on training split with best hyperparameter
        self.model_ = self._make_pipeline(self.best_C_)
        self.model_.fit(X_train, y_train)

        if X_val is not None:
            if scorer == 'balanced_accuracy':
                self.val_score_ = balanced_accuracy_score(y_val, self.model_.predict(X_val))
            else:  # roc_auc
                self.val_score_ = roc_auc_score(y_val, self.model_.predict_proba(X_val)[:, 1])
            print(f"Validation {self.opt_metric}: {self.val_score_:.4f}")

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X):
        """Predict class labels."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict(X)

    def get_feature_importance(self):
        """Get feature importance from L1 coefficients."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        coef = self.model_.named_steps['classifier'].coef_[0]

        if self.feature_names_ is not None:
            feature_names = self.feature_names_
        else:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        })

        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

        return importance_df

    # changed this one 
    def score_all_sequences(self, sequences_df, sequence_col='junction_aa', 
                           batch_size=500, k_list=[3, 4]):
        """
        Score all sequences using model coefficients (vectorized, batched, memory-aware).

        Parameters:
            sequences_df: DataFrame with unique sequences
            sequence_col: Column name containing sequences
            batch_size: Number of sequences to process in each batch
            k_list: List of k-mer sizes used in training

        Returns:
            DataFrame with added 'importance_score' column
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        scaler = self.model_.named_steps['scaler']
        coefficients = self.model_.named_steps['classifier'].coef_[0]
        coefficients = coefficients / scaler.scale_

        kmer_to_index = {kmer: idx for idx, kmer in enumerate(self.feature_names_)}

        scores = []
        total_seqs = len(sequences_df)
        sequences_list = sequences_df[sequence_col].tolist()
        
        # Process in batches for better performance and memory efficiency
        for batch_start in tqdm(range(0, total_seqs, batch_size), desc="Scoring sequences (batched)"):
            batch_end = min(batch_start + batch_size, total_seqs)
            batch_scores = []
            
            for seq in sequences_list[batch_start:batch_end]:
                counts = np.zeros(len(kmer_to_index), dtype=np.uint8)
                # Extract all k-mers from sequence
                for k in k_list:
                    for i in range(max(0, len(seq) - k + 1)):
                        kmer = seq[i:i + k]
                        if kmer in kmer_to_index:
                            counts[kmer_to_index[kmer]] = 1
                batch_scores.append(np.dot(counts, coefficients))
            
            scores.extend(batch_scores)
            
            # Periodic garbage collection
            if (batch_start // batch_size) % 10 == 0:
                gc.collect()

        result_df = sequences_df.copy()
        result_df['importance_score'] = scores
        return result_df

# Same as basline code template provided
def prepare_data(X_df, labels_df, id_col='ID', label_col='label_positive'):
    """
    Merge feature matrix with labels, ensuring alignment.

    Parameters:
        X_df: DataFrame with samples as rows (index contains IDs)
        labels_df: DataFrame with ID column and label column
        id_col: Name of ID column in labels_df
        label_col: Name of label column in labels_df

    Returns:
        X: Feature matrix aligned with labels
        y: Binary labels
        common_ids: IDs that were kept
    """
    if id_col in labels_df.columns:
        labels_indexed = labels_df.set_index(id_col)[label_col]
    else:
        # Assume labels_df index is already the ID
        labels_indexed = labels_df[label_col]

    common_ids = X_df.index.intersection(labels_indexed.index)

    if len(common_ids) == 0:
        raise ValueError("No common IDs found between feature matrix and labels")

    X = X_df.loc[common_ids]
    y = labels_indexed.loc[common_ids]

    print(f"Aligned {len(common_ids)} samples with labels")

    return X, y, common_ids

class ImmuneStatePredictor:
    """
    A template for predicting immune states from TCR repertoire data.

    Participants should implement the logic for training, prediction, and

    sequence identification within this class.

    Immune state predictor using combined k-mer encoding (3-mers + 4-mers).
    """

    def __init__(self, k_list=[3, 4], min_kmer_count=2, n_jobs=1, device='cpu', **kwargs):
        """
        Initializes the predictor.

        Args:
            k_list: List of k-mer lengths to use (default: [3, 4])
            min_kmer_count: Minimum count threshold for k-mers (memory optimization)
            n_jobs: Number of CPU cores to use
            device: Device for computation ('cpu' or 'cuda')
        """
        self.k_list = k_list
        self.min_kmer_count = min_kmer_count
        self.train_ids_ = None
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: 'cuda' was requested but is not available. Falling back to 'cpu'.")
            self.device = 'cpu'

        self.model = None
        self.important_sequences_ = None

    def fit(self, train_dir_path: str):
        """
        Trains the model on the provided training data using ALL available data.

        Args:
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            self: The fitted predictor instance.
        """
        print(f"\n[Training] Starting fit on {train_dir_path}")
        print(f"[Training] K-mer sizes: {self.k_list}")
        print(f"[Training] Min k-mer count: {self.min_kmer_count}")

        # Load and encode k-mers with combined k values
        X_train_df, y_train_df = load_and_encode_kmers_combined(
            train_dir_path, 
            k_list=self.k_list,
            min_kmer_count=self.min_kmer_count
        )

        X_train, y_train, train_ids = prepare_data(X_train_df, y_train_df,
                                                   id_col='ID', label_col='label_positive')

        self.model = KmerClassifier(
            c_values=[1, 0.2, 0.1, 0.05, 0.03],
            cv_folds=5,
            opt_metric='roc_auc',
            random_state=42, # Baseline used 123
            n_jobs=self.n_jobs
        )

        # Use all training data (val_size=0.0) to maximize training data usage; seems better for final model
        self.model.tune_and_fit(X_train, y_train, val_size=0.0)

        self.train_ids_ = train_ids

        # Identify important sequences
        self.important_sequences_ = self.identify_associated_sequences(
            train_dir_path=train_dir_path, 
            top_k=50000  # Updated to 50k for submission
        )

        print("[Training] Training complete.")
        #MemoryMonitor.log_memory("After training")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """
        Predicts probabilities for examples in the provided path.

        Args:
            test_dir_path (str): Path to the directory with test TSV files.

        Returns:
            pd.DataFrame: Predictions with proper format.
        """
        print(f"\n[Prediction] Making predictions for {test_dir_path}...")
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Please call `fit` first.")

        X_test_df, _ = load_and_encode_kmers_combined(
            test_dir_path,
            k_list=self.k_list,
            min_kmer_count=1  # Lower threshold for test data
        )

        if self.model.feature_names_ is not None:
            X_test_df = X_test_df.reindex(columns=self.model.feature_names_, fill_value=0)

        repertoire_ids = X_test_df.index.tolist()

        # Make predictions
        probabilities = self.model.predict_proba(X_test_df)

        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })

        # Add placeholder columns for output format compatibility
        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0

        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        print(f"[Prediction] Completed predictions for {len(repertoire_ids)} examples.")
        #MemoryMonitor.log_memory("After prediction")
        return predictions_df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Identifies the top k important sequences from the training data, ranked by importance score.

        Args:
            top_k (int): Number of top sequences to return
            train_dir_path (str): Path to training directory

        Returns:
            pd.DataFrame: Top important sequences with scores, ranked by importance_score descending
        """
        print(f"\n[Sequence Identification] Identifying top {top_k} sequences...")
        dataset_name = os.path.basename(train_dir_path)

        # Load full dataset to get unique sequences
        full_df = load_full_dataset(train_dir_path)
        unique_seqs = full_df[['junction_aa', 'v_call', 'j_call']].drop_duplicates().reset_index(drop=True)
        
        print(f"[Sequence Identification] Scoring {len(unique_seqs)} unique sequences...")

        # Baseline: all_sequences_scored = self.model.score_all_sequences(unique_seqs, sequence_col='junction_aa')
        all_sequences_scored = self.model.score_all_sequences(
            unique_seqs, 
            sequence_col='junction_aa',
            batch_size=500,
            k_list=self.k_list
        )

        top_sequences_df = all_sequences_scored.nlargest(top_k, 'importance_score')
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df)+1)
        top_sequences_df['ID'] = top_sequences_df['dataset'] + '_seq_top_' + top_sequences_df['ID'].astype(str)
        top_sequences_df['label_positive_probability'] = -999.0# to enable compatibility with the expected output format
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        print(f"[Sequence Identification] Identified {len(top_sequences_df)} top sequences ranked by importance score.")
        #MemoryMonitor.log_memory("After sequence identification")
        return top_sequences_df

def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    """Trains the predictor on the training data."""
    print(f"\nFitting model on examples in `{train_dir}`...")
    predictor.fit(train_dir)


def _save_model(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Saves the trained model to a pickle file."""
    if predictor.model is None:
        raise ValueError("No trained model available to save")
    
    model_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(predictor.model, f)
    print(f"Trained model saved to `{model_path}`.")


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]) -> pd.DataFrame:
    """Generates predictions for all test directories and concatenates them."""
    all_preds = []
    for test_dir in test_dirs:
        print(f"\nPredicting on examples in `{test_dir}`...")
        preds = predictor.predict_proba(test_dir)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
        else:
            print(f"Warning: No predictions returned for {test_dir}")
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str) -> None:
    """Saves predictions to a TSV file."""
    if predictions.empty:
        raise ValueError("No predictions to save - predictions DataFrame is empty")

    preds_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, preds_path)
    print(f"Predictions written to `{preds_path}`.")


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Saves important sequences to a TSV file."""
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        raise ValueError("No important sequences available to save")

    seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, seqs_path)
    print(f"Important sequences written to `{seqs_path}`.")


def run_reproduce_prediction(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str, 
         k_list: List[int] = [3, 4], min_kmer_count: int = 2, save_model: bool = True) -> None:
    """Main pipeline for training and prediction."""
    validate_dirs_and_files(train_dir, test_dirs, out_dir)

    # LC: Changed here to use the updated class
    predictor = ImmuneStatePredictor(
        k_list=k_list,
        min_kmer_count=min_kmer_count,
        n_jobs=n_jobs,
        device=device
    )
    _train_predictor(predictor, train_dir)
    
    # Save the trained model
    if save_model:
        _save_model(predictor, out_dir, train_dir)
    
    predictions = _generate_predictions(predictor, test_dirs)
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)



# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dataset 1 Reproduction Script ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Example usage:
    python Dataset1_reproduce.py \\
        --train_dir /path/to/train_dataset_1 \\
        --test_dirs /path/to/test_dataset_1_1 /path/to/test_dataset_1_2 \\
        --out_dir /path/to/output \\
        --n_jobs 4
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

    args = parser.parse_args()
    

    K_MERS = [3, 4]  
    MIN_KMER_COUNT = 20  # Filter out

    try:
        run_reproduce_prediction(
            train_dir=args.train_dir,
            test_dirs=args.test_dirs,
            out_dir=args.out_dir,
            n_jobs=args.n_jobs,
            device="cpu",
            k_list=K_MERS,
            min_kmer_count=MIN_KMER_COUNT,
            save_model=True
        )
        print(f"\nDataset completed successfully.")
        
        concatenate_output_files(args.out_dir)
        print(f"\nConcatenated output files saved.")
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()


"""
    python Dataset4_reproduce.py \
        --train_dir /Users/quack/projects/airr/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets/train_dataset_4 \
        --test_dirs /Users/quack/projects/airr/adaptive-immune-profiling-challenge-2025/test_datasets/test_datasets/test_dataset_4 \
        --out_dir /Users/quack/projects/airr/output_reproduce \
        --n_jobs 4
"""
