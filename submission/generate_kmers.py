## imports used by the basic code template provided.

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import glob
import sys
import argparse
from collections import defaultdict, Counter
from typing import Iterator, Tuple, Union, List, Dict as PyDict
import pickle 
from pathlib import Path

## imports that are additionally used by this notebook

from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline

# --- GLOBAL VARIABLE FOR CACHE LOCATION ---
GLOBAL_CACHE_DIR = None
# ----------------------------------------

parser = argparse.ArgumentParser(description="Kmer Cache Generator CLI")
parser.add_argument("--train_dir", required=True,
                    help="Path to training data directory")
parser.add_argument("--test_dirs", required=True,
                    help="Path to test data directory")
parser.add_argument("--out_dir", required=True,
                    help="Path to output directory")
parser.add_argument("--n_jobs", type=int, default=1,
                    help="Number of CPU cores to use (for compatibility)")
parser.add_argument("--no-topseq", dest='topseq', action='store_false', default=True,
                    help="Disable ranking of important sequences (faster training)")
                    
args = parser.parse_args()

## some utility functions such as data loaders, etc.

def _get_cache_path(data_dir: str, k_lengths: List[int]) -> str:
    """Generates a consistent cache filename and path using the GLOBAL_CACHE_DIR."""
    global GLOBAL_CACHE_DIR
    
    if GLOBAL_CACHE_DIR is None:
         GLOBAL_CACHE_DIR = os.path.dirname(data_dir) 

    # Modified prefix to indicate all combined features are included
    feature_set_str = "k3_k4"
    
    dataset_name = os.path.basename(data_dir)
    
    return os.path.join(GLOBAL_CACHE_DIR, f"{feature_set_str}_{dataset_name}_features.pkl") #k3_k4_test_dataset_1_features.pkl

def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """ A generator to load immune repertoire data. """
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


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """ Loads all TSV files from a directory and concatenates them into a single DataFrame. """
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


def _get_kmer_counts_for_repertoire(data_df: pd.DataFrame, k: int) -> Counter:
    """Helper to count k-mers for a single repertoire using original Python loops."""
    kmer_counts = Counter()
    for seq in data_df['junction_aa'].dropna():
        if len(seq) >= k:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                kmer_counts[kmer] += 1
    return kmer_counts

# --- NEW: Function to extract all combined features ---
def load_and_extract_all_features(data_dir: str, k_lengths: List[int] = [3, 4]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all k-mers for a dataset, with caching.
    """
    
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    # Use dummy k_lengths to call the unified cache path function
    cache_path = _get_cache_path(data_dir, []) 

    # --- 1. CHECK CACHE ---
    if os.path.exists(cache_path):
        print(f"Loading combined features from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                combined_features_df, metadata_df = pickle.load(f)
            return combined_features_df, metadata_df
        except Exception as e:
            print(f"Warning: Failed to load cache file. Re-running encoding. Error: {e}")
            # Continue to recalculate features if cache load fails

    # --- 2. RUN ENCODING (If cache is missing or failed) ---
    data_loader_source = list(load_data_generator(data_dir=data_dir))
    total_files = len(data_loader_source)
    
    all_kmer_features = []
    metadata_records = []

    print(f"Starting feature calculation for {data_dir}...")

    # Iterate through all repertoires once
    for idx, item in enumerate(tqdm(data_loader_source, total=total_files, desc="Extracting all features")):
        
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            rep_id, data_df = item
            label = None
        
        # --- K-MER FEATURES ---
        rep_kmer_features = {'ID': rep_id}
        for k in k_lengths:
            kmer_counts = _get_kmer_counts_for_repertoire(data_df, k)
            prefixed_counts = {f"k{k}__{kmer}": count for kmer, count in kmer_counts.items()}
            rep_kmer_features.update(prefixed_counts)
        all_kmer_features.append(rep_kmer_features)
            
        # --- METADATA ---
        if idx == 0 or len(metadata_records) < total_files: # Only need to record metadata once per repertoire
            metadata_record = {'ID': rep_id}
            if label is not None:
                metadata_record['label_positive'] = label
            metadata_records.append(metadata_record)

    # --- COMBINE ALL FEATURES ---
    kmer_df = pd.DataFrame(all_kmer_features).fillna(0).set_index('ID')
    
    # Join all features on the repertoire ID (index)
    combined_features_df = pd.concat([kmer_df], axis=1, join='outer').fillna(0)
    metadata_df = pd.DataFrame(metadata_records)

    # --- 3. SAVE CACHE ---
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((combined_features_df, metadata_df), f)
        print(f"Features saved to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save features to cache file {cache_path}. Error: {e}")
    # ---------------------
    
    return combined_features_df, metadata_df
# -------------------------------------------------------------


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def get_repertoire_ids(data_dir: str) -> list:
    """ Retrieves repertoire IDs from the metadata file or filenames in the directory. """
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        repertoire_ids = metadata_df['repertoire_id'].tolist()
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        repertoire_ids = [os.path.basename(f).replace('.tsv', '') for f in sorted(tsv_files)]

    return repertoire_ids


def generate_random_top_sequences_df(n_seq: int = 50000) -> pd.DataFrame:
    """ Generates a random DataFrame simulating top important sequences. """
    seqs = set()
    while len(seqs) < n_seq:
        seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=15))
        seqs.add(seq)
    data = {
        'junction_aa': list(seqs),
        'v_call': ['TRBV20-1'] * n_seq,
        'j_call': ['TRBJ2-7'] * n_seq,
        'importance_score': np.random.rand(n_seq)
    }
    return pd.DataFrame(data)


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train directory `{train_dir}` does not exist."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` not found in train directory `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."

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
    """ Concatenates all test predictions and important sequences TSV files. """
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
    """Logistic Regression classifier for k-mer count data."""

    def __init__(self, random_state=123, n_jobs=1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model_ = None
        self.feature_names_ = None
        
    def _make_pipeline(self):
        """Create standardization + Logistic Regression pipeline."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                solver='liblinear'
            ))
        ])

    def tune_and_fit(self, X, y, val_size=0.2):
        """Fits the Logistic Regression model without grid search."""

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

        print("Fitting Logistic Regression model...")
        self.model_ = self._make_pipeline()
        self.model_.fit(X_train, y_train)

        if X_val is not None:
            val_probs = self.model_.predict_proba(X_val)[:, 1]
            self.val_score_ = roc_auc_score(y_val, val_probs)
            print(f"Validation ROC AUC: {self.val_score_:.4f}")

        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict(X)

    def get_feature_importance(self):
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        coefficients = self.model_.named_steps['classifier'].coef_[0]
        importance = np.abs(coefficients)

        if self.feature_names_ is not None:
            feature_names = self.feature_names_
        else:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_score': importance
        })

        importance_df = importance_df.sort_values('importance_score', ascending=False)

        return importance_df

    def score_all_sequences(self, sequences_df, sequence_col='junction_aa'):
        """ Score all sequences using global feature importance (absolute coefficients). """
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        feature_importance = self.get_feature_importance().set_index('feature')['importance_score']
        kmer_to_importance = feature_importance.to_dict()
        
        scores = []
        total_seqs = len(sequences_df)
        
        # Determine k-lengths from the feature names
        k_lengths = set()
        for name in self.feature_names_:
            if name.startswith('k'):
                try:
                    k_len = int(name.split('__')[0][1:])
                    k_lengths.add(k_len)
                except:
                    pass
        
        sorted_k_lengths = sorted(list(k_lengths))
        
        for seq in tqdm(sequences_df[sequence_col], total=total_seqs, desc="Scoring sequences"):
            seq_score = 0.0
            
            if isinstance(seq, str):
                for k in sorted_k_lengths:
                    if len(seq) >= k:
                        for i in range(len(seq) - k + 1):
                            kmer = seq[i:i + k]
                            prefixed_kmer = f"k{k}__{kmer}" 
                            seq_score += kmer_to_importance.get(prefixed_kmer, 0.0)
                            
            scores.append(seq_score)

        result_df = sequences_df.copy()
        result_df['importance_score'] = scores
        return result_df


def prepare_data(X_df, labels_df, id_col='ID', label_col='label_positive'):
    """ Merge feature matrix with labels, ensuring alignment. """
    if id_col in labels_df.columns:
        labels_indexed = labels_df.set_index(id_col)[label_col]
    else:
        labels_indexed = labels_df[label_col]

    common_ids = X_df.index.intersection(labels_indexed.index)

    if len(common_ids) == 0:
        raise ValueError("No common IDs found between feature matrix and labels")

    X = X_df.loc[common_ids]
    y = labels_indexed.loc[common_ids]

    print(f"Aligned {len(common_ids)} samples with labels")

    return X, y, common_ids


class ImmuneStatePredictor:
    """ A template for predicting immune states from TCR repertoire data. """

    def __init__(self, n_jobs: int = 1, device: str = 'cpu', **kwargs):
        self.train_ids_ = None
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        
        self.device = 'cpu'
            
        self.model = None
        self.important_sequences_ = None

    def fit(self, train_dir_path: str):
        """ Trains the model on the provided training data, reading features from cache. """

        # --- UPDATED TO USE COMBINED FEATURE FUNCTION ---
        X_train_df, y_train_df = load_and_extract_all_features(train_dir_path, k_lengths=[3, 4]) 
        # -----------------------------------------------

        X_train, y_train, train_ids = prepare_data(X_train_df, y_train_df,
                                                   id_col='ID', label_col='label_positive')
        
        self.model = KmerClassifier(
            random_state=123,
            n_jobs=self.n_jobs
        )
        '''
        self.model.tune_and_fit(X_train, y_train)

        self.train_ids_ = train_ids
        
        # This requires loading the full data to get the sequences for V/J calls.
        self.important_sequences_ = self.identify_associated_sequences(train_dir_path=train_dir_path, top_k=50000)
        
        print("Training complete.")
        '''
        print("Training kmer cache saved.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """ Predicts probabilities for examples in the provided path, reading features from cache. """
        print(f"Making predictions for data in {test_dir_path}...")
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Please call `fit` first.")

        # --- UPDATED TO USE COMBINED FEATURE FUNCTION ---
        X_test_df, _ = load_and_extract_all_features(test_dir_path, k_lengths=[3, 4]) 

        print("test kmer cache saved.")
        # -----------------------------------------------
        '''
        if self.model.feature_names_ is not None:
            # Align test features to the order of train features
            X_test_df = X_test_df.reindex(columns=self.model.feature_names_, fill_value=0)

        repertoire_ids = X_test_df.index.tolist()

        probabilities = self.model.predict_proba(X_test_df)

        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })

        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0

        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        print(f"Prediction complete on {len(repertoire_ids)} examples in {test_dir_path}.")
        '''
        predictions_df = None
        return predictions_df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        """ Identifies the top "k" important sequences (rows) from the training data that best explain the labels. """
        dataset_name = os.path.basename(train_dir_path)

        full_df = load_full_dataset(train_dir_path)
        unique_seqs = full_df[['junction_aa', 'v_call', 'j_call']].drop_duplicates()
        all_sequences_scored = self.model.score_all_sequences(unique_seqs, sequence_col='junction_aa')

        top_sequences_df = all_sequences_scored.nlargest(top_k, 'importance_score')
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df)+1)
        top_sequences_df['ID'] = top_sequences_df['dataset'] + '_seq_top_' + top_sequences_df['ID'].astype(str)
        top_sequences_df['label_positive_probability'] = -999.0
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        return top_sequences_df


## The `main` workflow

def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    """Trains the predictor on the training data."""
    print(f"Fitting model on examples in ` {train_dir} `...")
    predictor.fit(train_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]) -> pd.DataFrame:
    """Generates predictions for all test directories and concatenates them."""
    all_preds = []
    for test_dir in test_dirs:
        print(f"Predicting on examples in ` {test_dir} `...")
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


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    
    global GLOBAL_CACHE_DIR
    GLOBAL_CACHE_DIR = out_dir
    
    predictor = ImmuneStatePredictor(n_jobs=n_jobs,
                                     device=device) 
    _train_predictor(predictor, train_dir)
    predictions = _generate_predictions(predictor, test_dirs)
    #_save_predictions(predictions, out_dir, train_dir)
    #_save_important_sequences(predictor, out_dir, train_dir)


#METADATA_PATH = Path(BASE_PATH / "metadata.csv")

train_datasets_dir = Path(args.train_dir) 
test_datasets_dir = Path(args.test_dirs)
results_dir = Path(args.out_dir)

train_test_dataset_pairs = get_dataset_pairs(train_datasets_dir, test_datasets_dir)

TARGET_DATASET_IDS = {'1','2','3','4','5','6','7','8'}

print(f"Filtering datasets to process only IDs: {', '.join(TARGET_DATASET_IDS)}")

for train_dir, test_dirs in train_test_dataset_pairs:
    dataset_name = os.path.basename(train_dir)
    try:
        dataset_id = dataset_name.split('_')[-1] 
    except:
        dataset_id = ""
        
    if dataset_id in TARGET_DATASET_IDS:
        print(f"\n✅ Processing dataset ID {dataset_id} ({dataset_name})...")
        try:
            main(train_dir=train_dir, test_dirs=test_dirs, out_dir=results_dir, n_jobs=4, device="cpu")
        except FileNotFoundError as e:
            print(f"❌ ERROR: Skipping {dataset_name} because input TSV files for sequences were not found. Details: {e}")
            continue
    else:
        print(f"   ⏩ Skipping dataset {dataset_name} (ID {dataset_id}).")

"""
python generate_kmers.py \
    --train_dir /Users/quack/projects/airr/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets \
    --test_dirs /Users/quack/projects/airr/adaptive-immune-profiling-challenge-2025/test_datasets/test_datasets \
    --out_dir /Users/quack/projects/airr/output_reproduce
"""