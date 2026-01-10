import os
import numpy as np
import pandas as pd
import pickle
import glob
import scipy.stats as stats
from tqdm import tqdm
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from submission.utils import load_data_generator, get_repertoire_ids


class PublicCloneModel:
    """Fisher's exact test based public clone model."""
    def __init__(self, p_val=0.05, min_pos=3):
        self.p_val = p_val
        self.min_pos = min_pos
        self.selected = []
        self.clf = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', class_weight='balanced', random_state=42)
        
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
        
        # Sort to ensure deterministic iteration order
        for seq, (cp, cn) in sorted(counts.items()):
            if cp < self.min_pos: continue
            if cn > 0: continue
            _, p = stats.fisher_exact([[cp, cn], [n_pos-cp, n_neg-cn]], alternative='greater')
            if p < self.p_val: self.selected.append(seq)
            
        if self.selected:
            self.selected.sort()  # Ensure deterministic order
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


class ImmuneStatePredictor:
    """
    Ensemble predictor combining K-mer features, Public Clones, and ESM2 embeddings.
    """

    def __init__(self, n_jobs: int = 1, device: str = 'cpu', **kwargs):
        """
        Initializes the predictor.

        Args:
            n_jobs (int): Number of CPU cores to use for parallel processing.
            device (str): The device to use for computation (e.g., 'cpu', 'cuda').
            **kwargs: Additional hyperparameters for the model.
        """
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        self.device = device
        
        # Model components
        self.kmer_model = None
        self.public_model = None
        self.esm_model = None
        self.meta_model = None
        
        # Scalers
        self.kmer_scaler = None
        self.esm_scaler = None
        
        # Feature paths (to be configured)
        self.out_dir = kwargs.get('out_dir', None)
        self.kmer_path = kwargs.get('kmer_path', None)
        self.esm_path = kwargs.get('esm_path', None)
        self.best_esm_variant = None
        self.best_esm_model_type = None
        
        # ESM generation settings
        self.auto_generate_esm = kwargs.get('auto_generate_esm', False)
        self.esm_model_name = kwargs.get('esm_model_name', 'facebook/esm2_t6_8M_UR50D')
        self.esm_batch_size = kwargs.get('esm_batch_size', 128)
        
        # Training data references
        self.train_ids = None
        self.important_sequences_ = None
        self.kmer_feature_names_ = None  # Store training K-mer feature names
        
        # Configuration
        self.seed = 42
        self.n_folds = 5
        self.model_selection_method = kwargs.get('model_selection_method', 'hybrid')  # 'cv', 'weights', or 'hybrid'
        self.rank_topseq = kwargs.get('rank_topseq', True)  # Whether to rank important sequences

    def fit(self, train_dir_path: str):
        """
        Trains the model on the provided training data.

        Args:
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            self: The fitted predictor instance.
        """
        # Set seed for reproducibility
        np.random.seed(self.seed)
        
        # Ensure single-threaded BLAS/LAPACK for reproducibility
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        print("="*60)
        print("Starting Ensemble Training Pipeline")
        print("="*60)
        
        # 1. Load metadata and labels
        print("\n1. Loading metadata and labels...")
        master_labels = self._load_metadata_and_labels(train_dir_path)
        
        # 2. Auto-detect feature paths based on dataset
        dataset_name = os.path.basename(train_dir_path)
        print(f"\n2. Auto-detecting features for dataset: {dataset_name}")
        self._detect_feature_paths(dataset_name)
        
        # 2.5 Detect dataset type (synthetic vs experimental)
        self.dataset_type = self._detect_dataset_type(train_dir_path)
        
        # 3. Phase 1: ESM Grid Search (if ESM features available)
        if self.esm_path is not None:
            print("\n3. Running ESM Grid Search...")
            best_config = self._run_esm_grid_search(master_labels, dataset_name)
            if best_config is not None:
                self.best_esm_variant = best_config['variant']
                self.best_esm_model_type = best_config['model']
                esm_ids = best_config['ids']
                X_esm = best_config['X']
                y = best_config['y']
            else:
                print("  Falling back to non-ESM ensemble")
                esm_ids = master_labels.index
                X_esm = None
                y = master_labels.values
        else:
            # Check if auto-generation is enabled
            if self.auto_generate_esm:
                print("\n3. ESM features not found - attempting to generate...")
                success = self._generate_esm_features(train_dir_path, is_test=False)
                if success:
                    # Retry detection after generation
                    self._detect_feature_paths(dataset_name)
                    if self.esm_path is not None:
                        best_config = self._run_esm_grid_search(master_labels, dataset_name)
                        if best_config is not None:
                            self.best_esm_variant = best_config['variant']
                            self.best_esm_model_type = best_config['model']
                            esm_ids = best_config['ids']
                            X_esm = best_config['X']
                            y = best_config['y']
                        else:
                            esm_ids = master_labels.index
                            X_esm = None
                            y = master_labels.values
                    else:
                        esm_ids = master_labels.index
                        X_esm = None
                        y = master_labels.values
                else:
                    esm_ids = master_labels.index
                    X_esm = None
                    y = master_labels.values
            else:
                print("\n3. Skipping ESM Grid Search (no ESM features found)")
                print("   Note: Set auto_generate_esm=True to auto-generate ESM features")
                esm_ids = master_labels.index
                X_esm = None
                y = master_labels.values
        
        # 4. Load K-mer features
        print("\n4. Loading K-mer features...")
        X_kmer = self._load_kmer_features(esm_ids if X_esm is not None else master_labels.index)
        
        # 5. Load Public Clone sequences
        print("\n5. Loading Public Clone sequences...")
        seqs_dict = self._load_public_sequences(train_dir_path, esm_ids if X_esm is not None else master_labels.index)
        
        # 6. Train ensemble with cross-validation
        print("\n6. Training Ensemble Models...")
        self._train_ensemble(X_kmer, seqs_dict, X_esm, y, esm_ids if X_esm is not None else np.array(master_labels.index))
        
        # 7. Identify important sequences (conditional)
        if self.rank_topseq:
            print("\n7. Identifying important sequences...")
            self.important_sequences_ = self.identify_associated_sequences(
                top_k=50000, 
                dataset_name=dataset_name
            )
        else:
            print("\n7. Skipping important sequences identification (rank_topseq disabled)")
            self.important_sequences_ = None
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        return self
    
    def _load_metadata_and_labels(self, train_dir_path):
        """Loads ground truth labels from metadata."""
        meta_path = os.path.join(train_dir_path, "metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}")
        
        df_meta = pd.read_csv(meta_path)
        if 'filename' in df_meta.columns:
            df_meta['ID'] = df_meta['filename'].str.replace('.tsv', '', regex=False)
        
        return df_meta.set_index('ID')['label_positive'].astype(int)
    
    def _detect_feature_paths(self, dataset_name):
        """Auto-detect feature file paths based on dataset name in output directory."""
        if self.out_dir is None:
            print(f"  Warning: No output directory specified, cannot detect features")
            return
        
        # K-mer path
        kmer_dir = os.path.join(self.out_dir, "kmer")
        kmer_pattern = os.path.join(kmer_dir, f"*{dataset_name}*.pkl")
        kmer_files = glob.glob(kmer_pattern)
        if kmer_files:
            self.kmer_path = kmer_files[0]
            print(f"  Found K-mer features: {self.kmer_path}")
        else:
            print(f"  K-mer features not found in {kmer_dir}")
            if self.auto_generate_esm:
                print(f"  Will generate K-mer features for {dataset_name}")
        
        # ESM path - look for aggregated features
        agg_dir = os.path.join(self.out_dir, "aggregates")
        if os.path.exists(agg_dir):
            # Try to find any aggregated ESM files for this dataset
            esm_variants = glob.glob(os.path.join(agg_dir, f"*/esm2_{dataset_name}_aggregated_*.pkl"))
            if esm_variants:
                self.esm_path = agg_dir
                print(f"  Found ESM features in: {self.esm_path}")
            else:
                print(f"  ESM features not found in {agg_dir}")
                if self.auto_generate_esm:
                    print(f"  Will generate ESM features for {dataset_name}")
        else:
            print(f"  ESM aggregates directory does not exist: {agg_dir}")
            if self.auto_generate_esm:
                print(f"  Will create directory and generate ESM features")
    
    def _detect_dataset_type(self, train_dir_path):
        """Detect if dataset is synthetic or experimental based on sequence count variability.
        
        Returns:
            str: 'synthetic' if repertoires have uniform sequence counts (CV < 0.05),
                 'experimental' if repertoires have varying sequence counts (CV >= 0.05)
        """
        print("\n  Detecting dataset type (synthetic vs experimental)...")
        
        seq_counts = []
        files = [f for f in os.listdir(train_dir_path) if f.endswith('.tsv')]
        
        # Sample up to 50 repertoires for analysis
        for file in files[:50]:
            try:
                fpath = os.path.join(train_dir_path, file)
                df = pd.read_csv(fpath, sep='\t')
                seq_counts.append(len(df))
            except:
                continue
        
        if not seq_counts:
            print("    Warning: Could not analyze sequence counts, defaulting to experimental")
            return 'experimental'
        
        mean_count = np.mean(seq_counts)
        std_count = np.std(seq_counts)
        cv = std_count / mean_count if mean_count > 0 else 0
        
        dataset_type = 'synthetic' if cv < 0.05 else 'experimental'
        
        print(f"    Sequence counts: mean={mean_count:.0f}, std={std_count:.0f}, CV={cv:.4f}")
        print(f"    Dataset Type: {dataset_type.upper()}")
        
        return dataset_type
    
    def _get_ml_model(self, model_name):
        """Returns the ML classifier configuration."""
        if model_name == "ExtraTrees_shallow":
            return ExtraTreesClassifier(
                n_estimators=300, 
                max_depth=6, 
                min_samples_leaf=5, 
                n_jobs=1,  # Set to 1 for reproducibility
                random_state=self.seed
            )
        elif model_name == "SVM_Linear":
            return SVC(
                kernel="linear", 
                C=1.0, 
                probability=True, 
                random_state=self.seed
            )
        return None
    
    def _run_esm_grid_search(self, master_labels, dataset_name):
        """Phase 1: Grid search over ESM variants."""
        # Define ESM variants
        agg_dir = self.esm_path
        
        # Bert [max, mean] x Row [mean, max, std, mean_std]
        esm_variants = {
            "BertMax_RowMean": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_max/esm2_{dataset_name}_aggregated_mean.pkl"),
            "BertMax_RowMax": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_max/esm2_{dataset_name}_aggregated_max.pkl"),
            "BertMax_RowStd": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_max/esm2_{dataset_name}_aggregated_std.pkl"),
            "BertMax_RowMeanStd": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_max/esm2_{dataset_name}_aggregated_mean_std.pkl"),
            "BertMean_RowMean": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_mean/esm2_{dataset_name}_aggregated_mean.pkl"),
            "BertMean_RowMax": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_mean/esm2_{dataset_name}_aggregated_max.pkl"),
            "BertMean_RowStd": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_mean/esm2_{dataset_name}_aggregated_std.pkl"),
            "BertMean_RowMeanStd": os.path.join(agg_dir, f"aggregated_esm2_t6_8M_mean/esm2_{dataset_name}_aggregated_mean_std.pkl")
        }
        
        results = []
        for variant_name, path in esm_variants.items():
            ids, X, y = self._load_esm_data(path, master_labels)
            if X is None: continue
            
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            
            for model_name in ["ExtraTrees_shallow", "SVM_Linear"]:
                print(f"  Testing [{variant_name}] + [{model_name}] ... ", end="")
                fold_aucs = []
                
                for train_idx, val_idx in kf.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    clf = self._get_ml_model(model_name)
                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train)
                    X_val_sc = scaler.transform(X_val)
                    
                    clf.fit(X_train_sc, y_train)
                    preds = clf.predict_proba(X_val_sc)[:, 1]
                    fold_aucs.append(roc_auc_score(y_val, preds))
                
                avg_auc = np.mean(fold_aucs)
                print(f"AUC: {avg_auc:.4f}")
                
                results.append({
                    "variant": variant_name,
                    "model": model_name,
                    "auc": avg_auc,
                    "path": path,
                    "ids": ids,
                    "X": X,
                    "y": y
                })
        
        if len(results) == 0:
            print("\n  Warning: No ESM features could be loaded. Skipping ESM grid search.")
            return None
        
        best = sorted(results, key=lambda x: x['auc'], reverse=True)[0]
        print(f"\n>>> WINNER: {best['variant']} using {best['model']} (AUC: {best['auc']:.4f})")
        return best
    
    def _load_esm_data(self, path, master_labels):
        """Loads a specific ESM pickle and aligns it to master labels."""
        if not os.path.exists(path):
            print(f"    Warning: File not found: {path}")
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
    
    def _load_kmer_features(self, ids, min_kmer_count=20):
        """Load and align K-mer features with optional filtering."""
        # if self.kmer_path is None or not os.path.exists(self.kmer_path):
        #     print("  Warning: K-mer features not available, using dummy features")
        #     return np.random.rand(len(ids), 100)
        
        with open(self.kmer_path, "rb") as f:
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
        
        # List
        elif isinstance(kmer_raw, list):
            try:
                arr = np.array(kmer_raw)
                result_df = pd.DataFrame(arr, index=ids)
            except ValueError:
                # Variable length - use DataFrame
                result_df = pd.DataFrame(kmer_raw, index=ids)
        
        else:
            raise TypeError(f"Unexpected K-mer data type: {type(kmer_raw)}")
        
        # Apply K-mer count filtering
        if min_kmer_count > 0:
            # Calculate total counts per K-mer (across all samples)
            kmer_counts = result_df.sum(axis=0)
            # Keep only K-mers that appear at least min_kmer_count times
            valid_kmers = kmer_counts[kmer_counts >= min_kmer_count].index
            n_before = len(result_df.columns)
            result_df = result_df[valid_kmers]
            n_after = len(result_df.columns)
            print(f"  K-mer filtering: kept {n_after}/{n_before} features (min_count={min_kmer_count})")
        
        # Store feature names for alignment during prediction
        self.kmer_feature_names_ = result_df.columns.tolist()
        
        return result_df.values
    
    def _load_public_sequences(self, train_dir_path, ids):
        """Load public clone sequences."""
        seqs_dict = {}
        for pid in tqdm(ids, desc="  Loading sequences"):
            try:
                fpath = os.path.join(train_dir_path, f"{pid}.tsv")
                df = pd.read_csv(fpath, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
                seqs_dict[pid] = set(zip(df['junction_aa'], df['v_call'], df['j_call']))
            except:
                seqs_dict[pid] = set()
        return seqs_dict
    
    def _optimize_kmer_c(self, X_kmer, y, c_values):
        """Grid search to find optimal C parameter for K-mer Lasso model."""
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        best_c = c_values[0]
        best_auc = 0.0
        
        for c in c_values:
            fold_aucs = []
            
            for train_idx, val_idx in kf.split(X_kmer, y):
                X_train, X_val = X_kmer[train_idx], X_kmer[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_val_sc = scaler.transform(X_val)
                
                # Train model
                clf = LogisticRegression(
                    penalty='l1', C=c, solver='liblinear',
                    class_weight='balanced', random_state=self.seed
                )
                clf.fit(X_train_sc, y_train)
                
                # Evaluate
                preds = clf.predict_proba(X_val_sc)[:, 1]
                fold_aucs.append(roc_auc_score(y_val, preds))
            
            avg_auc = np.mean(fold_aucs)
            print(f"    C={c:.3f}: AUC={avg_auc:.4f}")
            
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_c = c
        
        return best_c
    
    def _train_ensemble(self, X_kmer, seqs_dict, X_esm, y, ids_array):
        """Train the ensemble with stacking and K-mer C optimization."""
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        # Grid search for best K-mer C value
        print("\n  Optimizing K-mer Lasso C parameter...")
        c_values = [1.0, 0.2, 0.1, 0.05, 0.03]
        best_c = self._optimize_kmer_c(X_kmer, y, c_values)
        print(f"  Best K-mer C: {best_c}")
        
        # Determine number of models
        n_models = 2 if X_esm is None else 3
        oof_preds = np.zeros((len(ids_array), n_models))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_kmer, y)):
            print(f"  Fold {fold+1}/{self.n_folds}")
            
            # A. K-mer Model with optimized C
            self.kmer_model = LogisticRegression(
                penalty='l1', C=best_c, solver='liblinear', 
                class_weight='balanced', random_state=self.seed
            )
            self.kmer_scaler = StandardScaler()
            Xk_tr = self.kmer_scaler.fit_transform(X_kmer[train_idx])
            Xk_val = self.kmer_scaler.transform(X_kmer[val_idx])
            self.kmer_model.fit(Xk_tr, y[train_idx])
            oof_preds[val_idx, 0] = self.kmer_model.predict_proba(Xk_val)[:, 1]
            
            # B. Public Clone Model
            self.public_model = PublicCloneModel()
            self.public_model.fit(seqs_dict, y[train_idx], ids_array[train_idx])
            oof_preds[val_idx, 1] = self.public_model.predict_proba(seqs_dict, ids_array[val_idx])
            
            # C. ESM Model (if available)
            if X_esm is not None:
                self.esm_model = self._get_ml_model(self.best_esm_model_type)
                self.esm_scaler = StandardScaler()
                Xe_tr = self.esm_scaler.fit_transform(X_esm[train_idx])
                Xe_val = self.esm_scaler.transform(X_esm[val_idx])
                self.esm_model.fit(Xe_tr, y[train_idx])
                oof_preds[val_idx, 2] = self.esm_model.predict_proba(Xe_val)[:, 1]
        
        # Train final meta-model on full data with optimized C
        self.kmer_scaler = StandardScaler()
        Xk_full = self.kmer_scaler.fit_transform(X_kmer)
        self.kmer_model = LogisticRegression(
            penalty='l1', C=best_c, solver='liblinear',
            class_weight='balanced', random_state=self.seed
        )
        self.kmer_model.fit(Xk_full, y)
        
        # Report K-mer model statistics
        n_active_features = np.sum(self.kmer_model.coef_ != 0)
        print(f"  K-mer model: {n_active_features} active features (C={best_c})")
        
        self.public_model = PublicCloneModel()
        self.public_model.fit(seqs_dict, y, ids_array)
        
        if X_esm is not None:
            self.esm_scaler = StandardScaler()
            Xe_full = self.esm_scaler.fit_transform(X_esm)
            self.esm_model = self._get_ml_model(self.best_esm_model_type)
            self.esm_model.fit(Xe_full, y)
        
        # Train meta-learner
        self.meta_model = LogisticRegression(
            penalty=None, 
            solver='lbfgs', 
            random_state=self.seed, 
            max_iter=1000,
            tol=1e-6
        )
        self.meta_model.fit(oof_preds, y)
        
        final_auc = roc_auc_score(y, self.meta_model.predict_proba(oof_preds)[:, 1])
        weights = self.meta_model.coef_[0]
        norm_w = weights / np.abs(weights).sum()
        
        print(f"\n  Final Ensemble CV AUC: {final_auc:.5f}")
        print(f"  Model Weights: K-mer={norm_w[0]:.2f}, Public={norm_w[1]:.2f}", end="")
        if X_esm is not None:
            print(f", ESM={norm_w[2]:.2f}")
        else:
            print()
        
        # Compute individual CV AUCs for each model
        model_names = ['K-mer', 'Public', 'ESM'] if X_esm is not None else ['K-mer', 'Public']
        individual_aucs = []
        for i in range(n_models):
            auc = roc_auc_score(y, oof_preds[:, i])
            individual_aucs.append(auc)
        
        print(f"\n  Individual Model CV AUCs:")
        for i, name in enumerate(model_names):
            print(f"    {name}: {individual_aucs[i]:.5f}")
        
        # Select ensemble strategy based on dataset type
        if self.dataset_type == 'synthetic':
            # SYNTHETIC: Use hybrid single-model selection
            print(f"\n  Dataset Type: SYNTHETIC - Using Hybrid Selection")
            
            if self.model_selection_method == 'weights':
                max_idx = np.argmax(np.abs(norm_w))
                print(f"  Selection Method: Weights")
                print(f"  Selected Model: {model_names[max_idx]} (weight={np.abs(norm_w[max_idx]):.2f}, CV AUC={individual_aucs[max_idx]:.5f})")
            elif self.model_selection_method == 'hybrid':
                # Check if any model has weight > 0.5
                max_weight_idx = np.argmax(np.abs(norm_w))
                max_weight = np.abs(norm_w[max_weight_idx])
                
                if max_weight > 0.5:
                    # Strong consensus - use weight-based selection
                    max_idx = max_weight_idx
                    print(f"  Selection Method: Hybrid (Weight > 0.5 threshold)")
                    print(f"  Selected Model: {model_names[max_idx]} (weight={max_weight:.2f}, CV AUC={individual_aucs[max_idx]:.5f})")
                else:
                    # No strong consensus - fall back to CV
                    max_idx = np.argmax(individual_aucs)
                    print(f"  Selection Method: Hybrid (No weight > 0.5, using CV)")
                    print(f"  Selected Model: {model_names[max_idx]} (CV AUC={individual_aucs[max_idx]:.5f}, weight={np.abs(norm_w[max_idx]):.2f})")
            else:  # 'cv'
                max_idx = np.argmax(individual_aucs)
                print(f"  Selection Method: CV AUC")
                print(f"  Selected Model: {model_names[max_idx]} (CV AUC={individual_aucs[max_idx]:.5f})")
            
            # Set simplified weights (only selected model gets 1.0)
            simplified_weights = np.zeros(n_models)
            simplified_weights[max_idx] = 1.0
            
            print(f"  Final Weights: {model_names[0]}={simplified_weights[0]:.2f}, {model_names[1]}={simplified_weights[1]:.2f}", end="")
            if X_esm is not None:
                print(f", {model_names[2]}={simplified_weights[2]:.2f}")
            else:
                print()
        
        else:
            # EXPERIMENTAL: Use weighted ensemble with positive weights only
            print(f"\n  Dataset Type: EXPERIMENTAL - Using Weighted Ensemble")
            
            # Filter out models with negative weights
            positive_mask = weights > 0
            
            if not np.any(positive_mask):
                print("  Warning: All weights are negative, using absolute values")
                positive_mask = np.ones(n_models, dtype=bool)
            
            # Normalize positive weights to sum to 1
            simplified_weights = np.zeros(n_models)
            positive_weights = np.abs(weights[positive_mask])
            simplified_weights[positive_mask] = positive_weights / positive_weights.sum()
            
            print(f"  Filtered Models (positive weights only):")
            for i, name in enumerate(model_names):
                if simplified_weights[i] > 0:
                    print(f"    {name}: weight={simplified_weights[i]:.3f}, CV AUC={individual_aucs[i]:.5f}")
                else:
                    print(f"    {name}: EXCLUDED (negative weight={norm_w[i]:.3f})")
            
            print(f"  Final Ensemble Weights: {model_names[0]}={simplified_weights[0]:.3f}, {model_names[1]}={simplified_weights[1]:.3f}", end="")
            if X_esm is not None:
                print(f", {model_names[2]}={simplified_weights[2]:.3f}")
            else:
                print()
        
        # Store simplified weights for prediction
        self.simplified_weights = simplified_weights
        
        self.train_ids = ids_array

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """
        Predicts probabilities for examples in the provided path.

        Args:
            test_dir_path (str): Path to the directory with test TSV files.

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call' columns.
        """
        print(f"Making predictions for data in {test_dir_path}...")
        
        if self.kmer_model is None:
            raise RuntimeError("The model has not been fitted yet. Please call `fit` first.")
        
        # Get test repertoire IDs
        repertoire_ids = get_repertoire_ids(test_dir_path)
        dataset_name = os.path.basename(test_dir_path)
        
        print(f"  Loading features for {len(repertoire_ids)} repertoires...")
        
        # 1. Load K-mer features for test data
        X_kmer_test = self._load_kmer_features_for_test(test_dir_path, repertoire_ids)
        
        # 2. Load public sequences for test data
        seqs_dict_test = self._load_public_sequences(test_dir_path, repertoire_ids)
        
        # 3. Load ESM features for test data (if available)
        X_esm_test = None
        if self.esm_model is not None:
            X_esm_test = self._load_esm_features_for_test(test_dir_path, repertoire_ids)
        
        # 4. Generate predictions from each model
        print(f"  Generating predictions...")
        
        # K-mer predictions
        Xk_scaled = self.kmer_scaler.transform(X_kmer_test)
        pred_kmer = self.kmer_model.predict_proba(Xk_scaled)[:, 1]
        
        # Public clone predictions
        pred_public = self.public_model.predict_proba(seqs_dict_test, repertoire_ids)
        
        # Combine predictions
        if X_esm_test is not None and self.esm_model is not None:
            Xe_scaled = self.esm_scaler.transform(X_esm_test)
            pred_esm = self.esm_model.predict_proba(Xe_scaled)[:, 1]
            stacked_preds = np.column_stack([pred_kmer, pred_public, pred_esm])
        else:
            stacked_preds = np.column_stack([pred_kmer, pred_public])
        
        # Use simplified weights (max weight only) instead of meta-model ensemble
        probabilities = np.dot(stacked_preds, self.simplified_weights)
        
        # Format output
        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [dataset_name] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })
        
        # Compatibility columns
        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0
        
        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
        
        print(f"  Prediction complete on {len(repertoire_ids)} repertoires.")
        return predictions_df
    
    def _load_kmer_features_for_test(self, test_dir_path, repertoire_ids):
        """Load K-mer features for test data and align with training features."""
        # Try to find pre-computed test k-mer features
        dataset_name = os.path.basename(test_dir_path)
        kmer_dir = os.path.join(self.out_dir, "kmer") if self.out_dir else "."
        test_kmer_pattern = os.path.join(kmer_dir, f"*{dataset_name}*.pkl")
        test_kmer_files = glob.glob(test_kmer_pattern)
        
        test_df = None
        
        if test_kmer_files:
            print(f"    Loading pre-computed K-mer features...")
            with open(test_kmer_files[0], "rb") as f:
                kmer_raw = pickle.load(f)
                
                # Handle different pickle formats
                # Test data: Usually a DataFrame directly
                if isinstance(kmer_raw, pd.DataFrame):
                    # Check if repertoire_ids are in the index
                    common_ids = kmer_raw.index.intersection(repertoire_ids)
                    if len(common_ids) > 0:
                        test_df = kmer_raw.loc[repertoire_ids]
                    else:
                        # IDs don't match - assume same order
                        print(f"    Warning: IDs don't match, assuming same order")
                        test_df = kmer_raw.copy()
                        test_df.index = repertoire_ids
                
                # Train data: Usually (DataFrame, ids) tuple
                elif isinstance(kmer_raw, tuple) and len(kmer_raw) >= 2:
                    # Extract features (first element) and IDs (second element)
                    features = kmer_raw[0]
                    stored_ids = kmer_raw[1]
                    
                    # Convert stored_ids to list/array if needed
                    if isinstance(stored_ids, pd.Index):
                        stored_ids = stored_ids.tolist()
                    elif isinstance(stored_ids, np.ndarray):
                        stored_ids = stored_ids.tolist()
                    
                    # Handle features based on type
                    if isinstance(features, pd.DataFrame):
                        test_df = features.loc[repertoire_ids]
                    elif isinstance(features, np.ndarray):
                        # Create DataFrame to align by IDs
                        df = pd.DataFrame(features, index=stored_ids)
                        test_df = df.loc[repertoire_ids]
                    elif isinstance(features, list):
                        # Assume features is list of arrays/vectors, one per repertoire
                        df = pd.DataFrame(features, index=stored_ids)
                        test_df = df.loc[repertoire_ids]
                    else:
                        raise TypeError(f"Unexpected feature type in tuple: {type(features)}")
                
                # Single element tuple
                elif isinstance(kmer_raw, tuple) and len(kmer_raw) == 1:
                    kmer_raw = kmer_raw[0]
                    if isinstance(kmer_raw, pd.DataFrame):
                        test_df = kmer_raw.loc[repertoire_ids]
                    elif isinstance(kmer_raw, np.ndarray):
                        test_df = pd.DataFrame(kmer_raw, index=repertoire_ids)
                    else:
                        raise TypeError(f"Unexpected data type in single-element tuple: {type(kmer_raw)}")
                
                # Direct array
                elif isinstance(kmer_raw, np.ndarray):
                    test_df = pd.DataFrame(kmer_raw, index=repertoire_ids)
                
                # List of features
                elif isinstance(kmer_raw, list):
                    # Check if it's a list with [features, ids] format (like tuple)
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
                            test_df = features.loc[repertoire_ids]
                        elif isinstance(features, (np.ndarray, list)):
                            df = pd.DataFrame(features, index=stored_ids)
                            test_df = df.loc[repertoire_ids]
                        else:
                            raise TypeError(f"Unexpected feature type in list: {type(features)}")
                    else:
                        # Try converting to array
                        try:
                            arr = np.array(kmer_raw)
                            test_df = pd.DataFrame(arr, index=repertoire_ids)
                        except ValueError:
                            # If conversion fails, features might be variable length
                            # Create DataFrame assuming repertoire_ids match order
                            test_df = pd.DataFrame(kmer_raw, index=repertoire_ids)
                
                else:
                    raise TypeError(f"Unexpected K-mer data type: {type(kmer_raw)}")
        else:
            # Compute K-mer features on the fly
            print(f"    Computing K-mer features on the fly...")
            from collections import Counter
            
            kmer_features = []
            for rep_id in tqdm(repertoire_ids, desc="    Computing K-mers"):
                try:
                    fpath = os.path.join(test_dir_path, f"{rep_id}.tsv")
                    df = pd.read_csv(fpath, sep='\t', usecols=['junction_aa'])
                    
                    kmer_counts = Counter()
                    for seq in df['junction_aa'].dropna():
                        # K=3 and K=4
                        for k in [3, 4]:
                            for i in range(len(seq) - k + 1):
                                kmer_counts[seq[i:i+k]] += 1
                    
                    kmer_features.append(kmer_counts)
                except:
                    kmer_features.append(Counter())
            
            # Convert to DataFrame
            test_df = pd.DataFrame(kmer_features, index=repertoire_ids).fillna(0)
        
        # Align test features with training features
        if self.kmer_feature_names_ is not None and test_df is not None:
            print(f"    Aligning features: test has {len(test_df.columns)}, train expects {len(self.kmer_feature_names_)}")
            
            # Add missing columns (features present in train but not in test)
            missing_cols = set(self.kmer_feature_names_) - set(test_df.columns)
            if missing_cols:
                # Create DataFrame with missing columns all at once
                missing_df = pd.DataFrame(0, index=test_df.index, columns=list(missing_cols))
                test_df = pd.concat([test_df, missing_df], axis=1)
            
            # Reorder to match training features (also removes extra columns)
            test_df = test_df[self.kmer_feature_names_]
            
            return test_df.values
        elif test_df is not None:
            return test_df.values
        else:
            raise RuntimeError("Failed to load K-mer features for test data")
    
    def _load_esm_features_for_test(self, test_dir_path, repertoire_ids):
        """Load ESM features for test data."""
        if self.esm_path is None or self.best_esm_variant is None:
            return None
        
        dataset_name = os.path.basename(test_dir_path)
        
        # Construct expected path
        if "BertMax" in self.best_esm_variant:
            bert_pool = "max"
        else:
            bert_pool = "mean"
        
        # Check for RowMeanStd BEFORE RowMean/RowStd to avoid incorrect matches
        if "RowMeanStd" in self.best_esm_variant:
            row_pool = "mean_std"
        elif "RowSum" in self.best_esm_variant:
            row_pool = "sum"
        elif "RowMean" in self.best_esm_variant:
            row_pool = "mean"
        elif "RowStd" in self.best_esm_variant:
            row_pool = "std"
        else:
            row_pool = "max"
        
        esm_test_path = os.path.join(
            self.esm_path,
            f"aggregated_esm2_t6_8M_{bert_pool}",
            f"esm2_{dataset_name}_aggregated_{row_pool}.pkl"
        )
        
        if not os.path.exists(esm_test_path):
            print(f"    Warning: ESM test features not found at {esm_test_path}")
            return None
        
        print(f"    Loading ESM features from {esm_test_path}")
        with open(esm_test_path, "rb") as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, (tuple, list)) and len(data) >= 1:
            df = pd.DataFrame(data[0], index=repertoire_ids)
        else:
            df = pd.DataFrame(data, index=repertoire_ids)
        
        return df.loc[repertoire_ids].values

    def _generate_esm_features(self, data_dir_path: str, is_test: bool = False):
        """
        Generate ESM2 embeddings if they don't exist.
        This is a wrapper that calls save_representation.py and aggregate_allpooling.py
        
        Args:
            data_dir_path: Path to the dataset directory
            is_test: Whether this is a test dataset
        """
        dataset_name = os.path.basename(data_dir_path)
        
        print(f"\n  {'='*56}")
        print(f"  Generating ESM2 Features for {dataset_name}")
        print(f"  {'='*56}")
        
        # Step 1: Generate raw ESM2 embeddings
        print(f"  Step 1: Extracting ESM2 representations...")
        try:
            import subprocess
            import sys
            
            # Run save_representation.py
            repr_script = os.path.join(self.base_dir, "save_representation.py")
            if os.path.exists(repr_script):
                cmd = [
                    sys.executable, repr_script,
                    "--data_dir", data_dir_path,
                    "--output_dir", os.path.join(self.base_dir, "workingfolder/representations/"),
                    "--model_name", self.esm_model_name,
                    "--batch_size", str(self.esm_batch_size),
                    "--device", self.device
                ]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  ✓ ESM2 extraction complete")
            else:
                print(f"  Warning: save_representation.py not found at {repr_script}")
                print(f"  Skipping ESM generation - will continue with K-mer + Public only")
                return False
                
        except Exception as e:
            print(f"  Error generating ESM representations: {e}")
            print(f"  Continuing without ESM features...")
            return False
        
        # Step 2: Aggregate embeddings with different pooling strategies
        print(f"  Step 2: Aggregating embeddings with pooling strategies...")
        try:
            agg_script = os.path.join(self.base_dir, "aggregate_allpooling.py")
            if os.path.exists(agg_script):
                cmd = [
                    sys.executable, agg_script,
                    "--dataset_name", dataset_name,
                    "--dataset_type", "test" if is_test else "train",
                    "--data_dir", data_dir_path,
                    "--repr_dir", os.path.join(self.base_dir, "workingfolder/representations/"),
                    "--output_dir", os.path.join(self.base_dir, "workingfolder/aggregates/")
                ]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  ✓ Aggregation complete")
            else:
                print(f"  Warning: aggregate_allpooling.py not found at {agg_script}")
                return False
                
        except Exception as e:
            print(f"  Error aggregating embeddings: {e}")
            return False
        
        # Update ESM path now that features are generated
        self.esm_path = os.path.join(self.out_dir, "aggregates") if self.out_dir else "aggregates"
        print(f"  ✓ ESM features successfully generated!")
        return True

    def identify_associated_sequences(self, dataset_name: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Identifies the top "k" important sequences (rows) from the training data that best explain the labels.

        Args:
            dataset_name (str): Name of the dataset
            top_k (int): The number of top sequences to return (based on some scoring mechanism).

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call' columns.
        """
        print(f"  Identifying top {top_k} important sequences...")
        
        # Use the public clone model's selected sequences as important sequences
        if self.public_model is not None and len(self.public_model.selected) > 0:
            selected_seqs = self.public_model.selected[:top_k]
            
            # Convert tuples back to columns
            sequences_data = []
            for seq_tuple in selected_seqs:
                if isinstance(seq_tuple, tuple) and len(seq_tuple) == 3:
                    sequences_data.append({
                        'junction_aa': seq_tuple[0],
                        'v_call': seq_tuple[1],
                        'j_call': seq_tuple[2]
                    })
                else:
                    sequences_data.append({
                        'junction_aa': str(seq_tuple),
                        'v_call': 'unknown',
                        'j_call': 'unknown'
                    })
            
            top_sequences_df = pd.DataFrame(sequences_data)
        else:
            # Fallback: generate placeholder sequences
            print("    Warning: No public sequences selected, using placeholder")
            top_sequences_df = pd.DataFrame({
                'junction_aa': [f'CASSXX{i}EQFF' for i in range(min(top_k, 1000))],
                'v_call': ['TRBV1-1*01'] * min(top_k, 1000),
                'j_call': ['TRBJ1-1*01'] * min(top_k, 1000)
            })
        
        # Format output
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df) + 1)
        top_sequences_df['ID'] = top_sequences_df['dataset'] + '_seq_top_' + top_sequences_df['ID'].astype(str)
        top_sequences_df['label_positive_probability'] = -999.0
        
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
        
        print(f"    Identified {len(top_sequences_df)} important sequences")
        return top_sequences_df
