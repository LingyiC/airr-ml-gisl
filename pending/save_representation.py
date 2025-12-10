import os
import glob
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: 'transformers' library not found. ESM option will not work.")


# ==========================================
# 1. REPRODUCIBILITY SETUP
# ==========================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(torch.cuda.is_available())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")

# ==========================================
# 2. CONFIGURATION
# ==========================================

class NotebookConfig:
    # --- PATHS ---
    train_dir = "/Users/lingyi/Documents/airr-ml/data/train_datasets"
    test_dirs = ["/Users/lingyi/Documents/airr-ml/data/test_datasets"]
    representation_out_dir = "/Users/lingyi/Documents/airr-ml/predict-airr-main/workingfolder/representations/"

    # --- DATA SELECTION ---
    # target_ids = [1, 5, 7, 8]  # Target Datasets 1 and 5
    target_ids = [1, 2, 3, 4]

    # --- MODEL ---
    model_name = "facebook/esm2_t6_8M_UR50D"
    batch_size = 128
    # pooling can be: 'cls', 'mean', 'max', or 'all'
    # Use 'all' to compute cls, mean, and max in one pass.
    pooling = "all"

    # --- DEBUG / SAMPLING ---
    # DEBUG MODE: 1 sequence per patient/repertoire
    debug = False
    debug_frac = 0.1
    debug_seq_limit = 1

    # --- RUNTIME ---
    device = "cuda"
    seed = 42             

args = NotebookConfig()
os.makedirs(args.representation_out_dir, exist_ok=True) 

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================

def load_data_generator(data_dir: str, metadata_filename='metadata.csv', debug=False, debug_frac=0.5):
    """
    Reads raw TSV files. 
    """
    metadata_path = os.path.join(data_dir, metadata_filename)
    files_to_process = []
    
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        if debug:
            # Patient/Repertoire Subsampling (if metadata is used)
            metadata_df = metadata_df.groupby('label_positive', group_keys=False).apply(
                lambda x: x.sample(frac=min(1.0, debug_frac), random_state=args.seed)
            ).reset_index(drop=True)
            
        for row in metadata_df.itertuples(index=False):
            files_to_process.append((row.filename, row.repertoire_id, row.label_positive))
    else:
        all_tsvs = sorted(glob.glob(os.path.join(data_dir, '*.tsv')))
        if debug:
            n_samples = max(1, int(len(all_tsvs) * min(1.0, debug_frac * 400)))
            random.seed(args.seed) 
            all_tsvs = random.sample(all_tsvs, n_samples)

        for f in all_tsvs:
            fname = os.path.basename(f)
            rep_id = fname.replace('.tsv', '')
            files_to_process.append((fname, rep_id, None))

    for fname, rep_id, label in files_to_process:
        file_path = os.path.join(data_dir, fname)
        try:
            cols = ['junction_aa', 'v_call', 'j_call']
            df = pd.read_csv(file_path, sep='\t', usecols=cols)
            yield rep_id, df, label
        except Exception as e:
            print(f"Warning: Failed to load {fname}: {e}")
            continue

def _save_representations(dataset_name: str, rep_id: str, embeddings, is_test: bool):
    """
    Saves the final sequence representations (320D) to a compressed .npz file.

    If `embeddings` is:
      - np.ndarray: saved as 'embeddings'
      - dict[str, np.ndarray]: saved with each key as a separate array (e.g. 'cls', 'mean', 'max')
    """
    mode = "test" if is_test else "train"
    
    # Create dataset-specific subdirectory
    dataset_folder = os.path.join(args.representation_out_dir, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    
    filename = f"{rep_id}_embeddings.npz"
    filepath = os.path.join(dataset_folder, filename)
    
    if isinstance(embeddings, dict):
        np.savez_compressed(filepath, **embeddings)
    else:
        np.savez_compressed(filepath, embeddings=embeddings)

    print(f"   💾 Saved embeddings for {rep_id} to {filepath}")

# ==========================================
# 4. MODEL ARCHITECTURES
# ==========================================

class PretrainedEncoder(torch.nn.Module):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", batch_size=32, pooling="cls"):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers not found.")

        print(f"Loading pretrained model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pretrained_model = AutoModel.from_pretrained(model_name)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        hidden_size = self.pretrained_model.config.hidden_size

        # Kept for reference but bypassed in forward()
        self.fc = nn.Linear(hidden_size, 128)
        self.norm = nn.LayerNorm(128)
        self.batch_size = batch_size

        # Pooling strategy: 'cls', 'mean', 'max', or 'all'
        if pooling not in ['cls', 'mean', 'max', 'all']:
            raise ValueError(f"pooling must be 'cls', 'mean', 'max', or 'all', got '{pooling}'")
        self.pooling = pooling

        print(f"✅ Encoder configured with {pooling} pooling, output dimension: {hidden_size}D")


    def forward(self, sequences: list):
        """
        Returns embeddings based on the specified pooling strategy.

        Pooling options:
        - 'cls': Use CLS token (first token)
        - 'mean': Mean pooling over sequence (excluding special tokens)
        - 'max': Max pooling over sequence (excluding special tokens)
        - 'all': Returns dict with 'cls', 'mean', and 'max' from a single forward pass
        """

        if self.pooling == "all":
            cls_batches = []
            mean_batches = []
            max_batches = []
        else:
            all_embeddings = []

        for i in range(0, len(sequences), self.batch_size):
            batch_seqs = sequences[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=40
            )
            inputs = {k: v.to(self.pretrained_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.pretrained_model(**inputs)

            hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            attention_mask = inputs['attention_mask']

            # Build mask excluding special tokens: CLS (first) and EOS (last actual token)
            mask = attention_mask.clone()
            mask[:, 0] = 0  # Exclude CLS

            for j in range(mask.size(0)):
                seq_len = attention_mask[j].sum()
                if seq_len > 1:
                    mask[j, seq_len - 1] = 0  # Exclude EOS

            if self.pooling == "cls":
                embeddings = hidden_states[:, 0, :]

                all_embeddings.append(embeddings)

            elif self.pooling == "mean":
                masked_hidden = hidden_states * mask.unsqueeze(-1)
                sum_hidden = masked_hidden.sum(dim=1)
                embeddings = sum_hidden / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

                all_embeddings.append(embeddings)

            elif self.pooling == "max":
                masked_hidden = hidden_states.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
                embeddings, _ = masked_hidden.max(dim=1)

                all_embeddings.append(embeddings)

            elif self.pooling == "all":
                # CLS
                cls_emb = hidden_states[:, 0, :]

                # Mean pooling
                masked_hidden_mean = hidden_states * mask.unsqueeze(-1)
                sum_hidden = masked_hidden_mean.sum(dim=1)
                mean_emb = sum_hidden / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

                # Max pooling
                masked_hidden_max = hidden_states.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
                max_emb, _ = masked_hidden_max.max(dim=1)

                cls_batches.append(cls_emb)
                mean_batches.append(mean_emb)
                max_batches.append(max_emb)

        if self.pooling == "all":
            return {
                "cls": torch.cat(cls_batches, dim=0),
                "mean": torch.cat(mean_batches, dim=0),
                "max": torch.cat(max_batches, dim=0),
            }
        else:
            H_raw = torch.cat(all_embeddings, dim=0)
            return H_raw

# ==========================================
# 5. REPRESENTATION EXTRACTOR LOGIC
# ==========================================

class RepresentationExtractor:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
        self.encoder = self._get_encoder()

    def _get_encoder(self):
        encoder = PretrainedEncoder(
            model_name=self.cfg.model_name,
            batch_size=self.cfg.batch_size,
            pooling=self.cfg.pooling
        ).to(self.device)
        encoder.eval()
        return encoder

    def _process_bag(self, seqs: list):
        """
        Applies debug sampling/limiting per repertoire.
        *** 50,000 sequence cap removed here. ***
        """
        if self.cfg.debug and len(seqs) > self.cfg.debug_seq_limit:
            # If debug=True, sample only 'debug_seq_limit' sequences (e.g., 1)
            seqs = random.sample(seqs, self.cfg.debug_seq_limit)
        
        # The 50000 cap logic was removed. If debug=False, all sequences are returned.
        return seqs 

    def extract_representations(self, data_dir_path: str, is_test: bool):
        dataset_name = os.path.basename(data_dir_path)
        
        print(f"\n=== Extracting Representations for {dataset_name} (Test={is_test}) ===")
        
        # Create dataset-specific subdirectory
        dataset_folder = os.path.join(self.cfg.representation_out_dir, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        
        data_gen = load_data_generator(
            data_dir_path, 
            metadata_filename='metadata.csv' if not is_test else 'dummy', 
            debug=self.cfg.debug, 
            debug_frac=self.cfg.debug_frac
        )
        
        for rep_id, df, _ in tqdm(data_gen, desc="Processing Repertoires"):
            # Check if embeddings already exist
            expected_filepath = os.path.join(dataset_folder, f"{rep_id}_embeddings.npz")
            if os.path.exists(expected_filepath):
                print(f"   ⏭️  Skipping {rep_id} (already exists)")
                continue
            
            if len(df) == 0:
                continue
            
            seq_list = df['junction_aa'].dropna().tolist()
            if not seq_list:
                continue

            processed_seqs = self._process_bag(seq_list)

            # Check if processing will be very large (advisory print)
            if not self.cfg.debug and len(processed_seqs) > 100000:
                 print(f"   [INFO] Processing large repertoire {rep_id} with {len(processed_seqs):,} sequences.")

            with torch.no_grad():
                H = self.encoder(processed_seqs)

            # Convert to numpy (handles both tensor and dict-of-tensors)
            if isinstance(H, dict):
                H_numpy = {k: v.cpu().numpy() for k, v in H.items()}
            else:
                H_numpy = H.cpu().numpy()
            
            _save_representations(dataset_name, rep_id, H_numpy, is_test)

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    if not TRANSFORMERS_AVAILABLE:
        print("🛑 Cannot run without 'transformers' library. Please install it.")
        exit()
        
    seed_everything(args.seed)
    
    extractor = RepresentationExtractor(args)
    
    # --- 1. PROCESS TRAINING DATASETS ---
    train_folders = sorted(glob.glob(os.path.join(args.train_dir, "train_dataset_*")))
    
    filtered_train_folders = []
    for f in train_folders:
        try:
            curr_id = int(os.path.basename(f).split('_')[-1])
            if curr_id in args.target_ids:
                filtered_train_folders.append(f)
        except ValueError:
            continue
    train_folders = filtered_train_folders
    
    print(f"🎯 Targeted TRAIN Datasets: {[os.path.basename(f) for f in train_folders]}")

    for train_path in train_folders:
        extractor.extract_representations(train_path, is_test=False)

    # --- 2. PROCESS TEST DATASETS ---
    test_folders = []
    
    # Iterate through target IDs (1, 5, 7, 8)
    for target_id in args.target_ids:
        suffix = str(target_id)
        # Construct the general test dataset name (e.g., 'test_dataset_7')
        dataset_base_name = f"test_dataset_{suffix}"
        
        # Search for all folders/files that start with this base name in the test directories
        for td in args.test_dirs: # td is usually ['/home/ccdd/Documents/airr-ml/data/test_datasets']
            
            # This line finds sub-dirs like test_dataset_7_1, test_dataset_7_2
            potential_sub_dirs = glob.glob(os.path.join(td, f"{dataset_base_name}_*"))
            
            # This checks for the base folder itself (e.g., test_dataset_7)
            base_dir = os.path.join(td, dataset_base_name)
            if os.path.isdir(base_dir):
                test_folders.append(base_dir)
            
            # Add all sub-directories found
            test_folders.extend(potential_sub_dirs)
            
    print(f"\n🎯 Targeted TEST Datasets: {[os.path.basename(f) for f in test_folders]}")
    
    for test_path in test_folders:
        extractor.extract_representations(test_path, is_test=True)

    print("\n✅ Representation Extraction Complete.")
    print(f"Find your 320D representations in: {args.representation_out_dir}")
