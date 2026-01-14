"""
Generate ESM2 representations for AIRR datasets.
This script loads sequences from train/test datasets and generates embeddings using ESM2 models.
"""

import os
import glob
import random
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
# REPRODUCIBILITY SETUP
# ==========================================

def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
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
# CONFIGURATION
# ==========================================

class RepresentationConfig:
    """Configuration for representation generation."""
    
    # Paths - should be set by the calling code
    train_dir = None  # Set via arguments
    test_dirs = None  # Set via arguments
    representation_out_dir = None  # Set via arguments

    # Model settings
    model_name = "facebook/esm2_t6_8M_UR50D"
    batch_size = 128
    pooling = ["mean", "max"]  # Options: 'cls', 'mean', 'max', list of methods, or 'all'
    
    @property
    def model_identifier(self):
        """Extract model identifier from model name (e.g., 't6_8M' from 'facebook/esm2_t6_8M_UR50D')."""
        # Extract the part between 'esm2_' and '_UR' or end of string
        import re
        match = re.search(r'esm2_([^_/]+_[^_/]+)', self.model_name)
        if match:
            return match.group(1)
        return "t6_8M"  # fallback default

    # Debug settings
    debug = False
    debug_frac = 0.1
    debug_seq_limit = 1

    # Runtime
    device = "cuda"
    num_gpus = -1  # Number of GPUs to use (1 = single GPU, >1 = DataParallel multi-GPU, -1 = all available GPUs)
    seed = 42


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def load_data_generator(data_dir: str, metadata_filename='metadata.csv', debug=False, debug_frac=0.5, seed=42):
    """
    Generator that yields (repertoire_id, dataframe, label) for each repertoire.
    
    Args:
        data_dir: Path to dataset directory
        metadata_filename: Name of metadata file
        debug: Whether to use debug mode
        debug_frac: Fraction of data to use in debug mode
        seed: Random seed
    
    Yields:
        Tuple of (repertoire_id, dataframe, label)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)
    files_to_process = []
    
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        if debug:
            metadata_df = metadata_df.groupby('label_positive', group_keys=False).apply(
                lambda x: x.sample(frac=min(1.0, debug_frac), random_state=seed)
            ).reset_index(drop=True)
            
        for row in metadata_df.itertuples(index=False):
            files_to_process.append((row.filename, row.repertoire_id, row.label_positive))
    else:
        all_tsvs = sorted(glob.glob(os.path.join(data_dir, '*.tsv')))
        if debug:
            n_samples = max(1, int(len(all_tsvs) * min(1.0, debug_frac * 400)))
            random.seed(seed) 
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


def save_representations(dataset_name: str, rep_id: str, embeddings, output_dir: str):
    """
    Save sequence representations to a compressed .npz file.
    
    Args:
        dataset_name: Name of the dataset
        rep_id: Repertoire ID
        embeddings: Embeddings (can be numpy array or dict of arrays)
        output_dir: Base output directory
    """
    dataset_folder = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    
    filename = f"{rep_id}_embeddings.npz"
    filepath = os.path.join(dataset_folder, filename)
    
    if isinstance(embeddings, dict):
        np.savez_compressed(filepath, **embeddings)
    else:
        np.savez_compressed(filepath, embeddings=embeddings)

    print(f"   💾 Saved embeddings for {rep_id} to {filepath}")


# ==========================================
# MODEL
# ==========================================

class PretrainedEncoder(torch.nn.Module):
    """ESM2 encoder with multiple pooling strategies."""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", batch_size=32, pooling="cls", num_gpus=1):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not found.")

        print(f"Loading pretrained model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pretrained_model = AutoModel.from_pretrained(model_name)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        hidden_size = self.pretrained_model.config.hidden_size
        self.fc = nn.Linear(hidden_size, 128)
        self.norm = nn.LayerNorm(128)
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        # Validate pooling
        if isinstance(pooling, str):
            if pooling not in ['cls', 'mean', 'max', 'all']:
                raise ValueError(f"pooling must be 'cls', 'mean', 'max', or 'all', got '{pooling}'")
            self.pooling = pooling
        elif isinstance(pooling, list):
            valid_methods = ['cls', 'mean', 'max']
            for method in pooling:
                if method not in valid_methods:
                    raise ValueError(f"Each pooling method must be in {valid_methods}, got '{method}'")
            self.pooling = pooling
        else:
            raise ValueError(f"pooling must be a string or list, got {type(pooling)}")

        print(f"✅ Encoder configured with {pooling} pooling, output dimension: {hidden_size}D")
        
        # Multi-GPU setup
        if num_gpus > 1 or num_gpus == -1:
            gpu_count = torch.cuda.device_count()
            if num_gpus == -1:
                num_gpus = gpu_count
            if gpu_count > 1:
                print(f"🔥 Using DataParallel with {min(num_gpus, gpu_count)} GPUs")
                self.pretrained_model = nn.DataParallel(self.pretrained_model, device_ids=list(range(min(num_gpus, gpu_count))))
                self.batch_size = batch_size * min(num_gpus, gpu_count)  # Scale batch size with number of GPUs
            else:
                print(f"⚠️  Only 1 GPU available, using single GPU mode")

    def forward(self, sequences: list):
        """
        Generate embeddings for sequences.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Embeddings (tensor or dict of tensors based on pooling strategy)
        """
        # Determine which pooling methods to compute
        if self.pooling == "all":
            methods_to_compute = ['cls', 'mean', 'max']
        elif isinstance(self.pooling, list):
            methods_to_compute = self.pooling
        else:
            methods_to_compute = [self.pooling]
        
        # Initialize storage for each method
        if len(methods_to_compute) > 1:
            batches = {method: [] for method in methods_to_compute}
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

            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']

            # Build mask excluding special tokens
            mask = attention_mask.clone()
            mask[:, 0] = 0  # Exclude CLS

            for j in range(mask.size(0)):
                seq_len = attention_mask[j].sum()
                if seq_len > 1:
                    mask[j, seq_len - 1] = 0  # Exclude EOS

            # Compute requested pooling methods
            for method in methods_to_compute:
                if method == "cls":
                    emb = hidden_states[:, 0, :]
                elif method == "mean":
                    masked_hidden = hidden_states * mask.unsqueeze(-1)
                    sum_hidden = masked_hidden.sum(dim=1)
                    emb = sum_hidden / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
                elif method == "max":
                    masked_hidden = hidden_states.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
                    emb, _ = masked_hidden.max(dim=1)
                
                if len(methods_to_compute) > 1:
                    batches[method].append(emb)
                else:
                    all_embeddings.append(emb)

        # Return results
        if len(methods_to_compute) > 1:
            return {method: torch.cat(batches[method], dim=0) for method in methods_to_compute}
        else:
            H_raw = torch.cat(all_embeddings, dim=0)
            return H_raw


# ==========================================
# REPRESENTATION EXTRACTOR
# ==========================================

class RepresentationExtractor:
    """Main class for extracting representations from AIRR datasets."""
    
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
        self.encoder = self._get_encoder()

    def _get_encoder(self):
        """Initialize and return encoder model."""
        encoder = PretrainedEncoder(
            model_name=self.cfg.model_name,
            batch_size=self.cfg.batch_size,
            pooling=self.cfg.pooling,
            num_gpus=self.cfg.num_gpus
        ).to(self.device)
        encoder.eval()
        return encoder

    def _process_bag(self, seqs: list):
        """Apply debug sampling if needed."""
        if self.cfg.debug and len(seqs) > self.cfg.debug_seq_limit:
            seqs = random.sample(seqs, self.cfg.debug_seq_limit)
        return seqs 

    def extract_representations(self, data_dir_path: str, is_test: bool):
        """
        Extract representations for a dataset.
        
        Args:
            data_dir_path: Path to dataset directory
            is_test: Whether this is a test dataset
        """
        dataset_name = os.path.basename(data_dir_path)
        
        print(f"\n=== Extracting Representations for {dataset_name} (Test={is_test}) ===")
        
        dataset_folder = os.path.join(self.cfg.representation_out_dir, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        
        data_gen = load_data_generator(
            data_dir_path, 
            metadata_filename='metadata.csv' if not is_test else 'dummy', 
            debug=self.cfg.debug, 
            debug_frac=self.cfg.debug_frac,
            seed=self.cfg.seed
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

            if not self.cfg.debug and len(processed_seqs) > 100000:
                 print(f"   [INFO] Processing large repertoire {rep_id} with {len(processed_seqs):,} sequences.")

            with torch.no_grad():
                H = self.encoder(processed_seqs)

            # Convert to numpy
            if isinstance(H, dict):
                H_numpy = {k: v.cpu().numpy() for k, v in H.items()}
            else:
                H_numpy = H.cpu().numpy()
            
            save_representations(dataset_name, rep_id, H_numpy, self.cfg.representation_out_dir)


def generate_for_dataset(dataset_num, dataset_type="train", config=None):
    """
    Generate representations for a specific dataset.
    
    Args:
        dataset_num: Dataset number (e.g., "1", "7_1")
        dataset_type: "train" or "test"
        config: RepresentationConfig instance (uses default if None)
    """
    if config is None:
        config = RepresentationConfig()
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Cannot run without 'transformers' library.")
    
    seed_everything(config.seed)
    extractor = RepresentationExtractor(config)
    
    if dataset_type == "train":
        data_path = os.path.join(config.train_dir, f"train_dataset_{dataset_num}")
    else:
        data_path = os.path.join(config.test_dirs[0], f"test_dataset_{dataset_num}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    extractor.extract_representations(data_path, is_test=(dataset_type == "test"))
    print(f"\n✅ Representation generation complete for {dataset_type}_dataset_{dataset_num}")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    import sys
    
    if not TRANSFORMERS_AVAILABLE:
        print("🛑 Cannot run without 'transformers' library. Please install it.")
        sys.exit(1)
    
    args = RepresentationConfig()
    seed_everything(args.seed)
    
    extractor = RepresentationExtractor(args)
    
    # Process training datasets
    train_folders = sorted(glob.glob(os.path.join(args.train_dir, "train_dataset_*")))
    
    print(f"🎯 Processing TRAIN Datasets: {[os.path.basename(f) for f in train_folders]}")

    for train_path in train_folders:
        extractor.extract_representations(train_path, is_test=False)

    # Process test datasets
    test_folders = []
    for td in args.test_dirs:
        # Find all test_dataset_* directories
        test_dirs_pattern = glob.glob(os.path.join(td, "test_dataset_*"))
        test_folders.extend([f for f in test_dirs_pattern if os.path.isdir(f)])
    
    test_folders = sorted(test_folders)
    print(f"\n🎯 Processing TEST Datasets: {[os.path.basename(f) for f in test_folders]}")
    
    for test_path in test_folders:
        extractor.extract_representations(test_path, is_test=True)

    print(f"\n✅ Representation Extraction Complete.")
    print(f"Find your representations in: {args.representation_out_dir}")
