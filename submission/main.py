import os

# Set PYTHONHASHSEED before other imports for reproducibility hash randomization?
os.environ['PYTHONHASHSEED'] = '0'

import argparse
import pandas as pd
import numpy as np
import subprocess
import sys
import glob
from typing import List
from submission.predictor import ImmuneStatePredictor
from submission.utils import save_tsv, validate_dirs_and_files
from submission.reproduce_checker import check_reproducibility, get_reproduce_script_path
from submission.generate_representations import generate_for_dataset, RepresentationConfig
from submission.aggregate_representations import aggregate_for_dataset, AggregateConfig
from submission.generate_kmers import ensure_kmer_features_exist


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


def _save_model(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Saves the trained ensemble model to a pickle file."""
    import pickle
    
    model_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_ensemble_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(predictor, f)
    print(f"Trained model saved to `{model_path}`.")


def _check_aggregated_features_exist(dataset_name: str, aggregate_dir: str) -> bool:
    """Check if aggregated features exist for a dataset."""
    # Check for at least one aggregated pickle file
    pattern = os.path.join(aggregate_dir, f"*/esm2_{dataset_name}_aggregated_*.pkl")
    files = glob.glob(pattern)
    return len(files) > 0


def _generate_representations_direct(data_dir: str, dataset_type: str, config: RepresentationConfig):
    """Generate representations using the actual data directory path."""
    from submission.generate_representations import RepresentationExtractor, seed_everything
    
    seed_everything(config.seed)
    extractor = RepresentationExtractor(config)
    extractor.extract_representations(data_dir, is_test=(dataset_type == "test"))


def _aggregate_representations_direct(data_dir: str, dataset_name: str, dataset_type: str, config: AggregateConfig):
    """Aggregate representations using the actual data directory path."""
    from submission.aggregate_representations import apply_pooling
    import pickle
    
    features_dir = os.path.join(config.representation_dir, dataset_name)
    
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # Check what files are actually in the features directory
    npz_files = glob.glob(os.path.join(features_dir, "*.npz"))
    print(f"  Found {len(npz_files)} .npz files in {features_dir}")
    if len(npz_files) > 0:
        print(f"    Example files: {[os.path.basename(f) for f in npz_files[:3]]}")
    
    # Parse dataset number for test datasets (needed for sample_submissions.csv)
    # For custom dataset names, we'll skip sample_submissions lookup
    dataset_num = None
    if dataset_name.startswith("test_dataset_"):
        dataset_num = dataset_name.replace("test_dataset_", "")
    
    # Collect all npz files to remove (only after ALL aggregations are done)
    all_npz_files_to_remove = set()
    
    # Process each combination of BERT pooling and row pooling
    for bert_method in config.bert_pooling_methods:
        print(f"\n{'='*70}")
        print(f"Processing BERT Pooling: {bert_method.upper()}")
        print(f"{'='*70}")
        
        embedding_key = config.bert_pooling_key_map[bert_method]
        
        for row_pooling_method in config.row_pooling_methods:
            print(f"\n  {'-'*66}")
            print(f"  Row Pooling Method: {row_pooling_method.upper()}")
            print(f"  {'-'*66}")
            
            # Construct output path
            output_dir = os.path.join(
                config.aggregate_out_dir,
                f"aggregated_esm2_t6_8M_{bert_method}"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = f"esm2_{dataset_name}_aggregated_{row_pooling_method}.pkl"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"  Output File: {output_path}")
            
            if os.path.exists(output_path):
                print(f"  ⏭️  Skipping: Output file already exists")
                continue
            
            try:
                if dataset_type == "train":
                    metadata_path = os.path.join(data_dir, "metadata.csv")
                    
                    if not os.path.exists(metadata_path):
                        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
                    
                    # Load metadata and get repertoire IDs
                    df_metadata = pd.read_csv(metadata_path)
                    print(f"    Metadata columns: {list(df_metadata.columns)}")
                    
                    # Try to identify the ID column
                    if 'repertoire_id' in df_metadata.columns:
                        id_col = 'repertoire_id'
                    elif 'ID' in df_metadata.columns:
                        id_col = 'ID'
                    elif 'filename' in df_metadata.columns:
                        # Extract ID from filename
                        df_metadata['repertoire_id'] = df_metadata['filename'].str.replace('.tsv', '', regex=False)
                        id_col = 'repertoire_id'
                    else:
                        raise ValueError(f"Cannot find ID column in metadata. Columns: {list(df_metadata.columns)}")
                    
                    df_labels = df_metadata.set_index(id_col)['label_positive'].astype(int)
                    
                    # Aggregate features
                    repertoire_features = []
                    valid_ids = []
                    npz_files_to_remove = []
                    
                    print(f"    Looking for {len(df_labels)} repertoire files...")
                    files_found = 0
                    
                    for rep_id in df_labels.index:
                        filepath = os.path.join(features_dir, f"{rep_id}_embeddings.npz")
                        
                        if not os.path.exists(filepath):
                            if files_found < 3:  # Only print first few warnings
                                print(f"    Warning: File not found: {filepath}")
                            continue
                        
                        files_found += 1
                        with np.load(filepath) as data:
                            if embedding_key not in data:
                                print(f"    Warning: Key '{embedding_key}' not found in {filepath}. Available keys: {list(data.keys())}")
                                continue
                            embeddings = data[embedding_key]
                            aggregated_vector = apply_pooling(embeddings, row_pooling_method)
                            repertoire_features.append(aggregated_vector)
                            valid_ids.append(rep_id)
                            npz_files_to_remove.append(filepath)
                    
                    if len(repertoire_features) == 0:
                        raise RuntimeError(f"No ESM2 feature files loaded. Expected files like: {os.path.join(features_dir, '{rep_id}_embeddings.npz')}")
                    
                    X_features = np.stack(repertoire_features)
                    y_labels = df_labels.loc[valid_ids].values
                    
                    # Collect files for removal after all aggregations
                    all_npz_files_to_remove.update(npz_files_to_remove)
                    
                    print(f"    X_features shape: {X_features.shape}")
                    print(f"    y_labels shape: {y_labels.shape}")
                    output_data = (X_features, y_labels)
                    
                else:  # test
                    if dataset_num is not None:
                        # Standard test dataset with sample_submissions.csv
                        from submission.aggregate_representations import load_test_dataset_embeddings
                        X_features, sample_ids = load_test_dataset_embeddings(
                            config.sample_submissions_path,
                            dataset_num,
                            features_dir,
                            embedding_key,
                            row_pooling_method,
                            remove_npz=True
                        )
                    else:
                        # Custom test dataset - load all .npz files
                        print(f"    Loading custom test dataset (no sample_submissions.csv)...")
                        
                        sample_embeddings = []
                        sample_ids = []
                        npz_files = sorted(glob.glob(os.path.join(features_dir, "*_embeddings.npz")))
                        
                        if len(npz_files) == 0:
                            raise RuntimeError(f"No .npz files found in {features_dir}")
                        
                        print(f"    Found {len(npz_files)} .npz files")
                        
                        for npz_file in npz_files:
                            sample_id = os.path.basename(npz_file).replace("_embeddings.npz", "")
                            with np.load(npz_file) as data:
                                if embedding_key not in data:
                                    print(f"    Warning: Key '{embedding_key}' not found in {npz_file}. Available keys: {list(data.keys())}")
                                    continue
                                embeddings = data[embedding_key]
                                aggregated_vector = apply_pooling(embeddings, row_pooling_method)
                                sample_embeddings.append(aggregated_vector)
                                sample_ids.append(sample_id)
                        
                        # Collect files for removal after all aggregations
                        all_npz_files_to_remove.update(npz_files)
                        
                        if len(sample_embeddings) == 0:
                            raise RuntimeError(f"No valid embeddings found in {features_dir}")
                        
                        X_features = np.stack(sample_embeddings)
                    
                    print(f"    X_features shape: {X_features.shape}")
                    print(f"    Sample IDs: {len(sample_ids)}")
                    output_data = (X_features, sample_ids)
                
                # Save
                print(f"    -> Saving data to {output_path}...")
                with open(output_path, "wb") as f:
                    pickle.dump(output_data, f)
                
                print(f"    -> Saving complete!")
                
            except Exception as e:
                print(f"    ❌ Error during aggregation: {e}")
                raise
    
    # Now remove all npz files after ALL aggregations are complete
    if all_npz_files_to_remove:
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


def _ensure_representations_exist(data_dir: str, dataset_type: str, out_dir: str) -> None:
    """Ensure representations and aggregations exist for a dataset."""
    dataset_name = os.path.basename(data_dir)
    
    # Setup config with the correct output directory
    rep_config = RepresentationConfig()
    rep_config.representation_out_dir = os.path.join(out_dir, "representations")
    
    agg_config = AggregateConfig()
    agg_config.representation_dir = os.path.join(out_dir, "representations")
    agg_config.aggregate_out_dir = os.path.join(out_dir, "aggregates")
    
    # Step 0: Ensure k-mer features exist
    print(f"\n📊 Step 0: Checking k-mer features for {dataset_name}...")
    ensure_kmer_features_exist(data_dir, dataset_type, out_dir)
    
    # Check if aggregated features already exist
    if _check_aggregated_features_exist(dataset_name, agg_config.aggregate_out_dir):
        print(f"✅ Aggregated features already exist for {dataset_name}")
        return
    
    print(f"\n{'='*70}")
    print(f"⚠️  Aggregated features not found for {dataset_name}")
    print(f"   Generating representations and aggregations...")
    print(f"{'='*70}\n")
    
    # Step 1: Generate representations if needed
    rep_dir = os.path.join(rep_config.representation_out_dir, dataset_name)
    if not os.path.exists(rep_dir) or len(glob.glob(os.path.join(rep_dir, "*.npz"))) == 0:
        print(f"\n📊 Step 1/2: Generating ESM2 representations...")
        try:
            _generate_representations_direct(data_dir, dataset_type, rep_config)
        except Exception as e:
            print(f"❌ Error generating representations: {e}")
            raise
    else:
        print(f"✅ Representations already exist for {dataset_name}")
    
    # Step 2: Aggregate representations
    print(f"\n📊 Step 2/2: Aggregating representations...")
    try:
        _aggregate_representations_direct(data_dir, dataset_name, dataset_type, agg_config)
    except Exception as e:
        print(f"❌ Error aggregating representations: {e}")
        raise
    
    print(f"\n✅ Feature generation complete for {dataset_name}\n")


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    
    # Ensure representations and aggregations exist for all datasets
    print("\n" + "="*70)
    print("Checking feature availability...")
    print("="*70)
    
    # Check train dataset
    _ensure_representations_exist(train_dir, "train", out_dir)
    
    # Check test datasets
    for test_dir in test_dirs:
        _ensure_representations_exist(test_dir, "test", out_dir)
    
    # Check for reproducibility - if input matches a known Kaggle dataset
    kaggle_reproduce_dir = os.path.join(os.path.dirname(__file__), 'kaggle_reproduce')
    matched_dataset = check_reproducibility(train_dir, test_dirs, kaggle_reproduce_dir)
    
    if matched_dataset:
        # Found a match - use the reproduction script
        reproduce_script_name = f"Dataset{matched_dataset}_reproduce.py"
        reproduce_script = os.path.join(kaggle_reproduce_dir, reproduce_script_name)
        
        if os.path.exists(reproduce_script):
            print(f"\n🚀 Launching reproduction script for Dataset {matched_dataset}")
            print(f"   Script: {reproduce_script}")
            print(f"   Train: {train_dir}")
            print(f"   Test: {', '.join(test_dirs)}")
            print(f"   Output: {out_dir}")
            print("\n" + "="*70)
            
            # Build command
            cmd = [
                sys.executable,  # python3
                reproduce_script,
                '--train_dir', train_dir,
                '--test_dirs'] + test_dirs + [
                '--out_dir', out_dir,
                '--n_jobs', str(n_jobs)
            ]
            
            print(f"Running: {' '.join(cmd)}\n")
            
            # Run the reproduction script
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                print("\n✅ Reproduction script completed successfully!")
            else:
                print(f"\n❌ Reproduction script failed with return code: {result.returncode}")
                sys.exit(result.returncode)
            
            return
        else:
            print(f"⚠️  Reproduction script not found: {reproduce_script}")
            print("   Falling back to standard predictor...")
    
    # No match found - use standard predictor
    print("\nUsing standard predictor pipeline...")
    predictor = ImmuneStatePredictor(n_jobs=n_jobs,
                                     device=device,
                                     out_dir=out_dir)  # instantiate with any other parameters as defined by you in the class
    _train_predictor(predictor, train_dir)
    predictions = _generate_predictions(predictor, test_dirs)
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)
    _save_model(predictor, out_dir, train_dir)


def run():
    parser = argparse.ArgumentParser(description="Immune State Predictor CLI")
    parser.add_argument("--train_dir", required=True, help="Path to training data directory")
    parser.add_argument("--test_dirs", required=True, nargs="+", help="Path(s) to test data director(ies)")
    parser.add_argument("--out_dir", required=True, help="Path to output directory")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of CPU cores to use. Use -1 for all available cores.")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Device to use for computation ('cpu' or 'cuda').")
    args = parser.parse_args()
    main(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)


if __name__ == "__main__":
    run()
