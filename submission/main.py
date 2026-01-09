import os

# Set PYTHONHASHSEED before other imports for reproducibility hash randomization?
os.environ['PYTHONHASHSEED'] = '0'

import argparse
import pandas as pd
import subprocess
import sys
from typing import List
from submission.predictor import ImmuneStatePredictor
from submission.utils import save_tsv, validate_dirs_and_files
from submission.reproduce_checker import check_reproducibility, get_reproduce_script_path


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


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    
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
