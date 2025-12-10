"""
Simple test to verify the ImmuneStatePredictor interface is correct.
"""
import sys
import os

# Add submission to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from submission.predictor import ImmuneStatePredictor
    
    print("✓ Successfully imported ImmuneStatePredictor")
    
    # Test instantiation
    predictor = ImmuneStatePredictor(n_jobs=4, device='cpu', base_dir='/Users/lingyi/Documents/airr-ml/')
    print("✓ Successfully instantiated predictor")
    
    # Check required methods exist
    assert hasattr(predictor, 'fit'), "Missing fit() method"
    print("✓ fit() method exists")
    
    assert hasattr(predictor, 'predict_proba'), "Missing predict_proba() method"
    print("✓ predict_proba() method exists")
    
    assert hasattr(predictor, 'identify_associated_sequences'), "Missing identify_associated_sequences() method"
    print("✓ identify_associated_sequences() method exists")
    
    # Check attributes
    assert hasattr(predictor, 'n_jobs'), "Missing n_jobs attribute"
    assert predictor.n_jobs == 4, f"n_jobs should be 4, got {predictor.n_jobs}"
    print("✓ n_jobs attribute correct")
    
    assert hasattr(predictor, 'device'), "Missing device attribute"
    assert predictor.device == 'cpu', f"device should be 'cpu', got {predictor.device}"
    print("✓ device attribute correct")
    
    print("\n" + "="*60)
    print("✓ ALL INTERFACE CHECKS PASSED!")
    print("="*60)
    print("\nThe ImmuneStatePredictor class is ready and compatible with")
    print("the AIRR-ML-25 challenge template requirements.")
    print("\nTo run the full pipeline, use:")
    print("python3 -m submission.main --train_dir /path/to/train_dir \\")
    print("                           --test_dirs /path/to/test_dir \\")
    print("                           --out_dir /path/to/output \\")
    print("                           --n_jobs 4 --device cpu")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install dependencies first:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
