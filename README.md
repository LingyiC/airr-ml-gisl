# Ensemble AIRR Predictor - GISL 

This repository provides a reproducible machine learning pipeline for AIRR-ML challenge.
The pipeline automatically selects the best-performing strategy based on dataset characteristics.

## Overview

The pipeline is built on **three core methods**:

* **K-mer–based features**
* **Public clone–based features**
* **ESM (protein language model) embeddings**

Different datasets use different combinations of these methods.


## Dataset–Method Mapping

| Dataset   | Methods Used       |
| --------- | ------------------ |
| Dataset 1 | K-mer              |
| Dataset 2 | K-mer              |
| Dataset 3 | Public clone       |
| Dataset 4 | K-mer              |
| Dataset 5 | K-mer              |
| Dataset 6 | Public clone       |
| Dataset 7 | Public clone + ESM |
| Dataset 8 | K-mer + ESM        |



## Reproducing Kaggle Results

Each Kaggle dataset has a dedicated reproduction script.

### Example: Dataset 1

```bash
python3 submission/kaggle_reproduce/Dataset1_reproduce.py \
   --train_dir /Users/lingyi/Documents/airr-ml/data/train_datasets/train_dataset_1 \
   --test_dirs /Users/lingyi/Documents/airr-ml/data/test_datasets/test_dataset_1 \
   --out_dir /Users/lingyi/Documents/airr-ml/workingFolder/output \
   --n_jobs 4
```

### Skip Top Sequencing Step

To disable top-sequence filtering (save time):

```bash
python3 submission/kaggle_reproduce/Dataset1_reproduce.py \
   --train_dir /Users/lingyi/Documents/airr-ml/data/train_datasets/train_dataset_1 \
   --test_dirs /Users/lingyi/Documents/airr-ml/data/test_datasets/test_dataset_1 \
   --out_dir /Users/lingyi/Documents/airr-ml/workingFolder/output \
   --no-topseq \
   --n_jobs 4
```



## Running on New Datasets

For new datasets, use the **main pipeline**.
The system will automatically select the best approach.

### Full Run (Includes ESM — Default)

```bash
python3 -m submission.main \
   --train_dir ./train_datasets/train_dataset_x \
   --test_dirs ./test_datasets/test_dataset_x \
   --out_dir /Users/lingyi/Documents/airr-ml/workingFolder/output \
   --no-reproduce \
   --no-topseq \
   --n_jobs 4
```

> ⚠️ **Note**: ESM embeddings are computationally expensive and may take a long time.



### Quick Run (Skip ESM — Faster; Do not recommend on experimental data)

```bash
python3 -m submission.main \
   --train_dir ./train_datasets/train_dataset_x \
   --test_dirs ./test_datasets/test_dataset_x \
   --out_dir /Users/lingyi/Documents/airr-ml/workingFolder/output \
   --no-reproduce \
   --no-topseq \
   --no-esm \
   --n_jobs 4
```



## Key Arguments

| Argument         | Description                             |
| ---------------- | --------------------------------------- |
| `--train_dir`    | Path to training dataset                |
| `--test_dirs`    | Path(s) to test dataset(s)              |
| `--out_dir`      | Output directory                        |
| `--n_jobs`       | Number of parallel workers              |
| `--no-topseq`    | Disable top sequence filtering          |
| `--no-esm`       | Disable ESM embeddings                  |
| `--no-reproduce` | Skip Kaggle-specific reproduction logic |



## Notes

* Kaggle reproduction scripts are **dataset-specific**
* The main pipeline is **dataset-agnostic** and recommended for new data
* ESM improves performance on some datasets but significantly increases runtime
