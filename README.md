# 🔮 Ensemble AIRR Predictor - GISL Team 

This repository provides a reproducible machine learning pipeline for AIRR-ML challenge (Rank 9th).
The pipeline automatically selects the best-performing strategy based on dataset characteristics. 

## 📋 Overview

The pipeline is built on **three core methods**:

* **K-mer–based features**
* **Public clone–based features**
* **ESM (protein language model) embeddings**

Different datasets use different combinations of these methods.

## 🗂️ Dataset–Method Mapping

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



## 🔄 Reproducing Kaggle Results

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



## 🚀 Running on New Datasets

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

> 💡 **Tip**: By default `--num_gpus -1` to use all available GPUs. The batch size will be automatically scaled by the number of GPUs.



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



## 🔑 Key Arguments

| Argument            | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `--train_dir`       | Path to training dataset                                                    |
| `--test_dirs`       | Path(s) to test dataset(s)                                                  |
| `--out_dir`         | Output directory                                                            |
| `--n_jobs`          | Number of parallel workers (default: 1)                                     |
| `--no-topseq`       | Disable top sequence filtering                                              |
| `--no-esm`          | Disable ESM embeddings                                                      |
| `--no-reproduce`    | Skip Kaggle-specific reproduction logic                                     |
| `--num_gpus`        | Number of GPUs for ESM (default: -1, all available GPUs)                   |
| `--esm_batch_size`  | Batch size for ESM encoding (default: 128, auto-scaled with multiple GPUs) |
| `--esm_model_name`  | ESM model to use (default: facebook/esm2_t6_8M_UR50D). Other options: esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D |
| `--esm_grid_models` | Models for ESM grid search (default: ExtraTrees_shallow, SVM_Linear). Options: LogReg_L1, LogReg_L2, LogReg_ElasticNet, SVM_Linear, SVM_RBF, RandomForest_shallow, ExtraTrees_shallow, GradientBoosting, AdaBoost, GaussianNB |



## 🐳 Running the docker container

The docker container is available as `th8623/airr25_gisl` on [Docker Hub](https://hub.docker.com/r/th8623/airr25_gisl).

Pull the image from Docker Hub

```
docker pull th8623/airr25_gisl:latest
```

Or build an image from this repository

```
docker build \
  --platform linux/amd64 \
  -t your_namespace/airr25_gisl:0.1.3 .
```

Reproducing Kaggle results (Dataset 1)

```bash
docker run --rm \
  -v /path/to/train_dataset_1:/data/train_dataset_1 \
  -v /path/to/test_dataset_1:/data/test_dataset_1 \
  -v /path/to/output:/output \
  th8623/airr25_gisl:latest \
  --train_dir /data/train_dataset_1 \
  --test_dir /data/test_dataset_1 \
  --out_dir /output \
  --n_jobs 4 \
  --no-esm
```

Running on new datasets

```bash
docker run --rm \
  -v /path/to/train_dataset:/data/train_dataset \
  -v /path/to/test_dataset:/data/test_dataset \
  -v /path/to/output:/output \
  th8623/airr25_gisl:latest \
  --train_dir /data/train_dataset \
  --test_dir /data/test_dataset \
  --out_dir /output \
  --n_jobs 4 \
  --no-reproduce \
  --no-topseq
```

## 🔬 Pipeline Overview
![Pipeline Overview](Pipeline.png)

<sub>🍌 *Pipeline figure generated by nano banana*</sub>

## 📝 Notes

* Kaggle reproduction scripts are **dataset-specific**
* The main pipeline is **dataset-agnostic** and recommended for new data
* ESM improves performance on some datasets but significantly increases runtime

## 👥 Contributors

* **Team GISL** from Columbia University

