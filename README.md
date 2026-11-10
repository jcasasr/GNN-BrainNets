# Interpretable Feature-Preserving Graph Neural Networks and Latent Augmentation for Multimodal Multiple Sclerosis Classification

This repository contains the code for the study:

> Casas-Roma, J., Martinez-Heras, E., Pontillo, G., Solana, E., Vivó, F., Petracca, M., Calvi, A., Solé-Ribalta, A., Llufriu, S., Prados, F., Lozano-Bagén, T. *Interpretable Feature-Preserving Graph Neural Networks and Latent Augmentation for Multimodal Multiple Sclerosis Classification*. Manuscript in preparation.

The scripts implement a workflow for preparing subject cohorts, augmenting data with variational autoencoders (VAEs), constructing multimodal graphs, computing graph features, training and evaluating graph neural networks (GNNs), and visualizing learned feature importance.

## Setup

The Conda environment is defined in [`environment.yaml`](./environment.yaml). From the repository root, create and activate it with:

```bash
conda env create -f environment.yaml
conda activate gnn-brainnets
```

The environment uses Python 3.8 and includes the libraries used by the project, including NumPy, pandas, scikit-learn, PyTorch, PyTorch Geometric, NetworkX, Optuna, and plotting tools. If Conda reports that the environment already exists, update it with:

```bash
conda env update -f environment.yaml --prune
```

The current environment file contains machine-specific paths and platform-specific package builds. On a different computer or operating system, review its `prefix` and package pins and adjust them for your Conda/PyTorch setup before creating the environment.

The glass-brain visualization in `7-3_plot-glass-brain.py` additionally requires `nilearn`, which is not listed in the environment file. Install it in the active environment if you plan to use that script:

```bash
python -m pip install nilearn
```

## Data and configuration

The scripts are run from the repository root and use settings in [`config.py`](./config.py). In particular, update `PATH_BCN`, `PATH_NAP`, and `PATH` to point to your input cohort directories and the directory where serialized subjects should be saved. Some scripts also define experiment settings directly near the top of the file; edit those constants before running an experiment.

### Input data for subject creation

[`1-1_create_subjects.py`](./1-1_create_subjects.py) builds subject objects from a cohort directory containing:

- `ID_info.csv`: one row per subject, with the columns `ID`, `origin`, `ID_old`, `mstype`, `gender`, `edss`, `dobirth`, `doscan`, `dostart`, `age`, and `DD`.
- `data/data_FA.npy`, `data/data_GM.npy`, and `data/data_RS.npy`: arrays of subject connectivity matrices, aligned by row with `ID_info.csv`.

The matrices are converted to absolute-valued, zero-diagonal matrices and stored in each subject as `FA`, `GM`, and `RS`. The included creator therefore expects matrices as NumPy arrays, not a standalone CSV of matrix values. If your matrices are in CSV files, convert/organize them into the expected NumPy arrays and prepare the matching subject metadata CSV before running the script.

## `Subject` data model

[`Subject.py`](./Subject.py) defines the object used to carry each participant and their graph data through the pipeline. A `Subject` has:

| Field | Contents |
|---|---|
| ID | Project subject identifier, generally cohort-prefixed (for example, `BCN-0001`). |
| Cohort | Cohort of origin, such as `BCN`, `NAP`, or `SYN`. |
| Old ID | Original identifier from the source dataset. |
| MS type | Integer diagnostic label: `0` healthy volunteer (HV), `1` RRMS, `2` SPMS, or `3` PPMS. |
| Attributes | Dictionary of subject metadata, such as age, gender, and clinical measures. |
| Matrices | Dictionary of adjacency matrices, such as `FA`, `GM`, `RS`, thresholded matrices, and multilayer graphs. |
| Metrics | Dictionary of per-node graph metrics used as GNN node features. |

Use the getters and setters (`get_attribute`/`set_attribute`, `get_matrix`/`set_matrix`, and `get_metric`/`set_metric`) to access these dictionaries. `get_mstype(type="binary")` maps HV to `0` and all MS subtypes to `1`; its default multiclass form returns the four labels above.

## Workflow

Run the numbered scripts in order as appropriate for your experiment. Paths and many options are configured in the scripts and [`config.py`](./config.py), rather than supplied as command-line arguments.

### 1. Create and inspect cohorts — `1-x-x.py`

1. Set the cohort input and output paths in [`config.py`](./config.py).
2. Run [`1-1_create_subjects.py`](./1-1_create_subjects.py) to load `ID_info.csv` and the FA/GM/RS matrix arrays, create `Subject` objects, and save them under `PATH`.
3. Use [`1-2_test_cohorts.py`](./1-2_test_cohorts.py), [`1-2_test_subject.py`](./1-2_test_subject.py), and [`1-2_test_single_subject.py`](./1-2_test_single_subject.py) to inspect cohort composition and stored subject data.
4. The remaining `1-x-x.py` scripts provide exploratory analyses, including per-edge and per-subject cohort comparisons and matrix density summaries.

For example:

```bash
python 1-1_create_subjects.py
python 1-2_test_cohorts.py
```

### 2. Optional VAE data augmentation — `2-2-x`

The VAE scripts use the upper-triangular entries (excluding the diagonal) of each 76-node adjacency matrix as a flattened input. The trained VAE reconstructs matrix values in the `[0, 1]` range.

1. [`2-2-1_VAE_pretrain.py`](./2-2-1_VAE_pretrain.py) performs Optuna hyperparameter search by cohort and matrix modality (`FA`, `GM`, `RS`).
2. [`2-2-2_VAE_train.py`](./2-2-2_VAE_train.py) trains the VAEs using the configured parameters and saves model weights in `models/`.
3. [`2-2-3_VAE_test-subjects.py`](./2-2-3_VAE_test-subjects.py) analyzes original and synthetic samples in matrix and latent spaces, including visualizations and summary output.
4. [`2-2-4_VAE_create-subjects.py`](./2-2-4_VAE_create-subjects.py) generates synthetic matrices by diagnostic class, combines the modalities into synthetic `Subject` objects, and saves them with the `SYN` ID prefix. Configure `NUM_SAMPLES`, `RANDOM_SCALE`, the source cohort, and `COHORT_NEW` before running it.
5. [`2-2-5_VAE_full-pipeline_TEST.py`](./2-2-5_VAE_full-pipeline_TEST.py) is an additional pipeline/test script.

The first script can run many Optuna trials and may take a long time. The later training and generation steps rely on the VAE parameters and model files being available.

### 3. Build multimodal graphs — `4-1-x`

Run [`4-1_create_ML_model.py`](./4-1_create_ML_model.py) after cohort preparation to create and store graph structures for each subject. It constructs:

- `ML`: a multilayer graph combining the FA, GM, and RS matrices.
- `ML-THR`: the corresponding graph using thresholded modality matrices.

The graph helper code uses a 76-node layer for each modality. Inspect and validate available subject matrices before proceeding.

### 4. Compute graph metrics/node features — `5-1-x`

Run [`5-1_compute_sl_metrics.py`](./5-1_compute_sl_metrics.py) to compute node-level metrics for the configured single-layer matrices and save them to each subject. The default metric list includes degree, strength, local efficiency, closeness and betweenness centrality, clustering, eigenvector centrality, and PageRank. These per-node values become input features (embeddings) for the GNNs.

### 5. Train and evaluate GNNs — `6-x-x`

- [`6-1_train_gnn_binary.py`](./6-1_train_gnn_binary.py) evaluates binary classification: HV (`0`) versus MS (`1`).
- [`6-2_train_gnn_multiclass.py`](./6-2_train_gnn_multiclass.py) evaluates four-class classification: HV, RRMS, SPMS, and PPMS.

Both scripts compare GCN, GAT, GIN, Chebyshev, GraphSAGE, and FP-GCN architectures on multimodal graph inputs. Their experiment settings include stratified four-fold cross-validation, repeated iterations, optimizer hyperparameters, graph/embedding combinations, and optional VAE augmentation. When augmentation is enabled, synthetic subjects matching `DA_PATTERN` are added to each training fold; real subjects remain the test fold. The scripts report fold-level and aggregate performance metrics, including accuracy, F1, ROC-AUC, and precision-recall measures. Adjust model grids, cohort patterns, augmentation settings, and device options near the top of the selected script before starting these experiment loops.

### 6. Train a selected model and visualize feature importance — `7-x-x`

After selecting an architecture and configuration from the evaluation experiments:

1. Configure [`7-1_train_best_model.py`](./7-1_train_best_model.py) with the desired model, graph type, and hyperparameters. It trains the selected architecture on the full configured real cohort (rather than reserving a held-out test fold) and saves the model weights under `models/`. Its checked-in defaults select binary FP-GCN experiments for `MM` and `MM-THR`.
2. Run [`7-2_print-heatmap.py`](./7-2_print-heatmap.py) to export the FP-GCN learned weights by ROI and modality/metric to CSV and create absolute and signed heatmaps under `results/`.
3. Use [`7-3_plot-glass-brain.py`](./7-3_plot-glass-brain.py) with an exported weights CSV to display the most important ROIs on a glass-brain plot. For example:

```bash
python 7-3_plot-glass-brain.py results/<model>_weights.csv --top-k 10 --out results/top_rois_glass_brain.png
```

The plotting script uses approximate MNI coordinates by default. For publication-quality localization, provide an ROI-coordinate CSV with `ROI`, `x`, `y`, and `z` columns using `--coords`.

## Outputs

Generated subject pickles are written beneath the configured `PATH`; VAE and selected GNN weights are stored in `models/`. Plots and tables are written to `figs/` and `results/` by the respective scripts. Ensure any output directories used by a script exist before running it.
