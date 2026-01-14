# Molecule-Text Retrieval

Graph neural network for molecular graph and text description generation.

## Installation

```bash
uv sync
```

## Data Setup

Place your preprocessed graph data files in the `data/` directory:
- `train_graphs.pkl`
- `validation_graphs.pkl`
- `test_graphs.pkl`

## Usage

Run the following scripts in order:

### 1. Run Stage 1

Pre-train stage 1 :

```bash
python3 -m scripts.stage1
```
This generates:
- `checkpoints/mlp_adapter_stage1.pth`

### 2. Run Stage 2

Pre-train stage 2:

```bash
python3 -m scripts.stage2
```

This generates:
- `checkpoints/mlp_adapter_stage2.pth`
- `checkpoints/gnn_stage2.pth`

### 3. Run Final Stage:

Galactica and MLPAdapter and GNN fine-tuning

```bash
python3 -m scripts.stage3
```

This generates models weights and tokenizer to use for inference:
- `checkpoints/galactica_full_final`
- `checkpoints/mlp_adapter_final.pth`
- `checkpoints/gnn_final.pth`

### 4. Run Generation

Generate descriptions for test molecules:

```bash
python3 -m scripts.test_results
```

This generates `galactica_full_final.csv` with generated descriptions for each test molecule.

## Output

- `gnn_final.pth`: Trained GCN model
- `mlp_adapter_final.pth`: Trained MLPAdapter
- `galactica_full_final`: Folder with fine-tuned Galactica and tokenizer
- `galactica_full_final.csv`: Generated descriptions for test set

## Alternative

You can run `molecular_graph_captioning.ipynb` in Google Colab