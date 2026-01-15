# Molecular Graph Captioning - Data Baseline

Baseline implementation for molecular graph captioning using autoencoders and text embeddings.

## Project Overview

This module provides tools for preprocessing molecular graphs, generating embeddings, and training models for molecular-to-text alignment using graph neural networks and language models.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

### Data Directory
- `data/` - Preprocessed data and embeddings
  - `molqwen.pt` - Pre-computed molecular embeddings
  - `train_embeddings.csv` - Training set text embeddings
  - `validation_embeddings.csv` - Validation set text embeddings

### Core Modules

- **`data_utils.py`** - Dataset utilities and loading functions
- **`preprocessing.py`** - Graph preprocessing with feature extraction (LPE, RWSE, WL)
- **`generate_description_embeddings.py`** - Generate text embeddings for molecular descriptions
- **`inspect_graph_data.py`** - Data inspection and analysis utilities
- **`display_molecules.py`** - Visualization utilities for molecular structures

### Models & Training

- **`Auto_encoder_model.ipynb`** - Autoencoder architecture and training notebook

### Data & Results

- `test_retrieved_descriptions.csv` - Retrieved descriptions for test molecules

## Usage

### 1. Inspect Graph Data

Analyze the structure of your molecular graph data:

```bash
python inspect_graph_data.py
```

### 2. Preprocess Graphs

Prepare graphs with structural and spectral features:

```bash
python preprocessing.py
```

### 3. Generate Text Embeddings

Create embeddings for molecular descriptions:

```bash
python generate_description_embeddings.py
```

Generates:
- `data/train_embeddings.csv`
- `data/validation_embeddings.csv`

### 4. Train Autoencoder

Train the autoencoder model (see notebook):

```bash
jupyter notebook Auto_encoder_model.ipynb
```

### 5. Visualize Results

Display molecular structures and descriptions:

```bash
python display_molecules.py
```

## Requirements

See `requirements.txt` for dependencies.

