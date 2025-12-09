# Neurons That Panic

A mechanistic interpretability project studying misgeneralization in small transformer models by identifying "panic neurons" whose activations change significantly under adversarial inputs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1opVfgIxZDAbYgoW3gQ8Nk5-QScSDAO7I?usp=sharing)

## Create virtual environment & Installation
```
uv venv
```

```bash
uv pip install -e .
```

## Dependencies

- `transformer_lens` - Model loading and activation extraction
- `torch` - PyTorch for tensor operations
- `datasets` - Dataset loading (SST-2)
- `matplotlib` & `seaborn` - Plotting
- `pandas` & `numpy` - Data manipulation

## Usage

### Running the Full Experiment

```bash
python run.py [OPTIONS]
```

Options:
- `--model`: HuggingFace model path (default: `EleutherAI/pythia-160m-deduped`)
- `--num-prompts`: Number of prompts to process (default: `200`)
- `--k`: Number of panic components to rank (default: `20`)
- `--batch-size`: Batch size for processing (default: `16`)
- `--device`: Device to use (default: `cuda`)
- `--save`: Output directory (default: `artifacts/`)
- `--n-candidates`: Number of candidate tokens for adversarial generation (default: `500`)
- `--n-random`: Number of random trials for patching baseline (default: `5`)
- `--seed`: Random seed (default: `42`)
- `-v, --verbose`: Enable verbose output

Example:
```bash
python run.py --num-prompts 200 --k 20 --verbose
```

### Running Notebooks

The notebooks (`experiment.ipynb` and `sanity_checks.ipynb`) import functions from the `src/` modules. They can be run top-to-bottom and will produce the same results as `run.py`.

## Project Structure

```
src/
├── data.py    # Loading prompts from SST-2
├── model.py   # Loading model + tokenizer
├── adv.py     # Creating adversarial triggers
├── acts.py    # Activation extraction + hooks
├── dn.py      # Delta-norm computation + ranking
├── patch.py   # Causal patching
└── plot.py    # Visualizations
```

## Reproducing Results

- **Main Experiment Notebook**: `experiment.ipynb`
- **Sanity Checks Notebook**: `sanity_checks.ipynb`
- **Supplementary Materials**: ![Google Drive](https://drive.google.com/drive/folders/1jlVze75NJNizPTgy-_hHET-VfYZlTu6s?usp=sharing)