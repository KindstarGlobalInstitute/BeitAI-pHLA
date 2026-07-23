# BeitAI-pHLA

A peptide-HLA Binding Estimation using Immune Technology of Artificial Intelligence.

This repository implements a **Multiple Instance Learning (MIL)** approach for predicting peptide-HLA binding affinity. The model uses ESM-2 for peptide/MHC pseudo-sequence embedding, followed by DPCNN + CapsNet feature extraction and MIL-based attention pooling for multi-allele scenarios.


## Requirements

- Python >= 3.9
- numpy == 1.26.4
- pandas == 2.2.1
- torch == 2.2.1
- pytorch-cuda == 12.1
- fair-esm == 2.0.0


## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/KindstarGlobalInstitute/BeitAI-pHLA.git
   cd BeitAI-pHLA
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download model weights from Google Drive and place them in `model_MIL/`:
   - [Model Weights](https://drive.google.com/drive/folders/1TZ5crvmMiKCRJLJsdYxzrU89jTVMh7ER)
   - Expected files: `EL_Classification_train_split0.ckpt`, `split1.ckpt`, `split2.ckpt` (3 ensemble splits)

> **Note:** The first run will automatically download ESM-2 weights. This may take a few minutes depending on your network connection. Subsequent runs will use the cached weights.


## Data & Model Weights

Due to size constraints, the following files are **not** included in this repository and must be downloaded separately:

| Files | Location | Size |
|-------|----------|------|
| Model checkpoints (3 splits) | [Google Drive](https://drive.google.com/drive/folders/1TZ5crvmMiKCRJLJsdYxzrU89jTVMh7ER) | ~1.8 GB |
| Training and testing data | [Google Drive](https://drive.google.com/drive/folders/1TZ5crvmMiKCRJLJsdYxzrU89jTVMh7ER) | ~1 GB |

The small auxiliary files (allele mapping, pseudo-sequences, example data) are included in `data_random/`.


## Usage

### Quick Prediction

The main entry point for predicting peptide-HLA binding:

```bash
# Basic prediction (prints results to stdout)
python predict.py -i data_random/Example.txt

# Save results to a file
python predict.py -i data_random/Example.txt -o results/predictions.txt

# Use a specific GPU
python predict.py -i data_random/Example.txt -d cuda:1

# Use CPU with smaller batch size
python predict.py -i data_random/Example.txt -d cpu -b 64

# Custom model directory
python predict.py -i my_peptides.txt -o output.txt --model-dir ./my_models/
```

#### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `-i, --input` | Input file path (tab-separated: `Peptide<TAB>HLA`) | (required) |
| `-o, --output` | Output file path | stdout |
| `-d, --device` | Device for inference (`cuda:0`, `cuda:1`, `cpu`) | `cuda:0` |
| `-b, --batch-size` | Inference batch size | `128` |
| `--model-dir` | Directory containing `.ckpt` checkpoint files | `./model_MIL/` |
| `--pseudo-path` | Path to `MHC_pseudo.dat` | `./data_random/MHC_pseudo.dat` |
| `--allelelist-path` | Path to `allelelist` | `./data_random/allelelist` |

#### Input Format

The input file must be **tab-separated** with a header row:

```
Peptide	HLA
AIVDDAIEKL	HLA-A02:01
FPQMGRFTL	HLA-A03:01,HLA-B18:05,HLA-C04:01
```

- **Single-allele (SA):** One HLA allele per row.
- **Multi-allele (MA):** Multiple HLA alleles separated by commas per row. The model uses attention pooling to determine which allele(s) drive binding.

#### Output Format

The output contains the original columns plus a `Prediction` column (binding probability, 0-1):

```
Peptide	HLA	Prediction
AIVDDAIEKL	HLA-A02:01	0.8732
FPQMGRFTL	HLA-A03:01,HLA-B18:05,HLA-C04:01	0.6541
```


### Seq2Logo (Motif Visualization)

To generate sequence logo plots from predicted binding peptides, you can use [Seq2Logo](https://services.healthtech.dtu.dk/services/Seq2Logo-2.0/):

**Option 1: Online (no installation)**
1. Go to [Seq2Logo-2.0](https://services.healthtech.dtu.dk/services/Seq2Logo-2.0/)
2. Upload your peptide list (FASTA or plain text format)
3. Configure logo options and download the result

**Option 2: Local installation**
1. Download the Seq2Logo script from [DTU Health Tech](https://services.healthtech.dtu.dk/services/Seq2Logo-2.0/)
2. Install dependencies (Python 2.7 and LaTeX are required)
3. Run the motif analysis script:
   ```bash
   cd data_processing
   python MA_motif_analysis.py
   ```
4. Configure the paths in `MA_motif_analysis.py`:
   ```python
   seq2logo_path = '~/seq2logo-2.0/Seq2Logo.py'  # Update this path
   python_path = '~/anaconda3/envs/python2.7/bin/python'  # Update this path
   ```


### Ablation Experiments

Several ablation scripts are provided to evaluate different architectural variants:

```bash
# Standard BeitAI model (our full model)
python run_MIL_ablation_BeitAI.py

# Without ESM embeddings (random embedding layer)
python run_MIL_ablation_NoESM.py

# Without MIL (mean pooling instead of attention)
python run_MIL_ablation_NoMIL.py

# Replace DPCNN+CapsNet with Transformer
python run_MIL_ablation_transformer.py

# Full ablation suite
python run_MIL_ablation.py
```

### Training from Scratch

To train the model with your own data:

```bash
# 1. Prepare data splits
cd data_processing
python data_split.py

# 2. Run DDP training (3 GPUs)
cd ..
python run_MIL.py

# 3. OR run sensitivity analysis for pos_weight hyperparameter
python run_MIL_sensitivity.py
```


## Project Structure

```
BeitAI-pHLA/
├── predict.py                 # Quick prediction script
├── dataset_MIL.py             # MHC-EL dataset with MIL collate function
├── mhc_model_MIL.py           # Model definitions (MIL, ablations)
├── train_MIL.py               # Training loop with early stopping
├── run_MIL.py                 # DDP training entry point
├── run_MIL_ablation*.py       # Ablation study scripts
├── run_MIL_sensitivity.py     # Hyperparameter sensitivity analysis
├── test_our_data_MIL.py       # Batch inference for benchmarks
├── Net_Capsule.py             # Capsule network layer
├── Net_DPCNN.py               # Deep Pyramid CNN
├── Transformer.py             # Positional encoding and Transformer encoder
├── data_processing/           # Data splitting, external tool wrappers
│   ├── data_split.py
│   ├── Hobohm1.py
│   └── MA_motif_analysis.py
├── data_random/               # Auxiliary data files
│   ├── MHC_pseudo.dat         # HLA → pseudo-sequence mapping
│   ├── allelelist             # Allele name ↔ MHC name mapping
│   └── Example.txt            # Example input file
└── model_MIL/                 # Model checkpoint directory (download from GDrive)
```



## Support

If you have any questions or need further assistance, please contact [qiushigang@kindstar.com.cn] or visit the [GitHub Issues](https://github.com/KindstarGlobalInstitute/BeitAI-pHLA/issues) page.
