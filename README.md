# RDiff: Programmable Functional RNA Design

<p align="center">
  <strong>Programmable RNA design with diffusion models and structure-aware conditioning</strong>
</p>

<p align="center">
  <a href="https://github.com/ml4bio/RDiff"><img src="https://img.shields.io/badge/Project-GitHub-181717?logo=github" alt="GitHub"></a>
  <a href="https://github.com/ml4bio/RDiff/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-supported-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
</p>

RDiff is an RNA design framework for generating functional RNA candidates through programmable structural and interaction constraints. The repository provides two complementary inference modules:

- **`RDiff_seq`** — inverse-folding sequence design conditioned on RNA 3D structures, with optional secondary-structure, noncanonical-interaction, and fixed-motif constraints.
- **`RDiff_str`** — RNA 3D backbone generation conditioned on secondary structure, noncanonical interactions, or user-provided scaffolding motifs.

> **Repository scope.** This public release is focused on inference. Training scripts, training datasets, and training dataloaders are not included.

## Highlights

- 🧬 Structure-conditioned RNA sequence design
- 🧱 Conditional RNA 3D backbone generation
- 🔗 Watson–Crick and noncanonical base-pair conditioning
- 🧩 Fixed-sequence constraints and motif scaffolding
- ⚙️ Batch inference with configurable diffusion sampling

## Repository structure

```text
RDiff/
├── RDiff_seq/        # RNA inverse folding and sequence design
├── RDiff_str/        # RNA 3D backbone generation and scaffolding
├── env_RDiff.yaml    # Conda environment specification
└── LICENSE           # MIT License
```

## Installation

```bash
conda env create -f env_RDiff.yaml
conda activate RDiff
```

## Pretrained checkpoints

Pretrained checkpoints are available in the following directory:

<https://drive.google.com/drive/folders/11o--a0hqlBrGhnD7BaJ1Wpfw3VinjDur?usp=sharing>

After downloading, place the checkpoints in the corresponding directories:

```text
RDiff_seq/RDiff_seq_ckpts/                 # Sequence-design checkpoints
RDiff_str/RDiff_str_ckpts/                 # Structure-generation checkpoints
RDiff_str/RDiff_scaffolding_ckpts/         # Scaffolding checkpoints
```

You may also provide absolute checkpoint paths in the commands below. Checkpoint names with an `nc` suffix denote subtype-level non-Watson–Crick-conditioned models; use these models when supplying detailed noncanonical interaction labels through `NC_sparse` or `input_nc_npz`.

## Quick start

### Sequence design

Design RNA sequences conditioned on one or more RNA 3D structures:

```bash
cd RDiff_seq
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/rdiff_seq_ss.py \
  --no_wandb \
  --mode inf_dir \
  --inf_folder /path/to/pdb_or_directory \
  --config.eval.output_fasta_file output.fasta \
  --config.eval.n_samples=2500 \
  --config.eval.model_path RDiff_seq_ckpts/ribo_woss_cluster/step_500000.pth
```

`--inf_folder` accepts either a single `.pdb` file or a directory containing `.pdb` files. For fixed-sequence masks, secondary-structure conditioning, and detailed noncanonical interaction conditioning, see [`RDiff_seq/README.md`](RDiff_seq/README.md).

### Structure generation

Generate RNA backbones conditioned on a dot-bracket secondary structure:

```bash
cd RDiff_str
python inference_rdiff_str.py --config-name=rna_cww_ss_training \
  inference.name=hhr_generation \
  inference.weights_path=RDiff_str_ckpts/cww_ss_1250k.pth \
  inference.num_samples=10 \
  inference.target_length=49 \
  "inference.ss_db_constrain='(((((....(..(((((..))))).).(((((((..))))))).)))))'" \
  inference.diffusion.num_t=500 \
  inference.seed=42
```

For secondary-structure plus noncanonical interaction conditioning, motif scaffolding, and contig sweeps, see [`RDiff_str/README.md`](RDiff_str/README.md) and [`RDiff_str/scaffolding_pipeline.md`](RDiff_str/scaffolding_pipeline.md).

## Outputs

- `RDiff_seq` writes designed sequences to FASTA files.
- `RDiff_str` writes generated RNA backbone structures as PDB files under `inference.output_dir/inference.name/`.

## Citation

If you use RDiff in your work, please cite:

> **Programmable functional RNA design with a virtual scientist**
>
> Han Huang, Tianqing Zhang, Liang Hong, Ziqian Lin, Dongchen He, Jiuming Wang, Tian-Ying Wu, Huimin Li, Da Han, Le Yu, Jiyue Jiang, Yimin Fan, Yanruisheng Shao, Irwin King, Pengan Chen, Chun Kit Kwok, James J. Collins, and Yu Li. *Nature Biotechnology* (2026).

```bibtex
@article{rhoagent,
  title   = {Programmable functional RNA design with a virtual scientist},
  author  = {Han Huang, Tianqing Zhang, Liang Hong, Ziqian Lin, Dongchen He, Jiuming Wang, Tian-Ying Wu, Huimin Li, Da Han, Le Yu, Jiyue Jiang, Yimin Fan, Yanruisheng Shao, Irwin King, Pengan Chen, Chun Kit Kwok, James J. Collins, and Yu Li},
  journal = {Nature Biotechnology},
  year    = {2026},
  doi     = {10.1038/s41587-026-03362-3}
}
```

## License

This project is released under the [MIT License](LICENSE).
