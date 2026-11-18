# RDiff_seq

Inference code for RNA inverse folding. Training datasets and training dataloaders are not included in this public release.

## Tertiary-Structure Conditioning

`--inf_folder` can be a directory of `.pdb` files or one `.pdb` file.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/rdiff_seq_ss.py \
  --no_wandb \
  --mode inf_dir \
  --inf_folder /path/to/pdb_or_directory \
  --config.eval.output_fasta_file output.fasta \
  --config.eval.n_samples=2500 \
  --config.eval.model_path RDiff_seq_ckpts/ribo_woss_cluster/step_500000.pth
```

If `output_fasta_file` is relative, it is written under the input PDB directory. If omitted, the default is `output_seq.fasta` under that directory.

## Fixed Sequence Constraints

Enable fixed motifs with:

```bash
--config.eval.fixed_part=True
```

Use `A/G/C/U` for fixed bases and `-` for positions to redesign.

Direct command-line mask:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/rdiff_seq_ss.py \
  --no_wandb \
  --mode inf_dir \
  --inf_folder /path/to/example.pdb \
  --config.eval.fixed_part=True \
  --config.eval.fixed_seq='--------------UAC-------------GCA' \
  --config.eval.n_samples=150 \
  --config.eval.model_path RDiff_seq_ckpts/ribo_woss_cluster/step_500000.pth
```

FASTA mask next to each PDB:

```text
picked_pdbs/
  scaffold_001.pdb
  scaffold_001.fasta
  scaffold_002.pdb
  scaffold_002.fasta
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/rdiff_seq_ss.py \
  --no_wandb \
  --mode inf_dir \
  --inf_folder /path/to/picked_pdbs \
  --config.eval.fixed_part=True \
  --config.eval.n_samples=150 \
  --config.eval.model_path RDiff_seq_ckpts/ribo_woss_cluster/step_500000.pth
```

Use FASTA masks from another directory:

```bash
--config.eval.fixed_seq_path /path/to/fasta_dir
```

Use one FASTA file for a single PDB:

```bash
--config.eval.fixed_seq_path /path/to/fixed_mask.fasta
```

When `fixed_part=True`, `fixed_seq` has priority over `fixed_seq_path`; otherwise the code searches for `<pdb_stem>.fasta` or `<pdb_stem>.fa`.

## Secondary-Structure Conditioning

Use dot-bracket input for one-channel WC base-pair conditioning:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/rdiff_seq_ss.py \
  --no_wandb \
  --mode inf_dir \
  --inf_folder /path/to/example.pdb \
  --config.data.ss_channel=1 \
  "--config.eval.input_db=((((....))))" \
  --config.eval.n_samples=250 \
  --config.eval.model_path RDiff_seq_ckpts/ribo_ss1_cluster/step_500000.pth
```

Or pass a dense NumPy map:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/rdiff_seq_ss.py \
  --no_wandb \
  --mode inf_dir \
  --inf_folder /path/to/pdb_dir \
  --config.data.ss_channel=1 \
  --config.eval.input_ss_npy /path/to/input_ss.npy \
  --config.eval.n_samples=250 \
  --config.eval.model_path RDiff_seq_ckpts/ribo_ss1_cluster/step_500000.pth
```

`input_ss.npy` should have shape `[C, L, L]`; the first `ss_channel` channels are used.

For multi-channel secondary-structure maps:

- `ss_channel=1`: WC base pairs
- `ss_channel=2`: WC base pairs + non-WC base pairs
- `ss_channel=3`: WC base pairs + non-WC base pairs + stacking interactions

## Secondary Structure + Noncanonical Type Conditioning

This mode conditions on tertiary structure, a dense `ssmap3` file, and sparse detailed noncanonical base-pair labels.

Use checkpoints whose names include `nc` for this subtype-level NWC conditioning mode.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/rdiff_seq_ss.py \
  --no_wandb \
  --mode inf_dir \
  --inf_folder /path/to/example.pdb \
  --config.eval.output_fasta_file output_seq.fasta \
  --config.eval.batch_size=10 \
  --config.eval.n_samples=100 \
  --config.eval.input_ss_npy /path/to/example_ssmap3.npy \
  --config.eval.input_nc_npz /path/to/example_NC_sparse.npz \
  --config.data.ss_channel=3 \
  --config.data.use_nc_detail_cond=True \
  --config.data.nc_channel=11 \
  --config.eval.model_path RDiff_seq_ckpts/ribo_ss3_nc_cluster/checkpoints/step_500000.pth
```

`example_ssmap3.npy` should be `[C, L, L]` with `C >= 3`. The convention is channel 0 for WC base pairs, channel 1 for non-WC base pairs, and channel 2 for stacking interactions.

`example_NC_sparse.npz` must contain `nt1`, `nt2`, and `type_num` arrays. `nt1` and `nt2` are 0-based residue indices; `type_num` is the NC type ID and is applied symmetrically during inference.

NC type IDs:

```text
1:tWW  2:cWH/cHW  3:tWH/tHW  4:cWS/cSW  5:tWS/tSW  6:cHH
7:tHH  8:cHS/cSH  9:tHS/tSH  10:cSS  11:tSS
```

## Output FASTA

Headers use:

```text
><pdb_stem>_<sample_index>
```

Recommended downstream filters include long homopolymer runs, GC content, sequence complexity, and self-consistency with the input structure.
