# RDiff_str

Inference code for RNA 3D backbone generation. Training scripts and training dataloaders are not included in this public release.

## WC-Pair Conditional Generation

```bash
python inference_rdiff_str.py --config-name=rna_cww_ss_training \
  inference.name=hhr_generation \
  inference.weights_path=RDiff_str_ckpts/cww_ss_1250k.pth \
  inference.num_samples=10 \
  inference.target_length=49 \
  "inference.ss_db_constrain='(((((....(..(((((..))))).).(((((((..))))))).)))))'" \
  inference.diffusion.num_t=500 \
  inference.seed=42
```

For older checkpoints trained without IPA pLDDT, add:

```bash
model.ipa.use_plddt=False
```

## Secondary Structure + Noncanonical Conditions

This mode expects paired condition files:

```text
condition_dir/
  ssmap3/<target>.npy
  NC_sparse/<target>.npz
```

`ssmap3/<target>.npy` can be `[3, N, N]` or `[N, N, 3]`. `NC_sparse/<target>.npz` contains `nt1`, `nt2`, and `type_num` arrays using the 11-class noncanonical interaction mapping.

Use checkpoints whose names include `nc` for this subtype-level NWC conditioning mode.

Run all targets in a condition directory:

```bash
python inference_rdiff_str.py --config-name=rna_ss_partial_nc_inference \
  inference.task_name=ss_nc_conditional \
  inference.condition_dir=./time_censor \
  inference.weights_path=RDiff_str_ckpts/ss3_nc_1000k.pth \
  inference.output_dir=./inference_outputs \
  inference.name=time_censor_ss_partial_nc \
  inference.num_samples=10 \
  inference.gpu_id=0
```

Run one target pair:

```bash
python inference_rdiff_str.py --config-name=rna_ss_partial_nc_inference \
  inference.task_name=ss_nc_conditional \
  inference.ssmap_path=./time_censor/ssmap3/example.npy \
  inference.nc_sparse_path=./time_censor/NC_sparse/example.npz \
  inference.weights_path=RDiff_str_ckpts/ss3_nc_1000k.pth \
  inference.output_dir=./inference_outputs \
  inference.name=example_ss_partial_nc \
  inference.num_samples=10 \
  inference.gpu_id=0
```

Outputs are written to:

```text
<output_dir>/<name>/<target>/<target>_sample_<i>.pdb
```

## Prepare ssmap3 and NC_sparse

Convert an interaction CSV with columns `nt1`, `nt2`, `type`, and `classification-1`:

```bash
python data_process/prepare_ss_nc_from_csv.py \
  --csv example.csv \
  --seq example.seq \
  --out_dir time_censor \
  --target example
```

The converter writes:

```text
time_censor/
  ssmap3/example.npy
  NC_sparse/example.npz
```

## Scaffolding

For one scaffolding run:

```bash
python inference_rdiff_str.py --config-name=rna_scaffolding_training \
  inference.task_name=scaffolding \
  inference.name=scaffold_example \
  inference.weights_path=RDiff_scaffolding_ckpts/scaffolding_500k.pth \
  inference.target_pdb_path=input_exp/example.pdb \
  inference.target_length=90 \
  inference.ori_length=49 \
  inference.use_contigs=True \
  inference.contigs='8/A3-16/6/A29-47/10' \
  inference.num_samples=5
```

For contig sweeps, use `inference.task_name=scaffolding_iter`. Gap tokens support fixed lengths (`8`), ranges (`8-12`), and open minimum lengths (`8+`) that are expanded to match the target length.

```bash
python inference_rdiff_str.py --config-name=rna_scaffolding_training \
  inference.task_name=scaffolding_iter \
  inference.name=scaffold_sweep \
  inference.weights_path=RDiff_scaffolding_ckpts/scaffolding_500k.pth \
  inference.target_pdb_path=input_exp/example.pdb \
  inference.ori_length=49 \
  inference.target_length=90 \
  inference.use_contigs=True \
  inference.contigs='8+/A3-16/6-10/A29-47/8+' \
  inference.scaffold_iter.length_min=80 \
  inference.scaffold_iter.length_max=100 \
  inference.scaffold_iter.length_step=5 \
  inference.num_samples=2
```

More filtering and scaffolding examples are in `scaffolding_pipeline.md`.
