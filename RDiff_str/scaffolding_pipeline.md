# STEP1: Check if AlphaFold3 could reproduce the structure of motif

```
> Broccoli WT
GCGGAGACGGUCGGGUCCAGAUAUUCGUAUCUGUCGAGUAGAGUGUGGGCUCCGC

> Pepper WT
GCUCCCCAAUCGUGGCGUGUCGGCCUGCUUCGGCAGGCACUGGCGCCGGGAGC
```

* Broccoli aptamer AF3 ion and ligand: 3 K, 2 NA, 1 2ZY
* Pepper aptamer AF3 ion and ligand: 5 Mg, 1 J8F

# STEP2: Run RDiff_str inference for scaffolding

## Broccoli scaffolding
```
# single contig for scaffolding (could be used for testing)
python inference_rdiff_str.py --config-name=rna_scaffolding_training \
inference.name=broc_scaffolding inference.weights_path=RDiff_scaffolding_ckpts/scaffolding_500k.pth \
inference.target_pdb_path=examples/8k7w_broccoli_core.pdb inference.target_length=120 \
inference.contigs="30/A3-16/15/A29-47" inference.seed=123

# Use other ckpy example
python inference_rdiff_str.py --config-name=rna_scaffolding_training \
inference.name=broc_scaffolding inference.weights_path=RDiff_scaffolding_ckpts/scaffolding_500k.pth \
inference.target_pdb_path=examples/8k7w_broccoli_core.pdb inference.target_length=120 \
inference.contigs="30/A3-16/15/A29-47" inference.seed=182793 model.ipa.use_plddt=False

# sweep multiple contigs and lengths for scaffolding
python inference_rdiff_str.py --config-name=rna_scaffolding_training inference.task_name=scaffolding_iter \
inference.name=broc_scaffolding_iter inference.weights_path=RDiff_scaffolding_ckpts/scaffolding_500k.pth \
inference.target_pdb_path=examples/8k7w_broccoli_core.pdb inference.num_samples=5 inference.seed=42 \
inference.contigs="8+/A3-16/15+/A29-47/8+" \
inference.scaffold_iter.length_min=120 inference.scaffold_iter.length_max=350 \
inference.scaffold_iter.length_step=25 inference.scaffold_iter.max_contigs_per_length=200
```

## Pepper scaffolding
```
# single contig for scaffolding
python inference_rdiff_str.py --config-name=rna_scaffolding_training \
inference.name=pepper_scaffolding inference.weights_path=RDiff_scaffolding_ckpts/scaffolding_500k.pth \
inference.target_pdb_path=examples/7zj4_pepper_core.pdb inference.target_length=120 \
inference.contigs="30/E3-22/15/E35-47" inference.seed=123

# sweep multiple contigs and lengths for scaffolding
python inference_rdiff_str.py --config-name=rna_scaffolding_training inference.task_name=scaffolding_iter \
inference.name=pepper_scaffolding_iter inference.weights_path=RDiff_scaffolding_ckpts/scaffolding_500k.pth \
inference.target_pdb_path=examples/7zj4_pepper_core.pdb inference.num_samples=5 inference.seed=3721 \
inference.contigs="8+/E3-22/15+/E35-47/8+" \
inference.scaffold_iter.length_min=120 inference.scaffold_iter.length_max=350 \
inference.scaffold_iter.length_step=25 inference.scaffold_iter.max_contigs_per_length=200
```

The chain names in contigs should match the input motif PDB. In this repository,
`examples/8k7w_broccoli_core.pdb` uses chain `A`, while
`examples/7zj4_pepper_core.pdb` uses chain `E`.

### Contig sweep behavior

Use `inference.task_name=scaffolding_iter` when `inference.contigs` is a template
instead of one concrete contig. The sweep does not use the template string itself
as the final scaffold name. It first expands the template into concrete contigs,
randomly keeps at most `inference.scaffold_iter.max_contigs_per_length` contigs
for each target length, and then runs normal `scaffolding` on each selected
`(target_length, concrete_contig)` pair.

The sweep visits target lengths with:

```text
range(length_min, length_max + 1, length_step)
```

For the command above, this means:

```text
120, 145, 170, 195, 220, 245, 270, 295, 320, 345
```

Tokens in a contig template have these meanings:

- `30`: a fixed scaffold gap of 30 generated residues.
- `30-35`: a closed range, expanded to `30`, `31`, ..., `35`.
- `8+`: an open gap with minimum length 8. If any `+` token is present, the
  remaining residues are distributed across all `+` gaps so that each concrete
  contig has exactly the current `target_length`.
- `A3-16` or `E3-22`: fixed motif residues copied from the input motif PDB.

For example, broccoli uses:

```text
8+/A3-16/15+/A29-47/8+
```

The motif length is 33 residues: `A3-16` contributes 14 residues and `A29-47`
contributes 19 residues. At `target_length=120`, the three scaffold gaps must
sum to `120 - 33 = 87` residues, with minima `8`, `15`, and `8`. Therefore the
template expands to concrete contigs such as:

```text
8/A3-16/15/A29-47/64
8/A3-16/16/A29-47/63
...
64/A3-16/15/A29-47/8
```

There are 1653 concrete broccoli contigs at length 120. Because the example sets
`inference.scaffold_iter.max_contigs_per_length=200`, the script randomly keeps
200 of them for that length. The random subsampling is initialized from
`inference.seed`. Each selected concrete contig then receives a derived sampling
seed before diffusion is run.

For each selected concrete contig, `inference.num_samples` PDB files are written.
With `inference.num_samples=5`, the script writes sample indices `0` through `4`
for each selected contig.

Outputs are written under:

```text
inference.output_dir/inference.name/
```

For a concrete contig, the output basename is:

```text
<pdb_name>_scaffold_<target_length>_<concrete_contig_with_slashes_replaced_by_underscores>_<sample_i>.pdb
```

The matching fixed-sequence FASTA uses the same basename with `.fasta`.

For example, if the selected broccoli contig at length 120 is:

```text
8/A3-16/15/A29-47/64
```

sample `0` is written as:

```text
8k7w_broccoli_core_scaffold_120_8_A3-16_15_A29-47_64_0.pdb
```

# STEP3: Backbone quality checking and filtering

We will filter the backbones according to clashes and unreasonable distances between adjunct nucleotides.

```python
def analyze_pdb_clash(pdb_path: str, length: int) -> Dict:
    """
    Analyze PDB structure for clashes between C1' atoms and adjacent residue C1' distances.
    Returns dictionary with clash information and adjacent distance statistics.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', pdb_path)

    # Get all three atoms for each residue (original clash calculation)
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atoms.append(atom.get_coord())

    # Convert to numpy array for efficient distance calculation
    atoms_coords = np.array(atoms) # shape: (n*3, 3)

    # Calculate pairwise distances using vectorized operations
    atoms_coords_i = atoms_coords[:, np.newaxis, :]  # Shape: (n*3, 1, 3)
    atoms_coords_j = atoms_coords[np.newaxis, :, :]  # Shape: (1, n*3, 3)

    # Calculate distances using broadcasting
    distances = np.sqrt(np.sum((atoms_coords_i - atoms_coords_j) ** 2, axis=2))

    # Count clashes (distance < 1.1 Å)
    clash_mask = distances < 1.1  # shape: (n*3, n*3)
    np.fill_diagonal(clash_mask, False)  # Exclude self-distances

    # Reshape clash mask to group by residues (each residue has 3 atoms)
    clash_mask = clash_mask.reshape(length, 3, length, 3)

    # A residue has clash if any of its atoms clash with any atoms of other residues
    residue_clash = np.any(np.any(clash_mask, axis=3), axis=1)  # shape: (n_residues, n_residues)
    np.fill_diagonal(residue_clash, False)  # Exclude self-clashes

    # Count number of residues involved in clashes (if a residue clashes with multiple others, count once)
    residues_with_clash = np.any(residue_clash, axis=1)
    clash_ratio = np.sum(residues_with_clash) / length

    # Additional analysis: Calculate adjacent residue C1' distances
    c1_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'C1\'' in residue:
                    c1_atoms.append(residue['C1\''].get_coord())
                else:
                    # If C1' not found, use first available atom
                    for atom in residue:
                        c1_atoms.append(atom.get_coord())
                        break

    # Calculate adjacent residue C1' distances
    adjacent_distances = []
    for i in range(len(c1_atoms) - 1):
        dist = np.linalg.norm(c1_atoms[i] - c1_atoms[i + 1])
        adjacent_distances.append(dist)

    # Count adjacent distances exceeding 7Å
    adjacent_distances = np.array(adjacent_distances)
    long_distance_count = np.sum(adjacent_distances > 7.0)

    return {
        'clash_count': np.sum(residues_with_clash),
        'clash_ratio': clash_ratio,
        'adjacent_long_distance_count': long_distance_count,
    }
```

# STEP4: Sequence design with RDiff_seq

We generate sequences for the filtered backbones using RDiff_seq.
The motif sequence is expected to be fixed during design.

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/rdiff_seq_ss.py --no_wandb --mode inf_dir \
--inf_folder /path/to/pdb_dir --config.eval.fixed_part=True \
--config.eval.n_samples=150 \
--config.eval.model_path ribo_woss/checkpoints/step_600000.pth --config.seed 3721
```

# STEP5: Sequence filtering

Please filter the sequences with long continuous repeated nucleotides and repeated subsequences,
in addition to GC content.

# STEP6: Structure prediction with AF3 and further filtering

Predict the designed sequences with AlphaFold3 and filter the results with three
main criteria: motif preservation, consistency with the generated backbone, and
intuitive candidate prioritization.

## 6.1 Motif preservation

The first requirement is that the predicted AF3 structure preserves the intended
motif. Superpose the AF3 prediction to the reference motif using only the fixed
motif residues, then evaluate motif RMSD, motif lDDT, and whether the key
ligand-binding is retained.

If AF3 cannot reproduce the wild-type aptamer motif reliably, motif-RMSD based
filtering may be too strict. In that case, use a more conservative rule: the
designed sequence should not create large, unintended base-pairing between the
motif sequence and other scaffold regions.

Recommended motif checks:

- Motif RMSD after motif-only alignment.
- Ligand binding (if needed) and PAE.
- Absence of large unintended interactions between motif residues and distant
  scaffold residues, especially when the WT motif is not well reproduced by AF3.

## 6.2 Self-consistency with the generated backbone

The AF3 prediction results should be consistent with the backbones generated by RDiff_str to some extent,
but it is not recommended to directly select the highest value as it may miss some interesting cases.
It is more appropriate to select a threshold for filtering.

Potential metrics:

- TM-score between the generated backbone and the AF3 prediction.
- Global lDDT or per-residue lDDT against the generated backbone.
- Agreement between parsed secondary structures from the generated backbone and
  the AF3 prediction.

## 6.3 Intuitive recommendations for candidate designs

After motif and self-consistency filtering, prefer candidates whose AF3
predictions are relatively compact and internally organized. Very loose or
highly extended RNAs are likely to be more flexible, which may make the motif
less stably presented and reduce functional robustness.

Potential ranking signals (generated from AI):

- Smaller radius of gyration or end-to-end distance after normalizing by length.
- Higher long-range contact density outside the fixed motif.
- Fewer large flexible-looking arms or weakly connected domains.

These metrics are only initial suggestions. The exact definition and threshold
for this part should be further refined with AI-assisted analysis and manual
inspection.

One potentially interesting pattern is a circular topology in the
secondary structure: the designed scaffold appears to form a closed loop, and
this loop may be closed or stabilized by pseudoknot-like interactions. If the motif
is preserved in the circular, it may be more stable.
We could develop a specific metric to identify and prioritize such designs in the final candidate list.

Recommended final filtering order:

1. Remove AF3 predictions that do not preserve the motif.
2. Remove predictions that are inconsistent with the generated backbone by
   TM-score, lDDT, RMSD, or secondary-structure agreement.
3. Prioritize compact, internally connected candidates, including potentially
   stabilizing circular designs when the motif is still well-preserved.


# Example for FRET design, which has two functional motifs to preserve
The same pipeline can be applied to FRET design,
with the main difference being that there are two motifs to preserve instead of one.
The contig specification should include both motifs. For multi-chain motif PDBs,
the sampler automatically detects chain-aware scaffolding from the chains referenced
by the contig, so the standard `scaffolding_iter` task is sufficient. The filtering
criteria should check for preservation of both motifs in the AF3 predictions.

```bash
python inference_rdiff_str.py --config-name=rna_scaffolding_training inference.gpu_id=5 \
inference.name=fret_31A_scaffold inference.weights_path=RDiff_scaffolding_ckpts/scaffolding_500k.pth \
inference.task_name=scaffolding_iter inference.num_samples=2 inference.diffusion.num_t=200 \
inference.target_pdb_path=examples/FRET_translated_31A.pdb inference.target_length=145 \
inference.contigs="10+/A3-16/15+/A29-47/10+/E3-22/15+/E35-47/10+" \
inference.scaffold_iter.length_min=145 inference.scaffold_iter.length_max=145 \
inference.scaffold_iter.max_contigs_per_length=800  inference.seed=12768
```
