import argparse
import os
import pickle
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["nt1", "nt2", "type", "classification-1"]

NC_TYPE_TO_NUM = {
    "tWW": 1,
    "cWH": 2,
    "cHW": 2,
    "tWH": 3,
    "tHW": 3,
    "cWS": 4,
    "cSW": 4,
    "tWS": 5,
    "tSW": 5,
    "cHH": 6,
    "tHH": 7,
    "cHS": 8,
    "cSH": 8,
    "tHS": 9,
    "tSH": 9,
    "cSS": 10,
    "tSS": 11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ssmap3/*.npy and NC_sparse/*.npz from one interaction CSV file."
    )
    parser.add_argument("--csv", required=True, help="Input interaction CSV from RNApolis/DSSR/custom annotation.")
    parser.add_argument("--out_dir", required=True, help="Output root containing ssmap3/ and NC_sparse/.")
    parser.add_argument("--target", default=None, help="Output target name. Defaults to CSV basename.")
    parser.add_argument("--seq", default=None, help="Optional FASTA/sequence file used to determine RNA length.")
    parser.add_argument("--length", type=int, default=None, help="RNA length. Required if --seq is not provided.")
    parser.add_argument(
        "--idx_mapping",
        default=None,
        help=(
            "Optional pickle residue mapping. Supports either {target: {residue_id: index}} "
            "or {residue_id: index}."
        ),
    )
    parser.add_argument(
        "--residue_offset",
        type=int,
        default=1,
        help="Fallback offset for contiguous PDB residue numbers. index = residue_id - residue_offset.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_sequence_length(seq_path: str) -> int:
    sequence = []
    with open(seq_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            sequence.append(line)
    length = len("".join(sequence))
    if length <= 0:
        raise ValueError(f"No sequence residues found in {seq_path}")
    return length


def validate_columns(dataframe: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns in interaction CSV: {missing}")


def extract_residue_id(label: object) -> int:
    text = str(label).strip()
    parts = text.split(".")
    residue_part = parts[-1]
    matches = re.findall(r"-?\d+", residue_part)
    if not matches:
        raise ValueError(f"Failed to extract residue number from nucleotide label: {label}")
    return int(matches[-1])


def load_idx_mapping(path: Optional[str]) -> Optional[Dict]:
    if path is None:
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def lookup_mapping(mapping: Dict, target: str, residue_id: int) -> int:
    target_mapping = mapping.get(target, mapping)
    if residue_id in target_mapping:
        return int(target_mapping[residue_id])
    residue_key = str(residue_id)
    if residue_key in target_mapping:
        return int(target_mapping[residue_key])
    raise KeyError(f"Residue id {residue_id} is missing in idx_mapping for target {target}")


def map_nt_label(label: object, target: str, idx_mapping: Optional[Dict], residue_offset: int) -> int:
    residue_id = extract_residue_id(label)
    if idx_mapping is not None:
        return lookup_mapping(idx_mapping, target=target, residue_id=residue_id)
    return residue_id - residue_offset


def add_pair(matrix: np.ndarray, channel: int, i: int, j: int) -> None:
    length = matrix.shape[1]
    if i < 0 or i >= length or j < 0 or j >= length:
        raise ValueError(f"Interaction index ({i}, {j}) is outside RNA length {length}")
    matrix[channel, i, j] = 1.0
    matrix[channel, j, i] = 1.0


def build_outputs(
    dataframe: pd.DataFrame,
    target: str,
    length: int,
    idx_mapping: Optional[Dict],
    residue_offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ssmap3 = np.zeros((3, length, length), dtype=np.float32)
    sparse_triples = set()

    for _, record in dataframe.iterrows():
        interaction_type = normalize_text(record["type"]).lower()
        pair_type = normalize_text(record["classification-1"])

        if interaction_type not in ("base pair", "stacking"):
            continue

        nt1_idx = map_nt_label(record["nt1"], target=target, idx_mapping=idx_mapping, residue_offset=residue_offset)
        nt2_idx = map_nt_label(record["nt2"], target=target, idx_mapping=idx_mapping, residue_offset=residue_offset)

        if interaction_type == "stacking":
            add_pair(ssmap3, 2, nt1_idx, nt2_idx)
            continue

        if pair_type == "cWW":
            add_pair(ssmap3, 0, nt1_idx, nt2_idx)
        else:
            add_pair(ssmap3, 1, nt1_idx, nt2_idx)

        if pair_type in NC_TYPE_TO_NUM:
            sparse_triples.add((nt1_idx, nt2_idx, NC_TYPE_TO_NUM[pair_type]))

    if sparse_triples:
        triples = sorted(sparse_triples)
        nt1, nt2, type_num = zip(*triples)
        return (
            ssmap3,
            np.asarray(nt1, dtype=np.int32),
            np.asarray(nt2, dtype=np.int32),
            np.asarray(type_num, dtype=np.int32),
        )

    return (
        ssmap3,
        np.asarray([], dtype=np.int32),
        np.asarray([], dtype=np.int32),
        np.asarray([], dtype=np.int32),
    )


def main() -> None:
    args = parse_args()
    target = args.target or os.path.splitext(os.path.basename(args.csv))[0]

    if args.seq is None and args.length is None:
        raise ValueError("Provide --seq or --length so ssmap3 has the correct residue dimension.")
    length = read_sequence_length(args.seq) if args.seq is not None else args.length

    dataframe = pd.read_csv(args.csv, sep=",", index_col=False)
    validate_columns(dataframe)
    idx_mapping = load_idx_mapping(args.idx_mapping)

    ssmap3, nt1, nt2, type_num = build_outputs(
        dataframe=dataframe,
        target=target,
        length=length,
        idx_mapping=idx_mapping,
        residue_offset=args.residue_offset,
    )

    ssmap_dir = os.path.join(args.out_dir, "ssmap3")
    nc_sparse_dir = os.path.join(args.out_dir, "NC_sparse")
    os.makedirs(ssmap_dir, exist_ok=True)
    os.makedirs(nc_sparse_dir, exist_ok=True)

    ssmap_path = os.path.join(ssmap_dir, f"{target}.npy")
    nc_sparse_path = os.path.join(nc_sparse_dir, f"{target}.npz")
    np.save(ssmap_path, ssmap3)
    np.savez_compressed(nc_sparse_path, nt1=nt1, nt2=nt2, type_num=type_num)

    print(f"Saved {ssmap_path}: shape={ssmap3.shape}")
    print(f"Saved {nc_sparse_path}: nc_pairs={len(nt1)}")


if __name__ == "__main__":
    main()
