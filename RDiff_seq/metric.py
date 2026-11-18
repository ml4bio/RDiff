import numpy as np
from Bio import pairwise2
from tqdm import tqdm
import pdb
from datasets import utils as du
import os
import subprocess
from datasets.utils import parse_pdb_direct

NUM_TO_LETTER = np.array(['A', 'G', 'C', 'U'])


def test_recovery(model, dataset, device, sample_fn, name='TEST'):
    recovery = []
    # t = tqdm(dataset)
    # for i, data in enumerate(t):
    total_num = len(dataset)
    for i, data in enumerate(dataset):
        for key in data.keys():
            data[key] = data[key].to(device)

        sample = sample_fn(model, data)  # [n_samples, seq_len]
        # print("successful sampling")
        recovery_ = sample.eq(data['seq']).float().mean().cpu().numpy()
        recovery.append(recovery_)
        # t.set_description(f"{np.mean(recovery):.4f}")
        if i % 100 == 0:
            print(f"{i/total_num}: {np.mean(recovery):.4f}")
    print(f'{name} recovery median: {np.median(recovery):.4f} mean: {np.mean(recovery):.4f} +- {np.std(recovery):.4f}')
    return recovery

def seq_sim(seq0, seq1):
    """Sequence similarity between two sequences"""
    alignments = pairwise2.align.localxx(seq0, seq1)

    if len(alignments) > 0:
        c = 0
        for i in range(len(alignments[0].seqA)):
            if alignments[0].seqA[i] == alignments[0].seqB[i]:
                c += 1
        score = c / len(alignments[0].seqA)
        return score
    else:
        return None


def sequence_eval(model, dataloader, device, sample_fn, n_samples, eval_dir, name='TEST',
                  save_fasta=False):
    """calculate recovery rate and sequence similarity"""

    recovery = []
    seq_similarity = []
    res_dict = {}
    # name_list = np.load("./exp_test.npy")
    name_list = dataloader.dataset.data_names
    # parent_path = os.path.dirname(eval_dir)
    fasta_dir = os.path.join(eval_dir, 'fasta')
    if save_fasta:
        os.makedirs(fasta_dir, exist_ok=True)

    # Iter over dataloader
    t = tqdm(dataloader)
    for i, data in enumerate(t):
        for key in data.keys():
            data[key] = data[key].to(device)
        # Sample from model
        sample = sample_fn(model, data)  # [n_samples, seq_len]

        recovery_ = sample.eq(data['seq']).float().mean().cpu().numpy()
        recovery.append(recovery_)
        t.set_description(f"{np.mean(recovery):.4f}")

        pdb_id = name_list[i]
        # Decode back to sequence
        assert n_samples == 1
        for j in range(n_samples):
            seq0 = ''.join(list(NUM_TO_LETTER[data['seq'][0].cpu().numpy()]))
            seq1 = ''.join(list(NUM_TO_LETTER[sample[j].cpu().numpy()]))
            seq_similarity_ = seq_sim(seq0, seq1)
            if seq_similarity_ is None:
                seq_similarity_ = 0.0
            seq_similarity.append(seq_similarity_)
            res_dict[pdb_id] = [len(seq0), recovery_, seq_similarity_]

            # Save as fasta
            if save_fasta:
                du.sample_to_fasta(sample[j], pdb_id,
                                   os.path.join(fasta_dir, pdb_id + '.fasta'))

    print(f'{name} recovery median: {np.median(recovery):.4f} mean: {np.mean(recovery):.4f} +- {np.std(recovery):.4f}')
    print(f'{name} sequence similarity mean: {np.mean(seq_similarity):.4f} +- {np.std(seq_similarity):.4f}')
    return res_dict, fasta_dir
