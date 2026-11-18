from tools.contig_iterate import generate_contigs_from_spec
import random
import os
import subprocess
import sys
import pdb

def iterate_run_broc():
    segment1_str = "A3-16"
    segment2_str = "A29-47"

    min_gap1_len = 8
    min_gap2_len = 10
    min_gap3_len = 8

    length_list = [_ for _ in range(70, 351, 25)]  # Lengths from 70 to 351 with step of 25
    print(f"Length list {len(length_list)}: {length_list}")
    length_contigs = []

    # generate and sample possible contigs
    for length in length_list:
        contigs = generate_contigs_from_spec(
            length, segment1_str, segment2_str, min_gap1_len, min_gap2_len, min_gap3_len
        )

        if length <= 70:
            contigs = random.sample(contigs, 30)

        if len(contigs) > 200:
            contigs = random.sample(contigs, 200)

        if contigs:
            for contig in contigs:
                length_contigs.append((length, contig))

    for length, contig in length_contigs:
        print(f"Length: {length}, Contig: {contig}")
        seed = random.randint(0, 1000000)

        cmd = f"python inference_rdiff_str.py --config-name=rna_scaffolding_training "\
          f"inference.name=broc_scaffolding_500K/broc_{length}length_{contig.replace('/', '_')} inference.weights_path=../RDiff_scaffolding_ckpts/scaffolding_500k.pth "\
          f"inference.target_pdb_path=examples/8k7w_broccoli_core.pdb inference.target_length={length} " \
          f"inference.contigs={contig} inference.seed={seed} inference.num_samples=5"

        print(f"Running command: {cmd}")

        # run the command and only continue if it succeeds
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
            sys.exit(1)

        print("Length:", length, "Contig:", contig, "Seed:", seed, ' done!')

def iterate_run_pepper():
    segment1_str = "A3-22"
    segment2_str = "A35-47"

    min_gap1_len = 8
    min_gap2_len = 10
    min_gap3_len = 8

    length_list = [_ for _ in range(70, 351, 25)]  # Lengths from 70 to 351 with step of 25
    print(f"Length list {len(length_list)}: {length_list}")
    length_contigs = []

    # generate and sample possible contigs
    for length in length_list:
        contigs = generate_contigs_from_spec(
            length, segment1_str, segment2_str, min_gap1_len, min_gap2_len, min_gap3_len
        )

        if length <= 70:
            contigs = random.sample(contigs, 30)

        if len(contigs) > 200:
            contigs = random.sample(contigs, 200)

        if contigs:
            for contig in contigs:
                length_contigs.append((length, contig))

    for length, contig in length_contigs:
        print(f"Length: {length}, Contig: {contig}")
        seed = random.randint(0, 1000000)

        cmd = f"python inference_rdiff_str.py --config-name=rna_scaffolding_training "\
          f"inference.name=pepper_scaffolding_500K/pep_{length}length_{contig.replace('/', '_')} inference.weights_path=../RDiff_scaffolding_ckpts/scaffolding_500k.pth "\
          f"inference.target_pdb_path=examples/7zj4_pepper_core.pdb inference.target_length={length} " \
          f"inference.contigs={contig} inference.seed={seed} inference.num_samples=5"

        print(f"Running command: {cmd}")

        # run the command and only continue if it succeeds
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
            sys.exit(1)

        print("Length:", length, "Contig:", contig, "Seed:", seed, ' done!')

if __name__ == "__main__":
    # iterate_run_broc()
    iterate_run_pepper()
