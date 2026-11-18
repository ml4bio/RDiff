import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pdb

def write_backbone_atoms_to_pdb(rna_bb_pos, sample_file_n, fixed_mask=None, seq=None, confidence=None):
    """Export backbone atoms to pdb file.
    rna_bb_pos: [N, 3, 3]
    """
    if not sample_file_n.endswith('.pdb'):
        pdb_path = sample_file_n + '.pdb'
    else:
        pdb_path = sample_file_n
    if seq is None:
        resd_name = 'A' * rna_bb_pos.shape[0]
    else:
        resd_name = seq
    atom_names_1 = ["C4'", "C1'", "N9"]
    atom_names_2 = ["C4'", "C1'", "N1"]
    temp_factor = 0.0
    chain_id = '0'
    i_code = ' '
    occupancy = 1.0

    if fixed_mask is None:
        fixed_mask = np.ones(rna_bb_pos.shape[0], dtype=bool)

    with open(pdb_path, 'w') as f:
        n_atoms = 0
        # iter over residues
        for idx_resd in range(rna_bb_pos.shape[0]):
            # skip not fixed residues
            if not fixed_mask[idx_resd]:
                continue
            if resd_name[idx_resd] in ['A', 'G']:
                atom_names = atom_names_1
            elif resd_name[idx_resd] in ['C', 'U']:
                atom_names = atom_names_2
            else:
                raise ValueError(f'Unknown residue name: {resd_name[idx_resd]}')

            temp_factor = 0.0 if confidence is None else \
                float(100 * confidence.reshape(-1)[idx_resd - 1])

            # iter over atoms in residue
            for idx_atom in range(rna_bb_pos.shape[1]):
                atom_name = atom_names[idx_atom]
                n_atoms += 1
                charge = atom_name[0]
                line_str = ''.join([
                    'ATOM  ',
                    '%5d' % n_atoms,
                    '  ' + atom_name + ' ' * (3 - len(atom_name)),
                    '   %s' % resd_name[idx_resd],
                    ' %s' % chain_id,
                    ' ' * (4 - len(str(idx_resd + 1))),
                    '%s' % str(idx_resd + 1),
                    '%s   ' % i_code,
                    '%8.3f' % rna_bb_pos[idx_resd, idx_atom, 0],
                    '%8.3f' % rna_bb_pos[idx_resd, idx_atom, 1],
                    '%8.3f' % rna_bb_pos[idx_resd, idx_atom, 2],
                    '%6.2f' % occupancy,
                    '%6.2f' % temp_factor,
                    ' ' * 10,
                    '%2s' % charge,
                    '%2s' % ' ',
                ])
                assert len(line_str) == 80, 'line length must be exactly 80 characters: ' + line_str
                f.write(line_str + '\n')
    return pdb_path

def write_backbone_atoms_to_pdb_with_missing(rna_bb_pos, ori_pdb_path, sample_file_n, seq,
                                             missing_mask, res_id_offset=0):
    """Export backbone atoms to pdb file with missing residues; keep original atoms."""

    temp_factor = 0.0
    chain_id = '0'
    i_code = ' '
    occupancy = 1.0

    # read original pdb
    with open(ori_pdb_path, 'r') as f:
        lines = f.readlines()
    # get atom lines, may need to check chain id in the future
    atom_lines = []
    for line in lines:
        if line.startswith('ATOM'):
            atom_lines.append(line)

    # collect infos from atom_lines
    ori_resd_dict = {}
    for line in atom_lines:
        resd_idx = int(line[22:26]) + res_id_offset
        if resd_idx not in ori_resd_dict:
            ori_resd_dict[resd_idx] = []
        ori_resd_dict[resd_idx].append(line)

    # write new pdb, if residue is missing, add rna_bb_pos to atom_lines
    with open(sample_file_n, 'w') as f:
        n_atoms = 0
        # iter over residues
        for idx_resd in range(len(seq)):
            resd_name = seq[idx_resd]
            if resd_name in ['A', 'G']:
                atom_names = ["C4'", "C1'", "N9"]
            elif resd_name in ['C', 'U']:
                atom_names = ["C4'", "C1'", "N1"]

            # if residue is missing, add rna_bb_pos to atom_lines
            if missing_mask[idx_resd]:
                # iter over atoms in residue
                for idx_atom in range(rna_bb_pos.shape[1]):
                    atom_name = atom_names[idx_atom]
                    n_atoms += 1
                    charge = atom_name[0]
                    line_str = ''.join([
                        'ATOM  ',
                        '%5d' % n_atoms,
                        '  ' + atom_name + ' ' * (3 - len(atom_name)),
                        '   %s' % resd_name,
                        ' %s' % chain_id,
                        ' ' * (4 - len(str(idx_resd + 1))),
                        '%s' % str(idx_resd + 1),
                        '%s   ' % i_code,
                        '%8.3f' % rna_bb_pos[idx_resd, idx_atom, 0],
                        '%8.3f' % rna_bb_pos[idx_resd, idx_atom, 1],
                        '%8.3f' % rna_bb_pos[idx_resd, idx_atom, 2],
                        '%6.2f' % occupancy,
                        '%6.2f' % temp_factor,
                        ' ' * 10,
                        '%2s' % charge,
                        '%2s' % ' ',
                    ])
                    assert len(line_str) == 80, 'line length must be exactly 80 characters: ' + line_str
                    f.write(line_str + '\n')
            else:
                # use original atom infos from atom_lines, just replace n_atoms and idx_resd
                resd_lines = ori_resd_dict[idx_resd+1]
                for line in resd_lines:
                    n_atoms += 1
                    line = line[:6] + '%5d' % n_atoms + line[11:22] + '%4d' % (idx_resd+1) + line[26:]
                    f.write(line)


def extract_3atom_from_pdb(pdb_parser, pdb_path, total_length):
    """
    Extract 3 atoms (C4', C1', N9 for purines; C4', C1', N1 for pyrimidines) from a PDB file.

    Return:
        pos: np.array [N, 3, 3], N is the number of residues, 3 is the number of atoms (C4', C1', N9/N1)
        missing mask: np.array [N], True if the residue is missing, False otherwise
        seq: str, sequence of the RNA
    """

    seq = ['-' for _ in range(total_length)]
    pos = np.zeros((total_length, 3, 3))
    missing_mask = np.ones(total_length)

    structure = pdb_parser.get_structure('', pdb_path)
    struct_chains = {chain.id.upper(): chain for chain in structure.get_chains()}
    chain = list(struct_chains.values())[0]

    for res in chain:
        res_id = res.id[1] - 1
        if res_id >= total_length:
            raise ValueError(f'Residue ID {res_id} exceeds total length {total_length}.')
        if res.get_resname() in ['A', 'G']:
            atom_names = ["C4'", "C1'", "N9"]
            seq[res_id] = res.get_resname()
        elif res.get_resname() in ['C', 'U']:
            atom_names = ["C4'", "C1'", "N1"]
            seq[res_id] = res.get_resname()
        else:
            atom_names = None

        # extract atom coordinates
        res_pos = np.zeros((3, 3))
        if atom_names is not None:
            for i, atom_name in enumerate(atom_names):
                if atom_name in res:
                    res_pos[i] = res[atom_name].coord

        # check if all three atoms are present
        if np.any((np.sum(res_pos, axis=-1) == 0)):
            missing_mask[res_id] = 1.
        else:
            missing_mask[res_id] = 0.
            pos[res_id] = res_pos

    return pos, missing_mask, ''.join(seq)


def extract_specific_residue(pdb_parser, pdb_path, residue_ids):
    """
    Extract 3 atoms (C4', C1', N9 for purines; C4', C1', N1 for pyrimidines) from a PDB file.

    Return:
        pos: np.array [N, 3, 3], N is the number of residues, 3 is the number of atoms (C4', C1', N9/N1)
        missing mask: np.array [N], True if the residue is missing, False otherwise
        seq: str, sequence of the RNA
    """

    total_length = len(residue_ids)
    seq = ['-' for _ in range(total_length)]
    pos = np.zeros((total_length, 3, 3))
    missing_mask = np.ones(total_length)

    structure = pdb_parser.get_structure('', pdb_path)
    struct_chains = {chain.id.upper(): chain for chain in structure.get_chains()}

    for i, residue_id in enumerate(residue_ids):
        chain_id, res_num = residue_id.split('_')
        # extract atom coordinates according to chain_id and res_num
        chain = struct_chains.get(chain_id.upper())
        if chain is None:
            raise ValueError(f'Chain {chain_id} not found in the PDB file.')
        res = chain[int(res_num)]
        res_id = i
        res_name = res.get_resname()
        if res_name in ['A', 'G']:
            atom_names = ["C4'", "C1'", "N9"]
        elif res_name in ['C', 'U']:
            atom_names = ["C4'", "C1'", "N1"]
        else:
            atom_names = None
        seq[res_id] = res_name

        # extract atom coordinates
        res_pos = np.zeros((3, 3))
        if atom_names is not None:
            for i, atom_name in enumerate(atom_names):
                if atom_name in res:
                    res_pos[i] = res[atom_name].coord

        # check if all three atoms are present
        if np.any((np.sum(res_pos, axis=-1) == 0)):
            missing_mask[res_id] = 1.
        else:
            missing_mask[res_id] = 0.
            pos[res_id] = res_pos


    return pos, missing_mask, ''.join(seq)



def plot_contact_map(ct, save_file=None):
    # ct: np.array [N, N], 0/1
    plt.figure(figsize=(10, 10))
    plt.imshow(ct, cmap='Greys', origin='upper')
    # plt.savefig(save_file.replace('.png', '_upper.png'))
    if save_file is not None:
        plt.savefig(save_file)
    plt.close()


def generate_contig_residue_mapping(contig_string: str, add_chain_name=False) -> dict:
    """
    Parses a contig string and generates a mapping from old residue numbers
    to new residue numbers.

    Args:
        contig_string: The contig string, e.g., "[30/A3-16/10/A29-47]".
                       Numbers represent the length of newly generated residues (or gaps).
                       "CStart-End" (e.g., "A3-16") indicates residues from Start to End
                       on chain C of the original PDB that need to be retained and renumbered.
        add_chain_name: If True, the chain name will be included in the mapping keys.

    Returns:
        A dictionary where keys are old residue identifiers (e.g., 3)
        and values are new residue numbers (integers).

    Raises:
        ValueError: If a segment in the contig string is malformed or if start residue
                    is greater than end residue in a range.
    """
    if not contig_string:
        return {}

    # Remove leading/trailing square brackets
    if contig_string.startswith('[') and contig_string.endswith(']'):
        processed_string = contig_string[1:-1]
    else:
        # If the user might input a string without brackets, use it directly
        processed_string = contig_string

    parts = processed_string.split('/')

    residue_map = {}
    current_new_residue_number = 1  # New residue numbering starts from 1

    # Regex to parse "CStart-End" format segments, e.g., "A3-16"
    # This assumes the Chain ID is a single uppercase letter.
    # If the Chain ID can be more complex (e.g., "AB", "ChA", "1A"), this regex needs to be modified.
    # For example, r"([A-Za-z0-9]+)(\d+)-(\d+)" can be used for more general Chain IDs.
    range_pattern = re.compile(r"([A-Z])(\d+)-(\d+)")

    for part in parts:
        if not part:  # Handle cases like "//" or a trailing "/" resulting in an empty string
            continue

        try:
            # Try to parse the part as a number (representing a gap or length of new residues)
            gap_length = int(part)
            current_new_residue_number += gap_length
        except ValueError:
            # If not a number, it should be a residue range, e.g., "A3-16"
            match = range_pattern.fullmatch(part)  # Use fullmatch to ensure the entire part matches the pattern
            if match:
                chain_id = match.group(1)
                start_res_num = int(match.group(2))
                end_res_num = int(match.group(3))

                if start_res_num > end_res_num:
                    raise ValueError(
                        f"In segment '{part}', start residue number {start_res_num} "
                        f"is greater than end residue number {end_res_num}."
                    )

                for old_res_idx in range(start_res_num, end_res_num + 1):
                    if add_chain_name:
                        old_residue_id = f"{chain_id}_{old_res_idx}"
                        residue_map[old_residue_id] = current_new_residue_number
                    else:
                        residue_map[old_res_idx] = current_new_residue_number
                    current_new_residue_number += 1
            else:
                # Handle malformed segment
                raise ValueError(
                    f"Contig string '{contig_string}' contains a malformed segment: '{part}'"
                )

    return residue_map


def dot_bracket_to_pairs(dot_bracket, nota, notb):
    pairs = []
    stack = []

    for i, char in enumerate(dot_bracket):
        if char == nota:
            stack.append(i)
        elif char == notb:
            if stack:
                j = stack.pop()
                pairs.append((j, i))
                pairs.append((i, j))
            else:
                raise ValueError(f'{notb} without matching!')
    return pairs

def parse_cwwss_db(db_input):
    # extract pairs from dot-bracket notation
    small = dot_bracket_to_pairs(db_input, '(', ')')
    medium = dot_bracket_to_pairs(db_input, '[', ']')
    large = dot_bracket_to_pairs(db_input, '{', '}')
    all_pairs = small + medium + large

    # return ssmap [N, N, 1], where N is the length of db_input
    # if considering non-canonical pairs and stacking, can modify here to change the 1 channel to 3 channels
    length = len(db_input)
    ssmap = np.zeros((length, length, 1), dtype=np.float32)

    for i, j in all_pairs:
        ssmap[i, j, 0] = 1.0
        # ssmap[j, i, 0] = 1.0

    return ssmap
