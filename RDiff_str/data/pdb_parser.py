"""Parse the FASTA sequence & atom coordinates from the PDB file."""

import os
import re
import logging
import warnings
import numpy as np
import torch
import random
import hashlib
import ml_collections
from datetime import datetime
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from .rna_constants import RNA_CONSTANTS

def get_rand_str():
    """Get a randomized string.

    Args: n/a

    Returns:
    * rand_str: randomized string
    """

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    rand_val = random.random()
    rand_str_raw = '%s_%f' % (timestamp, rand_val)
    rand_str = hashlib.md5(rand_str_raw.encode('utf-8')).hexdigest()

    return rand_str


class PdbParserError(Exception):
    """Exceptions raised by the <PdbParser> class."""


class PdbParser():
    """Parser for PDB files.

    Note: GZ-compressed PDB file can also be provided (*.gz).
    """

    def __init__(self):
        """Constructor function."""

        # self.check_mode = check_mode  # choices: 'strict' OR 'lenient'

        self.frame_info = RNA_CONSTANTS

        self.atom_names = [
            'C4', "C1'", 'N9', 'N1', 'C2', "C4'", 'P',
            'N2', 'O6', "C2'", 'N4', "C3'", "O4'", 'N3',
            'OP2', 'OP1', "O2'", 'N6', "O5'", 'N7', "C5'",
            'C8', 'O2', 'C5', 'C6', 'O4', "O3'",
        ]

        self.atom_index_dict = {self.atom_names[i]: i for i in range(len(self.atom_names))}
        self.n_atoms_per_resd = len(self.atom_names)
        self.atom_mask_thres = 0.5  # minimal ratio of atoms w/ coordinates
        self.parser = PDBParser()
        # suppress all the warnings raised by <BioPython>
        warnings.simplefilter('ignore', BiopythonWarning)

    # def run(self, pdb_fpath, fas_fpath=None, structure=None, chain_id=None, aa_seq=None, sel=True):
    def run(self, pdb_fpath, fas_fpath=None, structure=None, chain_id=None, aa_seq=None, sel=True):
        """Parse the PDB file.

        Args:
        """

        # show the greeting message
        logging.debug('parsing the PDB file: %s (chain ID: <%s>)', pdb_fpath, chain_id)
        # attempt to parse the PDB file
        try:
            # check inputs
            if not os.path.exists(pdb_fpath):
                raise PdbParserError('PDB_FILE_NOT_FOUND')
            if (aa_seq is None and not os.path.exists(fas_fpath)):
                raise PdbParserError('FASTA_FILE_NOT_FOUND')

            if aa_seq is None:
                raise PdbParserError('AA_SEQ_NOT_FOUND')

            # parse the PDB file w/ biopython
            if structure is None:
                structure = self.__get_structure(pdb_fpath)

            # find the first chain matching the chain ID
            chain = self.__get_chain(structure, chain_id)

            # obtain atom coordinates & validness masks
            atom_cords, atom_masks = self.__get_atoms(chain, aa_seq)

            if sel:
                atom_cords, atom_masks = self.get_atoms_sel(aa_seq,
                                                            torch.FloatTensor(atom_cords),
                                                            torch.FloatTensor(atom_masks),
                                                            )
                atom_cords, atom_masks = atom_cords.data.cpu().numpy(), atom_masks.data.cpu().numpy()

            # set the error message to None
            error_msg = None

        except PdbParserError as error:
            logging.warning('PDB file path: %s / Error: %s', pdb_fpath, error)
            aa_seq, atom_cords, atom_masks, structure, error_msg = None, None, None, None, error

        return aa_seq, atom_cords, atom_masks, structure, error_msg

    def __get_structure(self, path):
        """Get the structure from the PDB file."""

        try:
            with open(path, 'r') as i_file:
                structure = self.parser.get_structure(get_rand_str(), i_file)
        except PDBConstructionException as error:
            raise PdbParserError('BIOPYTHON_FAILED_TO_PARSE') from error

        return structure

    @classmethod
    def __get_chain(cls, structure, chain_id):
        """Get the first chain matching the specified chain ID (could be None)."""

        chain = None
        for model in structure:
            for chain_curr in model:
                if (chain_id is None) or (chain_curr.get_id() == chain_id):
                    chain = chain_curr
                    break
            if chain is not None:
                break

        # check whether the specified chain has been found
        if chain is None:
            raise PdbParserError('CHAIN_NOT_FOUND')

        return chain

    def __get_atoms(self, chain, aa_seq):
        """Get atom coordinates & masks from the specified chain."""

        # obtain all the segments' information
        seg_infos = self.__get_seg_infos(chain)

        # find the valid offset for residue indices, if possible
        offset = self.__get_offset(seg_infos, aa_seq)

        # obtain atom coordinates & masks
        seq_len = len(aa_seq)
        atom_cords = np.zeros((seq_len, self.n_atoms_per_resd, 3), dtype=np.float32)
        atom_masks = np.zeros((seq_len, self.n_atoms_per_resd), dtype=np.int8)

        for residue in chain:
            # skip hetero-residues, and obtain the residue's index
            het_flag, idx_resd, _ = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues

            # update atom coordinates & masks
            for idx_atom, atom_name in enumerate(self.atom_names):
                if residue.has_id(atom_name):
                    atom_cords[idx_resd + offset, idx_atom] = residue[atom_name].get_coord()
                    atom_masks[idx_resd + offset, idx_atom] = 1

        # check whether sufficient ratio of atoms have coordinates
        # if (self.check_mode == 'strict') and (np.mean(atom_masks) < self.atom_mask_thres):
        #     raise PdbParserError('INSUFFICIENT_ATOMS_W_COORDINATES')

        return atom_cords, atom_masks

    def get_atoms_sel(self, aa_seq, atom_tns_all, mask_tns_all, topn_cords=-1):
        """Get per-atom 3D coordinates or validness masks for selected atom(s).

        Args:
        * aa_seq: amino-acid sequence
        * atom_tns_all: full per-atom 3D coordinates (L x M x 3) or validness masks (L x M)
        * atom_names_sel: list of selected atom names of length M'

        Returns:
        * atom_tns_sel: selected per-atom 3D coordinates (L x M' x 3) or validness masks (L x M')

        Note:
        * If only one atom name if provided, then its corresponding dimension is squeezed.
        """

        # initialization
        device = atom_tns_all.device
        n_atoms = self.frame_info.ATOM_NUM_MAX if topn_cords < 0 else topn_cords

        # build the indexing tensor for selected atom(s)
        idxs_vec_dict = {}  # atom indices
        msks_vec_dict = {}  # atom indices' validness masks

        for resd_name in ['A', 'U', 'G', 'C']:
            atom_names_all = self.frame_info.ATOM_NAMES_PER_RESD[resd_name]
            idxs_vec_np = np.zeros((n_atoms), dtype=np.int64)
            msks_vec_np = np.zeros((n_atoms), dtype=np.int8)
            for idx_atom_sel, atom_name_sel in enumerate(atom_names_all):

                if topn_cords > 0 and idx_atom_sel >= topn_cords:
                    continue

                if atom_name_sel in self.atom_index_dict.keys():
                    idxs_vec_np[idx_atom_sel] = self.atom_index_dict[atom_name_sel]
                    msks_vec_np[idx_atom_sel] = 1

            idxs_vec_dict[resd_name] = idxs_vec_np
            msks_vec_dict[resd_name] = msks_vec_np

        # remove additional keys
        valid_keys = set(['A', 'U', 'G', 'C'])
        idxs_vec_dict = {k: v for k, v in idxs_vec_dict.items() if k in valid_keys}
        msks_vec_dict = {k: v for k, v in msks_vec_dict.items() if k in valid_keys}
        # print(idxs_vec_dict.items())

        aa_seq = aa_seq.replace('T', 'U')
        aa_seq = ''.join(['A' if x not in valid_keys else x for x in aa_seq])

        # # determine the overall indexing tensor based on the amino-acid sequence
        idxs_mat_full_np = np.stack([idxs_vec_dict[x] for x in aa_seq], axis=0)
        msks_mat_full_np = np.stack([msks_vec_dict[x] for x in aa_seq], axis=0)

        idxs_mat_full = torch.tensor(idxs_mat_full_np, dtype=torch.int64, device=device)  # L x M'
        msks_mat_full = torch.tensor(msks_mat_full_np, dtype=torch.int64, device=device)  # L x M'

        # get per-atom 3D coordinates or validness masks for specified residue(s) & atom(s)
        if atom_tns_all.ndim == 2:
            atom_tns_sel = msks_mat_full * torch.gather(atom_tns_all, 1, idxs_mat_full)
            mask_tns_sel = msks_mat_full * torch.gather(mask_tns_all, 1, idxs_mat_full)
        else:
            n_dims_addi = atom_tns_all.shape[-1]
            atom_tns_sel = msks_mat_full.unsqueeze(dim=2) * torch.gather(
                atom_tns_all, 1, idxs_mat_full.unsqueeze(dim=2).repeat(1, 1, n_dims_addi))

            mask_tns_sel = msks_mat_full * torch.gather(mask_tns_all, 1, idxs_mat_full)

        # squeeze the dimension if only one atom is selected
        if n_atoms == 1:
            atom_tns_sel.squeeze_(dim=1)

        return atom_tns_sel, mask_tns_sel

    @classmethod
    def __get_seg_infos(cls, chain):
        """Get discontinous segments' information for the specified chain."""

        seg_infos = []
        for residue in chain:
            # obtain the current residue's basic information
            resd_name = residue.get_resname()
            # resd_name_1c = resd_name
            resd_name_1c = resd_name.replace('D', '')  # obtain the 1-character residue name
            het_flag, idx_resd, ins_code = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues
            if ins_code.strip() != '':
                raise PdbParserError('HAS_INSERTED_RESIDUES')

            # update the last segment, or add a new segment
            if len(seg_infos) >= 1 and seg_infos[-1]['ie'] == idx_resd:
                seg_infos[-1]['ie'] += 1
                seg_infos[-1]['seq'] += resd_name_1c
            else:
                seg_infos.append({
                    'ib': idx_resd,  # inclusive
                    'ie': idx_resd + 1,  # exclusive
                    'seq': resd_name_1c,  # 20 amino-acids + '.' for wild-card matches
                })

        # sort discontinous segments in the descending order of segment length
        seg_infos.sort(key=lambda x: x['ie'] - x['ib'], reverse=True)

        return seg_infos

    @classmethod
    def __get_offset(cls, seg_infos, aa_seq):
        """Get a valid offset of residue indices in ATOM records & amino-acid sequence."""

        offset_list = None
        for seg_info in seg_infos:
            regex = re.compile(seg_info['seq'])
            if offset_list is None:
                offset_list = [m.start() - seg_info['ib'] for m in re.finditer(regex, aa_seq)]
            else:
                offset_list_new = []
                for offset in offset_list:
                    if re.search(regex, aa_seq[seg_info['ib'] + offset:seg_info['ie'] + offset]):
                        offset_list_new.append(offset)
                offset_list = offset_list_new

            if len(offset_list) == 0:
                return -1
                # raise PdbParserError('NO_VALID_OFFSET')

        return offset_list[0]  # use the first valid offset

    def get_bb_atoms(self, seq, coord):
        seq_len = len(seq)
        bb_coord = np.zeros((seq_len, 3, 3), dtype=np.float32)
        bb_mask = np.zeros((seq_len), dtype=np.float32)
        for i in range(seq_len):
            resd_atom_name = self.frame_info.ATOM_NAMES_PER_RESD[seq[i]]
            bb_coord[i, 0] = coord[i, resd_atom_name.index("C4'")]
            bb_coord[i, 1] = coord[i, resd_atom_name.index("C1'")]
            if seq[i] in ['A', 'G']:
                bb_coord[i, 2] = coord[i, resd_atom_name.index("N9")]
            elif seq[i] in ['C', 'U']:
                bb_coord[i, 2] = coord[i, resd_atom_name.index("N1")]
            else:
                bb_coord[i, 2] = np.zeros((3), dtype=np.float32)
            bb_mask[i] = np.all(bb_coord[i] != 0.)
        return bb_coord, bb_mask