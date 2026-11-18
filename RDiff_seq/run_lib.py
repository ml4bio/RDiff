import os
import torch
from tqdm import tqdm
import numpy as np
import random
from models import *
from utils import *
import losses
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn, get_partial_sampling_fn
from datasets import utils as du
import functools
import tree


FIXED_SEQ_FASTA_EXTENSIONS = ('.fasta', '.fa')
FIXED_SEQ_ALLOWED_CHARS = set('AGCU-')


def _read_fixed_seq_fasta(fasta_path):
    records = []
    header = None
    seq_lines = []

    with open(fasta_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == '':
                continue
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(seq_lines).upper()))
                header = line[1:].strip()
                seq_lines = []
            else:
                if header is None:
                    header = ''
                seq_lines.append(line)

    if header is not None:
        records.append((header, ''.join(seq_lines).upper()))

    if len(records) == 0:
        raise ValueError(f'No FASTA records found in fixed sequence file: {fasta_path}')

    for record_header, seq in records:
        invalid_chars = sorted(set(seq) - FIXED_SEQ_ALLOWED_CHARS)
        if invalid_chars:
            raise ValueError(
                f"Invalid fixed sequence characters {invalid_chars} in {fasta_path}"
                f" record '{record_header}'. Allowed characters are A/G/C/U/-."
            )

    return records


def _find_fixed_seq_fasta(pdb_path, fixed_seq_path):
    pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]

    if fixed_seq_path != '':
        fixed_seq_path = os.path.abspath(fixed_seq_path)
        if os.path.isfile(fixed_seq_path):
            return fixed_seq_path
        if not os.path.isdir(fixed_seq_path):
            raise FileNotFoundError(f'fixed_seq_path does not exist: {fixed_seq_path}')
        search_dir = fixed_seq_path
    else:
        search_dir = os.path.dirname(os.path.abspath(pdb_path))

    candidates = [
        os.path.join(search_dir, pdb_stem + ext)
        for ext in FIXED_SEQ_FASTA_EXTENSIONS
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"No fixed sequence FASTA found for PDB {pdb_path}. Tried: {', '.join(candidates)}"
    )


def _get_fixed_seq_for_pdb(config, pdb_path, expected_len):
    fixed_seq_config = config.eval.fixed_seq.strip()
    if fixed_seq_config != '':
        fixed_seq = fixed_seq_config.upper()
        source = '--config.eval.fixed_seq'
    else:
        fixed_seq_path = getattr(config.eval, 'fixed_seq_path', '').strip()
        fasta_path = _find_fixed_seq_fasta(pdb_path, fixed_seq_path)
        fixed_seq = _read_fixed_seq_fasta(fasta_path)[0][1]
        source = fasta_path

    invalid_chars = sorted(set(fixed_seq) - FIXED_SEQ_ALLOWED_CHARS)
    if invalid_chars:
        raise ValueError(
            f"Invalid fixed sequence characters {invalid_chars} from {source}. "
            'Allowed characters are A/G/C/U/-.'
        )
    if len(fixed_seq) != expected_len:
        raise ValueError(
            f'Fixed sequence length does not match PDB length for {pdb_path}: '
            f'{len(fixed_seq)} from {source} vs {expected_len} residues.'
        )

    return fixed_seq


def set_random_seed(config):
    seed = config.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_condition_scalar_dim(config):
    total = config.data.ss_channel * 8
    if getattr(config.data, 'use_nc_detail_cond', False):
        total += config.model.nc_embedding_dim
    return total


def update_edge_in_dim(config):
    base_edge_scalar_dim = getattr(config.model, 'base_edge_scalar_dim', config.model.edge_in_dim[0])
    config.model.edge_in_dim = (
        base_edge_scalar_dim + get_condition_scalar_dim(config),
        config.model.edge_in_dim[1],
    )




def vpsde_inference_dir(config, workdir, inference_dir):
    """Runs inference for RNA inverse design in a given dir."""
    input_path = os.path.abspath(inference_dir)
    if os.path.isfile(input_path):
        if not input_path.lower().endswith('.pdb'):
            raise ValueError(f"Inference input file must be a .pdb file, got: {inference_dir}")
        inference_dir = os.path.dirname(input_path)
        file_list = [os.path.basename(input_path)]
    elif os.path.isdir(input_path):
        inference_dir = input_path
        file_list = sorted([file for file in os.listdir(inference_dir) if file.lower().endswith('.pdb')])
        if len(file_list) == 0:
            raise ValueError(f"No .pdb files found in inference directory: {inference_dir}")
    else:
        raise FileNotFoundError(f"Inference path not found: {inference_dir}")

    # update edge_in_dim
    update_edge_in_dim(config)
    # Initialize model
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    print('model size: {:.1f}MB'.format(model_size))

    # Checkpoint name
    if config.eval.model_path == '':
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "best.pth")
    else:
        checkpoint_path = config.eval.model_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    # Load checkpoint
    state = restore_checkpoint(checkpoint_path, state, device=config.device)
    ema.copy_to(model.parameters())

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scalar and inverse scalar
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup new sampling function
    if config.eval.partial_sampling:
        test_sampling_single_fn = get_partial_sampling_fn(config, noise_scheduler, config.eval.sampling_steps, inverse_scaler)
    else:
        test_sampling_single_fn = get_sampling_fn(config, noise_scheduler, config.eval.sampling_steps, inverse_scaler)
    pdb2data = functools.partial(du.PDBtoData, num_posenc=config.data.num_posenc,
                                 num_rbf=config.data.num_rbf, knn_num=config.data.knn_num)

    # determine the number of runs based on n_samples and eval_batch_size
    eval_batch_size = config.eval.batch_size
    if config.eval.n_samples > eval_batch_size:
        n_runs = config.eval.n_samples // eval_batch_size
        if config.eval.n_samples % eval_batch_size == 0:
            curr_n_samples = [eval_batch_size] * n_runs
        else:
            n_runs += 1
            curr_n_samples = [eval_batch_size] * (n_runs - 1) + [config.eval.n_samples % eval_batch_size]
    else:
        n_runs = 1
        curr_n_samples = [config.eval.n_samples]

    # create fasta dir
    if config.eval.output_fasta_file != '':
        # if fasta_file_path without a directory, use inference_dir as directory
        if os.path.isabs(config.eval.output_fasta_file):
            fasta_file_path = config.eval.output_fasta_file
        else:
            fasta_file_path = os.path.join(inference_dir, config.eval.output_fasta_file)
        # if is a path with directories not exist, create directories
        fasta_dir = os.path.dirname(fasta_file_path)
        if not os.path.exists(fasta_dir):
            os.makedirs(fasta_dir)
    else:
        fasta_file_path = os.path.join(inference_dir, 'output_seq.fasta')
    fasta_file = open(fasta_file_path, 'w')

    # setup condition maps for custom inference inputs
    base_ss_map = None
    if config.data.ss_channel != 0:
        if config.eval.input_ss_npy != '':
            base_ss_map = du.load_dense_ss_map(config.eval.input_ss_npy)
        elif config.eval.input_db != '':
            if config.data.ss_channel != 1:
                raise ValueError('Dot-bracket input only supports ss_channel == 1')
            base_ss_map = parse_cwwss_db(config.eval.input_db)
        else:
            raise ValueError('Please provide input_ss_npy or input_db when ss_channel > 0')

    for pdb_file in tqdm(file_list):
        pdb_id = pdb_file.replace('.pdb', '')
        pdb_path = os.path.join(inference_dir, pdb_file)

        cond_map = du.build_condition_map(base_ss_map, config.data.ss_channel)
        nc_type_map = None
        if config.data.use_nc_detail_cond:
            if config.eval.input_nc_npz == '':
                raise ValueError('Please provide input_nc_npz when use_nc_detail_cond=True')
            seq, _, _ = du.parse_pdb_direct(pdb_path)
            nc_type_map = du.load_nc_sparse_type_map(
                config.eval.input_nc_npz,
                length=len(seq),
                nc_channel=config.data.nc_channel,
                symmetric=True,
            )
        fixed_seq_str = None

        for i in range(n_runs):
            curr_n_sample = curr_n_samples[i]

            struct_data = pdb2data(pdb_path, ss_map=cond_map, nc_type_map=nc_type_map)

            if config.eval.partial_sampling:
                # replace node_s with config.eval.target_seq
                target_seq = config.eval.target_seq
                target_seq_num = du.seq_to_num(target_seq)
                target_node_s = (target_seq_num.unsqueeze(-1) == torch.arange(4).unsqueeze(0)).float()
                assert target_node_s.shape == struct_data['node_s'].shape
                struct_data['node_s'] = target_node_s

            # fixed sequence motif during sampling
            if config.eval.fixed_part:
                if fixed_seq_str is None:
                    fixed_seq_str = _get_fixed_seq_for_pdb(
                        config,
                        pdb_path,
                        expected_len=struct_data['seq'].shape[0],
                    )
                fix_seq_str = fixed_seq_str

                fix_indices = [i for i, char in enumerate(fix_seq_str) if char != '-']

                fix_seq_num = du.seq_to_num(fix_seq_str.replace('-', 'A'))
                fix_node_s = (fix_seq_num.unsqueeze(-1) == torch.arange(4).unsqueeze(0)).float()
                fix_mask = torch.zeros_like(fix_seq_num).bool()
                for index in fix_indices:
                    fix_mask[index] = True

                struct_data['fixed_mask'] = fix_mask
                # update struct_data['seq'] based on fix_seq_num and fix_mask
                struct_data['seq'] = torch.where(fix_mask, fix_seq_num, struct_data['seq'])
                struct_data['node_s'] = torch.where(fix_mask.unsqueeze(-1), fix_node_s, struct_data['node_s'])

            struct_data = tree.map_structure(
                lambda x: x.unsqueeze(0).repeat_interleave(curr_n_sample, dim=0).to(config.device), struct_data)

            samples = test_sampling_single_fn(model, struct_data, fixed_part=config.eval.fixed_part)

            # save to fasta dir
            for j in range(len(samples)):
                curr_id = int(np.array(curr_n_samples[:i]).sum()) + j
                curr_pdb_id = pdb_id + '_' + str(curr_id)
                # du.sample_to_fasta(samples[j], pdb_id,
                #                    os.path.join(fasta_dir, curr_pdb_id + '.fasta'))
                fasta_file.write(du.sample_to_str(samples[j], curr_pdb_id))

    fasta_file.close()
