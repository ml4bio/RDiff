import torch
import os

import numpy as np

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

    # return ssmap [1, N, N], where N is the length of db_input
    # if considering non-canonical pairs and stacking, can modify here to change the 1 channel to 3 channels
    length = len(db_input)
    ssmap = np.zeros((1, length, length), dtype=np.float32)

    for i, j in all_pairs:
        ssmap[0, i, j] = 1.0

    return ssmap

def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        print(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=True)  # change strict to False?
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def get_data_scaler(config):
    """Data normalizer"""
    # not consider bias here
    centered = config.data.seq_centered

    def scale_fn(seq):
        if centered:
            seq = seq * 2. - 1.
        return seq

    return scale_fn


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""

    centered = config.data.seq_centered

    def inverse_scale_fn(seq):
        if centered:
            seq = (seq + 1.) / 2.
        return seq

    return inverse_scale_fn