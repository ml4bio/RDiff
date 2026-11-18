"""RNA data type."""

import dataclasses
import numpy as np

@dataclasses.dataclass(frozen=True)
class RNA:
    """RNA structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # rna_constants.atom_types, i.e. the first three are C4', C1', N1'/N9'. (max: 23)
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Residue type represented as an integer between 0 and 3, where 0 is 'A', 1 is 'G', 2 is 'U', 3 is 'C'.
    resd_type: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom is present and 0.0 if not.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB.
    resd_index: np.ndarray  # [num_res]
