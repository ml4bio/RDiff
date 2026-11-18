import numpy as np


def rigid_transform_3D(A, B, verbose=False):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected

def get_aligned_rmsd_from_pdb(casp_path1, pred_path2, res_id_offset=0):

    # loop over casp_path1
    pdb1_info = {}
    for line in open(casp_path1, 'r'):
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            resd_name = line[17:20].strip()
            resd_idx = int(line[22:26]) + res_id_offset
            pos = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            key = (atom_name, resd_name, resd_idx)
            pdb1_info[key] = pos

    # loop over pred_path2
    pdb2_info = {}
    for line in open(pred_path2, 'r'):
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            resd_name = line[17:20].strip()
            resd_idx = int(line[22:26])
            pos = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            key = (atom_name, resd_name, resd_idx)
            pdb2_info[key] = pos

    # get same atom pos array from pdb1 and pdb2
    pos_1 = []
    pos_2 = []
    for key in pdb1_info:
        if key in pdb2_info:
            pos_1.append(pdb1_info[key])
            pos_2.append(pdb2_info[key])
    pos_1 = np.array(pos_1)
    pos_2 = np.array(pos_2)

    pdb.set_trace()
    aligned_rmsd = calc_aligned_rmsd(pos_1, pos_2)
    return aligned_rmsd

