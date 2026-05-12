import os
import numpy as np
from dipy.reconst.dti import fractional_anisotropy as FA


def get_eigenvalue_matrix(ev1, ev2, ev3):

    # sort eigenvalues
    lmds = np.sort([ev1, ev2, ev3])

    # matrix with diagonal eigenvalues, normed to realistic value range
    lmd_mtrx = np.diag(np.array([lmds[2], lmds[1], lmds[0]])) * 1e-3

    return lmd_mtrx


def get_diffusionTensor(eigenvalues, Q):

    # computing diffusion tensor with eigenvalues and rotation matrix for direction
    D = Q @ eigenvalues @ Q.T

    return D


def get_multiKompSignal(comp_signals, rng):

    comp_count = np.size(comp_signals, axis=0)

    # generate random weights in uniform distribution
    weight_borders = rng.uniform(low=0, high=1, size=comp_count-1)
    weight_areas = np.sort(np.append(weight_borders, [0, 1]))

    # random weighting for every compartment signal
    weights = []
    weighted_signals = []

    for t_idx in range(comp_count):
        # size of weight areas define the actual weight
        weight = weight_areas[t_idx + 1] - weight_areas[t_idx]
        weights.append(weight)

        # weighted signal for every single compartment
        weighted_comp = np.multiply(weight, comp_signals[t_idx])
        weighted_signals.append(weighted_comp)

    # summed up weighted compartment signals build a voxel signal
    voxel_signals = np.sum(weighted_signals, axis=0)

    return voxel_signals, weights


def fibonacci_sphere(n):

    points = []

    # The golden angle (in radians), used to achieve uniform spacing
    phi = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n):
        # Map index i to y-coordinate in [-1, 1]
        y = 1 - (i / float(n - 1)) * 2.0

        # Radius of the circle at height y on the unit sphere
        radius = np.sqrt(1.0 - y * y)

        # Angular position around the circle
        theta = phi * i

        # Convert spherical coordinates to Cartesian coordinates
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)


def get_rotation_matrix(r, rng):

    # generate random vector, non-parallel to r
    rand_vec = rng.normal(size=3)

    while np.allclose(np.cross(r, rand_vec), 0):
        rand_vec = rng.normal(size=3)

    # orthogonal vectors returned by cross product, normed on unit sphere
    v2 = np.cross(r, rand_vec)
    v2 /= np.linalg.norm(v2)

    v3 = np.cross(r, v2)
    v3 /= np.linalg.norm(v3)

    # orthogonal vectors build orientation matrix Q
    Q = np.column_stack((r, v2, v3))

    return Q


def compute_signal_and_spectrogram(D, b_value, r):

    # simulating diffusion signals with generated diffusion tensor and directions form a diffusion protocoll
    rDr = r.T @ D @ r

    signals = np.exp(-b_value * rDr)

    return signals


class DTIGenerator:
    def __init__(self, args, rng):
        super().__init__()

        self.rng = rng

        # load diffusion protocol
        self.direction_table = np.loadtxt(os.path.normpath(os.path.join(os.path.dirname(__file__), '../datagen', args.diff_file)), delimiter=",")

    def get_eigenvalues(self):
        # random eigenvalues between 0 and 3.5 from uniform distribution
        sample = self.rng.uniform(low=0, high=3.5, size=3)

        return sample

    def generate_voxel(self, n_comp: int, noiselvl: float):

        # generate voxel signal and compartment metrics 
        dir_list = []
        D_list = []
        signal_list = []
        md_list = []
        fa_list = []

        # for each compartment in one voxel
        for c in range(n_comp):

            # generate three random eigenvalues from define distribution
            lmd1, lmd2, lmd3 = self.get_eigenvalues()

            # define diagonal eigenvalue matrix
            eig_matrix = get_eigenvalue_matrix(lmd1, lmd2, lmd3)

            # generate continuous directions
            # random vector with x-coord >= 0
            direction = self.rng.normal(size=3)

            if direction[0] < 0:
                direction *= -1

            # define orientation matrix Q
            direction /= np.linalg.norm(direction)
            Q = get_rotation_matrix(direction, self.rng)

            # save normed directions for ground truth
            dir_list.append(direction)

            # compute diffusion tensor
            D_comp = get_diffusionTensor(eig_matrix, Q)
            D_list.append(D_comp)

            # signal simulation
            comp_signals = []

            for b in self.direction_table:
                # simulate signal for every diffusion protocol entry
                b_signal = compute_signal_and_spectrogram(D_comp, b[0], b[1:])
                comp_signals.append(b_signal)

            # save compartment signals
            signal_list.append(comp_signals)

            # calculate mean diffusivity of compartment (normed by factor 3 for a range near to [0,1])
            md = np.mean([lmd1, lmd2, lmd3]) / 3
            md_list.append(md)

            # calculate fractional anisotropy of compartment
            fa = FA(np.array([lmd1, lmd2, lmd3]))
            fa_list.append(fa)

        # sum of random weighted compartment signals build the complete voxel signal
        signals, weight_list = get_multiKompSignal(signal_list, self.rng)

        # add Gaussian noise
        signals += self.rng.normal(loc=0, scale=noiselvl*np.median(signals), size=len(signals))

        # return voxel signal and metrics of each compartment
        return signals, md_list, fa_list, dir_list, weight_list
