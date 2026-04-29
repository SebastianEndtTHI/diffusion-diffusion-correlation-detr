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


def get_multiKompSignal(comp_signals):

    comp_count = np.size(comp_signals, axis=0)

    # generate random weights in uniform distribution
    weight_borders = np.random.uniform(0, 1, comp_count - 1)
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


def get_rotation_matrix(r):

    # norm direction on unit sphere
    r /= np.linalg.norm(r)

    # generate random vector, non-parallel to r
    rand_vec = np.random.randn(3)

    while np.allclose(np.cross(r, rand_vec), 0):
        rand_vec = np.random.randn(3)

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
    def __init__(self, args):
        super().__init__()

        # load diffusion protocol
        self.direction_table = np.loadtxt(os.path.normpath(os.path.join(os.path.dirname(__file__), '../datagen', args.diff_file)), delimiter=",")

        # compartment parameters
        self.ev_dist = args.ev_dist
        self.dir_type = args.dir_type
        self.n_directions = args.n_dir

        # lambda 1 parameters in gm distribution
        self.means_l1 = [1.0, 0.9, 1.4, 1.9]  # WM, GM, CSF1, CSF2
        self.stds_l1 = [0.2, 0.15, 0.15, 0.5]
        self.weights_l1 = [0.35, 0.35, 0.15, 0.15]

        # lambda 2 and 3 parameters in gm distribution
        self.means_l2 = [0.5, 0.8, 1.2, 1.8]  # WM, GM, CSF1, CSF2
        self.stds_l2 = [0.15, 0.15, 0.15, 0.5]
        self.weights_l2 = [0.35, 0.35, 0.15, 0.15]

        # definition of discrete directions on fibonacci sphere
        directions = fibonacci_sphere(self.n_directions * 2)

        # filter for x-coords >= 0 for direction on the same half of unit sphere
        self.directions = directions[np.where(directions[:, 0] >= 0)]

    def get_eigenvalue(self, means, stds, weights, size=1):

        if self.ev_dist == "gm":

            # random eigenvalues from realistic gm distribution
            component = np.random.choice([0, 1, 2, 3], size=size, p=weights).flatten()[0]
            sample = np.random.normal(loc=means[component], scale=stds[component])

        elif self.ev_dist == "uniform":

            # random eigenvalues between 0 and 3.5 from uniform distribution
            sample = np.random.random() * 3.5

        return sample

    def generate_Voxel(self, n_comp: int):

        # generate voxel signal and compartment metrics 
        dir_list = []
        D_list = []
        signal_list = []
        md_list = []
        fa_list = []

        # for each compartment in one voxel
        for c in range(n_comp):

            # generate three random eigenvalues from define distribution
            lmd1 = self.get_eigenvalue(self.means_l1, self.stds_l1, self.weights_l1)
            lmd2 = self.get_eigenvalue(self.means_l2, self.stds_l2, self.weights_l2)
            lmd3 = self.get_eigenvalue(self.means_l2, self.stds_l2, self.weights_l2)

            # define diagonal eigenvalue matrix
            eig_matrix = get_eigenvalue_matrix(lmd1, lmd2, lmd3)

            # generate continuous directions
            if self.dir_type == "con":

                # random vector with x-coord >= 0
                direction = np.random.randn(3)

                while direction[0] < 0:
                    direction = np.random.randn(3)

                # define orientation matrix Q
                Q = get_rotation_matrix(direction)

                # save normed directions for ground truth
                direction /= np.linalg.norm(direction)
                dir_list.append(direction)

            # generate continuous directions
            elif self.dir_type == "disc":

                # choose random index to define direction for fibonacci sphere
                dir_idx = np.random.choice(range(self.directions.shape[0] - 1))

                # define orientation matrix Q
                Q = get_rotation_matrix(self.directions[dir_idx])

                # save direction index for ground truth
                dir_list.append(dir_idx + 1)

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
        signals, weight_list = get_multiKompSignal(signal_list)

        # return voxel signal and metrics of each compartment
        return signals, md_list, fa_list, dir_list, weight_list
