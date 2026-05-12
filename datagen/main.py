import pandas as pd
import numpy as np
from tqdm import tqdm

import argparse
import os
import datetime

import dtigenerator as gen


def get_args_parser():
    parser = argparse.ArgumentParser('Set dataset generator', add_help=False)

    # dataset
    parser.add_argument('--n_samples', default=100, type=int, help="number of samples to generate.")
    parser.add_argument('--n_comp_list', nargs="+", default=[1, 2, 3, 4, 5], type=int,
                        help="list with possible numbers of compartments in a sample.")
    parser.add_argument('--split', nargs="+", default=[0.8, 0.2], type=float, help="train/test split ratio.")
    parser.add_argument('--noiselvl', default=0.01, type=float,
                        help="noise level, e.g. 0.01 for 1% Gaussian noise relative to the median of the signal curve.")
    parser.add_argument('--seed', default=0, type=int, help="random seed for reproducibility.")

    # diffusion protocol
    parser.add_argument('--diff_file', default="diff_dirs_b2000.txt", type=str, help="path to diffusion protocol file.")

    return parser


def main(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    comp_list = args.n_comp_list

    rng = np.random.default_rng(args.seed)

    if len(args.split) != 2:
        raise ValueError("Split argument must contain exactly two values for train and test split.")
    datasplit = args.split / np.sum(args.split)
    split_labels = ["train", "test"]

    for split_idx, split in enumerate(datasplit):
        split_samples = int(np.round(args.n_samples * split))

        # loop generating samples
        dataset = []
        for n_comp in tqdm(comp_list):
            for i in tqdm(range(int(np.round(split_samples/len(comp_list))))):

                # generator initialization
                generator = gen.DTIGenerator(args, rng)

                # generate signal and ground truth with n_comp compartments
                S, md, fa, dir, w = generator.generate_voxel(n_comp, args.noiselvl)

                # building vector with voxel signal and ground truth for each compartment
                sample = list(S)

                for c in range(n_comp):

                    # MD and FA from all compartments in one voxel
                    sample.append(md[c])
                    sample.append(fa[c])

                    # continuous direction as vector with x-, y-, and z-coords
                    for coord in dir[c]:
                        sample.append(coord)

                    # weights from all compartments in one voxel
                    sample.append(w[c])

                # padding samples with nan for even size of max_n_comp compartments
                sample.extend([np.nan] * 6 * (np.max(comp_list) - n_comp))

                # save sample including mri signal (S), repeating MD (md), FA (fa), direction (dir), and weights (w) for all compartments, and number of compartments (n_comp)
                sample.append(n_comp)

                dataset.append(sample)

        # signal values defined in columns named 'mri_sigX' with X between 0 and (signal size - 1)
        signal_label = ["mri_sig" + str(s) for s in range(len(S))]

        # compartment metric columns
        base_labels = ["MD", "FA", "dir_X", "dir_Y", "dir_Z", "weight"]

        # metric columns named for example 'MD_comp1' describing every metric all compartments per sample
        comp_label = [f"{label}_comp{i}" for i in range(1, np.max(comp_list) + 1) for label in base_labels]

        # last column describes number of compartments per sample
        comp_label.append("n_comp")

        # save dataset in csv file
        column_label = signal_label + comp_label

        df = pd.DataFrame(dataset, columns=column_label)
        df.to_csv(os.path.normpath(os.path.join(os.path.dirname(__file__), '../data', timestamp +
                                                '_N' + str(args.n_samples) + '_ncomp' + str(args.n_comp_list) +
                                                '_noise' + str(args.noiselvl) + '_' + split_labels[split_idx] + '.csv')),
                  index_label='idx')


# code initialization
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # run data generation
    main(args)
