import dtigenerator as gen
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Set dataset generator', add_help=False)

    # dataset
    parser.add_argument('--n_samples', default=100000, type=int, help="number of samples to generate.")
    parser.add_argument('--n_comp_list', nargs="+", default=[1, 2, 3, 4], type=int,
                        help="list with possible numbers of compartments in a sample.")

    # diffusion protocol
    parser.add_argument('--diff_file', default="diff_dirs_b2000.txt", type=str, help="path to diffusion protocol file.")

    # compartments
    parser.add_argument('--ev_dist', default="uniform", type=str,
                        help="set eigenvalue distribution to 'uniform' for uniform or 'gm' realistic distribution.")
    parser.add_argument('--dir_type', default="con", type=str,
                        help="set direction information to 'con' for continuous or 'disc' for discrete representation.")
    parser.add_argument('--n_dir', default=30, type=int, help="number of discrete directions.")

    # save path
    parser.add_argument('--save_file_name', default="multi_compartment_dataset.csv", type=str)

    return parser


def main(args):
    comp_list = args.n_comp_list

    # loop generating samples
    dataset = []
    for n_comp in tqdm(comp_list):
        for i in tqdm(range(args.n_samples)):

            # generator initialization
            generator = gen.DTIGenerator(args)

            # generate signal and ground truth with n_comp compartments
            S, md, fa, dir, w = generator.generate_Voxel(n_comp)

            # building vector with voxel signal and ground truth for each compartment
            sample = list(S)

            for c in range(n_comp):

                # MD and FA from all compartments in one voxel
                sample.append(md[c])
                sample.append(fa[c])

                # continuous direction as vector with x-, y-, and z-coords
                if args.dir_type == "con":

                    for coord in dir[c]:
                        sample.append(coord)

                # discrete direction as class index
                elif args.dir_type == "disc":

                    sample.append(dir[c])

                # weights from all compartments in one voxel
                sample.append(w[c])

            # padding samples with nan for even size of max_n_comp compartments
            if args.dir_type == "con":
                sample.extend([np.nan] * 6 * (np.max(comp_list) - n_comp))

            elif args.dir_type == "disc":
                sample.extend([np.nan] * 4 * (np.max(comp_list) - n_comp))

            # save sample including mri signal (S), repeating MD (md), FA (fa), direction (dir), and weights (w) for all compartments, and number of compartments (n_comp)
            sample.append(n_comp)

            dataset.append(sample)

    # signal values defined in columns named 'mri_sigX' with X between 0 and (signal size - 1)
    signal_label = ["mri_sig" + str(s) for s in range(len(S))]

    # compartment metric columns
    if args.dir_type == "con":
        base_labels = ["MD", "FA", "dir_X", "dir_Y", "dir_Z", "weight"]

    elif args.dir_type == "disc":
        base_labels = ["MD", "FA", "direction", "weight"]

    # metric columns named for example 'MD_comp1' describing every metric all compartments per sample
    comp_label = [f"{label}_comp{i}" for i in range(1, np.max(comp_list) + 1) for label in base_labels]

    # last column describes number of compartments per sample
    comp_label.append("n_comp")

    # save dataset in csv file
    column_label = signal_label + comp_label

    df = pd.DataFrame(dataset, columns=column_label)
    df.to_csv(os.path.normpath(os.path.join(os.path.dirname(__file__), '../data', args.save_file_name)),
              index_label='idx')


# code initialization
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # run data generation
    main(args)
