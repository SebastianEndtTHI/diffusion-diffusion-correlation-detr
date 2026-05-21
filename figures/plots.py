import numpy as np
import matplotlib.pyplot as plt

import argparse
import pickle


def main(args):
    model_label = args.model_label

    with open("../models/" + model_label + "/detr_logs.pkl", 'rb') as f:
        logs = pickle.load(f)

    with open("../models/" + model_label + "/detr_output.pkl", 'rb') as f:
        out = pickle.load(f)

    pred = out['predictions']  # shape: (N, n_queries, 7)
    ref = out['ground_truths']
    ref = ref.reshape([ref.shape[0], ref.shape[1] // 6, 6])  # shape: (N, n_comp, 6)
    test_losses = np.array(out['losses'])
    n_comp = out['n_comp']

    pred[:, :, -1] = sigmoid(pred[:, :, -1])

    plot_diff_spectra(pred, ref, n_comp)

    true_mds, pred_mds, fractions_md = collect_matched_params(pred, ref, n_comp, param_idx=0)
    true_fas, pred_fas, fractions_fa = collect_matched_params(pred, ref, n_comp, param_idx=1)
    fractions_ang, ang_errors = collect_matched_dirs(pred, ref, n_comp)

    plot_eval_figure(true_mds, pred_mds, fractions_md,
                     true_fas, pred_fas, fractions_fa,
                     fractions_ang, ang_errors)

    return


def plot_diff_spectra(pred, ref, n_comp):
    target_n_comps = np.unique(n_comp)
    plt_indices = []
    for target in target_n_comps:
        try:
            idx = next(i for i, n in enumerate(n_comp) if n == target)
            plt_indices.append(idx)
        except StopIteration:
            raise ValueError(f"n_comp={target} not found in data set")

    n_cols = len(target_n_comps)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8),
                             constrained_layout=True)

    for i, pltidx in enumerate(plt_indices):
        # Prediction
        ax_pred = axes[0, i]
        mask = pred[pltidx, :, -1] > 0.5
        x = pred[pltidx, mask, 0]
        y = pred[pltidx, mask, 1]
        color = pred[pltidx, mask, 1, None] * np.absolute(pred[pltidx, mask, 2:5])
        size = pred[pltidx, mask, 5] * 200
        ax_pred.scatter(x, y, c=color, s=size, alpha=1)
        ax_pred.set_xlim([0, 2])
        ax_pred.set_ylim([0, 1])
        ax_pred.set_xlabel(r"MD [$10^{-3}$ mm$^2$/s]", fontsize=11)
        ax_pred.set_ylabel("FA", fontsize=11)
        ax_pred.tick_params(labelsize=9)
        ax_pred.grid(True, linestyle='--', alpha=0.3)
        ax_pred.set_box_aspect(1)
        ax_pred.set_title(r"$n_{c}=$" + f"{n_comp[pltidx]}", fontsize=12)
        if i == 0:
            ax_pred.text(-0.25, 0.5, "Predicted", transform=ax_pred.transAxes,
                         fontsize=11, va='center', ha='center', rotation=90)

        # Ground Truth
        ax_ref = axes[1, i]
        x_ref = ref[pltidx, :, 0]
        y_ref = ref[pltidx, :, 1]
        color_ref = ref[pltidx, :, 1, None] * np.absolute(ref[pltidx, :, 2:5])
        size_ref = ref[pltidx, :, 5] * 200
        ax_ref.scatter(x_ref, y_ref, c=color_ref, s=size_ref, alpha=1)
        ax_ref.set_xlim([0, 2])
        ax_ref.set_ylim([0, 1])
        ax_ref.set_xlabel(r"MD [$10^{-3}$ mm$^2$/s]", fontsize=11)
        ax_ref.set_ylabel("FA", fontsize=11)
        ax_ref.tick_params(labelsize=9)
        ax_ref.grid(True, linestyle='--', alpha=0.3)
        ax_ref.set_box_aspect(1)
        if i == 0:
            ax_ref.text(-0.25, 0.5, "Ground Truth", transform=ax_ref.transAxes,
                        fontsize=11, va='center', ha='center', rotation=90)

    plt.show()


def collect_matched_dirs(preds, gts, n_comps):
    # For every sample, extract (signal_fraction, angular_error_deg) for each valid matched compartment
    n_samples, _, n_feats = preds.shape

    fractions, ang_errors, comp_ids = [], [], []

    for i in range(n_samples):
        nc = n_comps[i]
        for j in range(nc):
            # signal fraction from ground truth
            frac = gts[i, j, 5]

            # direction vectors
            pred_dir = preds[i, j, 2:5]
            true_dir = gts[i, j, 2:5]

            # normalise
            pred_norm = np.linalg.norm(pred_dir)
            true_norm = np.linalg.norm(true_dir)
            if pred_norm < 1e-8 or true_norm < 1e-8:
                continue
            pred_dir = pred_dir / pred_norm
            true_dir = true_dir / true_norm

            # unsigned angle: clamp to [0, 90°] because v ≡ -v for fiber dirs
            cos_sim = np.clip(np.dot(pred_dir, true_dir), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(abs(cos_sim)))

            fractions.append(frac)
            ang_errors.append(angle_deg)
            comp_ids.append(j)

    return np.array(fractions), np.array(ang_errors)


def collect_matched_params(preds, gts, n_comps, param_idx):
    # extract (true_params, pred_params) for each valid matched compartment

    n_samples, _, _ = preds.shape
    true_params, pred_params, fractions = [], [], []

    for i in range(n_samples):
        nc = n_comps[i]
        for j in range(nc):
            true_params.append(gts[i, j, param_idx])
            pred_params.append(preds[i, j, param_idx])
            fractions.append(gts[i, j, 5])

    return np.array(true_params), np.array(pred_params), np.array(fractions)


def plot_eval_figure(true_mds, pred_mds, fractions_md,
                     true_fas, pred_fas, fractions_fa,
                     fractions_ang, ang_errors):
    """
    Combined evaluation figure: MD scatter | FA scatter | Angular error.
    Uses GridSpec with a dedicated colorbar column to avoid tight_layout conflicts.
    All three subplots share the plasma signal-fraction colormap and a single colorbar.
    """
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('plasma')
    scatter_kw = dict(cmap=cmap, norm=norm, s=8, alpha=0.5, rasterized=True)

    fig = plt.figure(figsize=(14.5, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.35)

    ax_md = fig.add_subplot(gs[0])
    ax_fa = fig.add_subplot(gs[1])
    ax_ang = fig.add_subplot(gs[2])
    ax_cb = fig.add_subplot(gs[3])

    # ── MD scatter ────────────────────────────────────────────────────────────
    ax_md.scatter(true_mds, pred_mds, c=fractions_md, **scatter_kw)
    lim_max = np.ceil(max(true_mds.max(), pred_mds.max()) * 10.) / 10.
    ax_md.plot([0, lim_max], [0, lim_max], 'k--', linewidth=1.0, label='Identity')
    ax_md.text(0.05, 0.89, f'$R^2 = {_r2(true_mds, pred_mds):.4f}$',
               transform=ax_md.transAxes, fontsize=10, va='top')
    ax_md.set_xlim(0, lim_max)
    ax_md.set_ylim(0, lim_max)
    ax_md.set_aspect('equal')
    ax_md.set_box_aspect(1)
    ax_md.set_xlabel(r'True MD [$10^{-3}$ mm$^2$/s]', fontsize=11)
    ax_md.set_ylabel(r'Predicted MD [$10^{-3}$ mm$^2$/s]', fontsize=11)
    ax_md.set_title('Mean Diffusivity', fontsize=12)
    ax_md.tick_params(labelsize=9)
    ax_md.grid(True, linestyle='--', alpha=0.3)
    ax_md.legend(fontsize=9)

    # ── FA scatter ────────────────────────────────────────────────────────────
    ax_fa.scatter(true_fas, pred_fas, c=fractions_fa, **scatter_kw)
    ax_fa.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Identity')
    ax_fa.text(0.05, 0.89, f'$R^2 = {_r2(true_fas, pred_fas):.4f}$',
               transform=ax_fa.transAxes, fontsize=10, va='top')
    ax_fa.set_xlim(0, 1)
    ax_fa.set_ylim(0, 1)
    ax_fa.set_aspect('equal')
    ax_fa.set_box_aspect(1)
    ax_fa.set_xlabel('True FA', fontsize=11)
    ax_fa.set_ylabel('Predicted FA', fontsize=11)
    ax_fa.set_title('Fractional Anisotropy', fontsize=12)
    ax_fa.tick_params(labelsize=9)
    ax_fa.grid(True, linestyle='--', alpha=0.3)
    ax_fa.legend(fontsize=9)

    # ── Angular error ─────────────────────────────────────────────────────────
    ax_ang.scatter(fractions_ang, ang_errors, c=fractions_ang, **scatter_kw)
    bins = np.linspace(0, 1, 20)
    bin_centers, bin_medians = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (fractions_ang >= lo) & (fractions_ang < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            bin_medians.append(np.median(ang_errors[mask]))
    ax_ang.plot(bin_centers, bin_medians, 'k-', linewidth=1.5,
                label='Median (binned)', zorder=5)
    mae = np.mean(ang_errors)
    med = np.median(ang_errors)
    print(f'Angular error — Mean: {mae:.2f}°  Median: {med:.2f}°')
    ax_ang.set_xlim(0, 1)
    ax_ang.set_ylim(bottom=0)
    ax_ang.set_box_aspect(1)
    ax_ang.set_xlabel('Signal Fraction (ground truth)', fontsize=11)
    ax_ang.set_ylabel('Angular Error (°)', fontsize=11)
    ax_ang.set_title('Main Direction', fontsize=12)
    ax_ang.tick_params(labelsize=9)
    ax_ang.grid(True, linestyle='--', alpha=0.3)
    ax_ang.legend(fontsize=9)

    # ── colorbar anchored to the rightmost axes box ───────────────────────────
    # Draw first so get_position() returns the final rendered position
    fig.canvas.draw()
    pos = ax_ang.get_position()          # Bbox in figure-fraction coordinates
    cax = fig.add_axes([pos.x1 + 0.015,  # left edge: just right of the axes
                        pos.y0,           # bottom: aligned with axes
                        0.012,            # width
                        pos.height])      # height: exactly matches axes
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    cax.set_ylabel('Signal Fraction (ground truth)', fontsize=10)
    cax.tick_params(labelsize=9)

    plt.show()


def _r2(true, pred):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR plots')
    parser.add_argument('--model_label', type=str, default="N250k_[1,2,3,4,5]_maxeig4_noise0,01_seed0",
                        help='model label for loading logs and output')
    parserargs = parser.parse_args()

    # run training
    main(parserargs)
