import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F


class HungarianLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        # loss component weights
        self.md_lmd = args.md_loss_weight
        self.fa_lmd = args.fa_loss_weight
        self.dir_lmd = args.dir_loss_weight
        self.wt_lmd = args.wt_loss_weight
        self.exs_lmd = args.exs_loss_weight

    def forward(self, y_pred, y_true, n_comps):
        """
        DETR-based loss function using Hungarian matching for computing the
        costs of MD, FA, direction vectors, compartment weights, and the 
        query existence score.

        :param y_pred:  Prediction tensor of shape (B, n_Q, 7). For each query,
                        the model predicts, in this exact order:
                        MD, FA, direction coordinates (x, y, z),
                        compartment weight, and existence score.

        :param y_true:  Target tensor of shape (B, n_compartments * 7). For each
                        compartment, the target values consist of MD, FA,
                        direction coordinates (x, y, z), compartment weight,
                        and existence score—in this exact order. All 
                        compartments of a sample are flattened into a single
                        vector. Samples with fewer compartments than the
                        dataset maximum may be padded; non-existent
                        compartments are filtered out using `n_comps`.

        :param n_comps: 1D tensor specifying the number of valid target
                        compartments per sample.
        """

        # loss initialization
        device = y_pred.device
        batch_size, columns = y_true.shape
        n_pred = y_pred.shape[-1] - 1
        y_true = y_true.reshape(batch_size, int(columns / n_pred), n_pred)

        # cost computation of MD, FA, and weights withs mean squared error
        md_loss = (y_pred[:, :, 0].unsqueeze(2) - y_true[:, :, 0].unsqueeze(1)) ** 2
        fa_loss = (y_pred[:, :, 1].unsqueeze(2) - y_true[:, :, 1].unsqueeze(1)) ** 2
        wt_loss = (y_pred[:, :, 5].unsqueeze(2) - y_true[:, :, 5].unsqueeze(1)) ** 2

        # direction costs computed with cosine similarity and weighted with the target's FA
        dir_sim = F.cosine_similarity(y_pred[:, :, 2:5].unsqueeze(2),
                                      y_true[:, :, 2:5].unsqueeze(1),
                                      dim=-1)

        dir_loss = (1 - dir_sim) * y_true[:, :, 1].unsqueeze(1)

        # cost matrix for all prediction-target combinations, with MD, FA, and direction costs
        cost_matrix_m = (self.md_lmd * md_loss +
                         self.fa_lmd * fa_loss +
                         self.dir_lmd * dir_loss)

        # prediction-target combinations weighted on target weight
        cost_matrix_weighted = cost_matrix_m * y_true[:, :, 5].unsqueeze(1)

        # adding weight costs to weighted MD, Fa, direction costs
        cost_matrix = cost_matrix_weighted + self.wt_lmd * wt_loss

        # normalized existence score costs
        extnc_scores = torch.sigmoid(y_pred[:, :, 6])
        extnc_cost = (1 - extnc_scores).unsqueeze(2).expand(-1, -1, y_true.size(1))

        # final cost matrix for matching
        cost_matrix = cost_matrix + self.exs_lmd * extnc_cost

        # cpu transfer
        cost_np = cost_matrix.detach().cpu().numpy()
        n_queries = y_pred.size(1)

        # Hungarian matching indexes
        all_pred_idx = []
        all_true_idx = []
        all_batch_idx = []

        for b in range(batch_size):
            # filtering paddings by the number of compartments
            n_comp = n_comps[b].item() if torch.is_tensor(n_comps[b]) else n_comps[b]
            real_costs = cost_np[b, :, :n_comp]

            # matching optimal prediction-target assignments
            row_idx, col_idx = linear_sum_assignment(real_costs)

            all_pred_idx.extend(row_idx)
            all_true_idx.extend(col_idx)
            all_batch_idx.extend([b] * len(row_idx))

        # loading indices on gpu if defined
        pred_indices = torch.tensor(all_pred_idx, device=device, dtype=torch.long)
        true_indices = torch.tensor(all_true_idx, device=device, dtype=torch.long)
        batch_indices = torch.tensor(all_batch_idx, device=device, dtype=torch.long)

        # metric's costs of the computed matches
        matched_md = md_loss[batch_indices, pred_indices, true_indices]
        matched_fa = fa_loss[batch_indices, pred_indices, true_indices]
        matched_dir = dir_loss[batch_indices, pred_indices, true_indices]
        matched_wt = wt_loss[batch_indices, pred_indices, true_indices]
        batch_weights = y_true[batch_indices, true_indices, 5]

        # weighting costs with target weights
        weighted_md = matched_md * batch_weights * self.md_lmd
        weighted_fa = matched_fa * batch_weights * self.fa_lmd
        weighted_dir = matched_dir * batch_weights * self.dir_lmd
        weighted_wt = matched_wt * self.wt_lmd

        # creating existence score labels, with 1 for matched predictions and 0 for the rest
        extnc = torch.zeros(batch_size, n_queries, device=device)
        extnc[batch_indices, pred_indices] = 1

        # positives weighted up for class balancing
        p_weight = (n_queries - extnc.sum(dim=-1)) / extnc.sum(dim=-1)
        p_weight = torch.clamp(p_weight, min=0.5, max=10.0)

        # binary cross entropy as existence score loss
        weighted_bce = nn.BCEWithLogitsLoss(pos_weight=p_weight.unsqueeze(1))
        cls_loss = weighted_bce(y_pred[:, :, 6], extnc)

        # computing average metric costs per sample
        batch_md_losses = torch.zeros(batch_size, device=device)
        batch_fa_losses = torch.zeros(batch_size, device=device)
        batch_dir_losses = torch.zeros(batch_size, device=device)
        batch_wt_losses = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            b_mask = batch_indices == b

            if b_mask.any():
                batch_md_losses[b] = weighted_md[b_mask].mean()
                batch_fa_losses[b] = weighted_fa[b_mask].mean()
                batch_dir_losses[b] = weighted_dir[b_mask].mean()
                batch_wt_losses[b] = weighted_wt[b_mask].mean()

        # final loss sums up metrics' costs of found matches and existence score loss per sample
        batch_losses = (batch_md_losses + batch_fa_losses +
                        batch_dir_losses + batch_wt_losses +
                        self.exs_lmd * cls_loss)

        # return average losses of batch
        loss = batch_losses.mean()
        md_mean = batch_md_losses.mean()
        fa_mean = batch_fa_losses.mean()
        dir_mean = batch_dir_losses.mean()
        wt_mean = batch_wt_losses.mean()
        extnc_mean = self.exs_lmd * cls_loss

        return loss, md_mean, fa_mean, dir_mean, wt_mean, extnc_mean

    def get_matching_indices(self, y_pred, y_true, n_comps):
        """
        Run Hungarian matching and return per-sample (pred_idx, true_idx) pairs,
        identical to the matching logic in forward() but without computing losses.

        Returns:
            List of (row_idx, col_idx) numpy arrays, one tuple per sample in the batch.
            row_idx[i] is the query index matched to ground-truth compartment col_idx[i].
        """
        device = y_pred.device
        batch_size, columns = y_true.shape
        n_pred = y_pred.shape[-1] - 1
        y_true_r = y_true.reshape(batch_size, int(columns / n_pred), n_pred)

        md_loss = (y_pred[:, :, 0].unsqueeze(2) - y_true_r[:, :, 0].unsqueeze(1)) ** 2
        fa_loss = (y_pred[:, :, 1].unsqueeze(2) - y_true_r[:, :, 1].unsqueeze(1)) ** 2
        wt_loss = (y_pred[:, :, 5].unsqueeze(2) - y_true_r[:, :, 5].unsqueeze(1)) ** 2

        dir_sim = torch.nn.functional.cosine_similarity(
            y_pred[:, :, 2:5].unsqueeze(2),
            y_true_r[:, :, 2:5].unsqueeze(1),
            dim=-1)
        dir_loss = (1 - dir_sim) * y_true_r[:, :, 1].unsqueeze(1)

        cost_matrix_m = (self.md_lmd * md_loss +
                         self.fa_lmd * fa_loss +
                         self.dir_lmd * dir_loss)
        cost_matrix_weighted = cost_matrix_m * y_true_r[:, :, 5].unsqueeze(1)
        cost_matrix = cost_matrix_weighted + self.wt_lmd * wt_loss

        extnc_scores = torch.sigmoid(y_pred[:, :, 6])
        extnc_cost = (1 - extnc_scores).unsqueeze(2).expand(-1, -1, y_true_r.size(1))
        cost_matrix = cost_matrix + self.exs_lmd * extnc_cost

        cost_np = cost_matrix.detach().cpu().numpy()

        indices = []
        for b in range(batch_size):
            n_comp = n_comps[b].item() if torch.is_tensor(n_comps[b]) else int(n_comps[b])
            real_costs = cost_np[b, :, :n_comp]
            row_idx, col_idx = linear_sum_assignment(real_costs)
            indices.append((row_idx, col_idx))

        return indices
