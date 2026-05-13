import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd


class TrainUtils:
    def __init__(self, model, args, optimizer=None, criterion=None):
        super().__init__()

        # setup initialization
        self.model = model
        self.opt = optimizer
        self.crit = criterion
        self.device = args.device

        # auxiliary loss 
        self.aux_loss = args.aux_loss
        self.aux_m = args.aux_m

    def train_model(self, train_data):

        # train one epoch
        epoch_loss = []
        self.model.train()

        for data, label, n_comp in train_data:

            X = data.to(self.device)  # (B, signal_size)
            y = label.to(self.device)  # (B, n_comp * 7)   7 ≙ MD, FA, X-dir, Y-dir, Z-dir, weight, existence score
            c = n_comp.to(self.device)  # (B,)

            self.opt.zero_grad()
            out = self.model(X)  # (B, n_queries, 7)

            # computing auxiliary loss
            if self.aux_loss:

                # main loss with complete transformer decoder
                hu_loss, *_ = self.crit(out['pred'], y, c)

                # adding loss of predictions from every decoder layer 
                for i, aux in enumerate(out['aux']):
                    aux_loss, *_ = self.crit(aux, y, c)
                    hu_loss += aux_loss * min([(i + 1) * self.aux_m, 1.0])

            else:
                # computing main loss
                hu_loss, *_ = self.crit(out, y, c)

                # updating weights
            hu_loss.backward()
            self.opt.step()

            # save loss for log file
            epoch_loss.append(hu_loss.cpu().item())

        return np.mean(epoch_loss)

    def pred_model(self, pred_data):

        # model evaluation
        pred_loss = []
        md_pred = []
        fa_pred = []
        di_pred = []
        wt_pred = []
        extnc_pred = []
        out = None

        self.model.eval()

        for data, label, n_comp in pred_data:

            X = data.to(self.device)  # (B, signal_size)
            y = label.to(self.device)  # (B, n_comp * 7)   7 ≙ MD, FA, X-dir, Y-dir, Z-dir, weight, existence score
            c = n_comp.to(self.device)  # (B,)

            out = self.model(X)  # (B, n_queries, 7)

            # computing main loss
            if self.aux_loss:
                loss = self.crit(out['pred'], y, c)
            else:
                loss = self.crit(out, y, c)

            # safe loss components for evaluation
            pred_loss.append(loss[0].cpu().item())
            md_pred.append(loss[1].cpu().item())
            fa_pred.append(loss[2].cpu().item())
            di_pred.append(loss[3].cpu().item())
            wt_pred.append(loss[4].cpu().item())
            extnc_pred.append(loss[5].cpu().item())

        return (np.mean(pred_loss), np.mean(md_pred), np.mean(fa_pred),
                np.mean(di_pred), np.mean(wt_pred), np.mean(extnc_pred)), \
               out

    @staticmethod
    def get_data(path, b_size, train=True, seed=None):

        # load dataset
        df_train = pd.read_csv(path).fillna(0)

        # save mri signals from columns named "mri_sigX" with X between 0 and (signal_size-1)
        X_data = df_train.filter(like="mri_sig").to_numpy(dtype=np.float32)

        # save targets from columns named "compX" with X between 1 and max(n_comp)
        y_np = df_train.filter(like="comp").to_numpy(dtype=np.float32)

        # separation of n_comp as last column in dataset
        y_data = y_np[:, :-1]
        ncomp_data = y_np[:, -1]

        # define tensors
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)
        ncomp_tensor = torch.tensor(ncomp_data, dtype=torch.int16)

        # create dataloader
        data_tensor = TensorDataset(X_tensor, y_tensor, ncomp_tensor)
        dataloader = DataLoader(data_tensor, batch_size=b_size, shuffle=train, generator=torch.Generator().manual_seed(seed))

        return dataloader

    def train_epoch(self, train_data, val_data, best_val_loss, model_path):
        # Training step
        train_loss = self.train_model(train_data)

        # Validation step using pred_model
        val_losses, _ = self.pred_model(val_data)
        # Model checkpointing
        if val_losses[0] < best_val_loss:
            torch.save(self.model.state_dict(), model_path)
            best_val_loss = val_losses[0]

        return train_loss, val_losses, best_val_loss
