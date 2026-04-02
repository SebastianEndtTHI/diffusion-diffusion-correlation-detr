import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd


class Train_Utils:
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

    def train_model(self, train_data, epoch):

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

    def pred_model(self, test_data):

        # model evaluation
        test_loss = []
        md_test = []
        fa_test = []
        di_test = []
        wt_test = []
        extnc_test = []
        q_loss = []

        self.model.eval()

        for data, label, n_comp in test_data:

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
            test_loss.append(loss[0].cpu().item())
            md_test.append(loss[1].cpu().item())
            fa_test.append(loss[2].cpu().item())
            di_test.append(loss[3].cpu().item())
            wt_test.append(loss[4].cpu().item())
            extnc_test.append(loss[5].cpu().item())

        return (np.mean(test_loss),
                np.mean(md_test),
                np.mean(fa_test),
                np.mean(di_test),
                np.mean(wt_test),
                np.mean(extnc_test))

    @staticmethod
    def get_data(path, b_size, train=True):

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
        dataloader = DataLoader(data_tensor, batch_size=b_size, shuffle=train)

        return dataloader

    def train_epoch(self, train_data, test_data, epoch):

        # train one epoch
        ep_loss = self.train_model(train_data, epoch)

        # evaluation run
        losses = self.pred_model(test_data)

        return ep_loss, losses
