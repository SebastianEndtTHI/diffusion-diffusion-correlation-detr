import torch
import numpy as np
import pickle

import argparse
import time
import datetime
import os
import random

import dl_models
import match_loss
from train_utils import TrainUtils


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # paths
    parser.add_argument('--train_data_file', default=None, type=str)
    parser.add_argument('--val_data_file', default=None, type=str)
    parser.add_argument('--test_data_file', default=None, type=str)

    parser.add_argument('--model_folder', default=None, type=str)

    parser.add_argument('--model_path', default=None, type=str, help="path of trained model.")
    parser.add_argument('--pretrain_path', default=None, type=str, help="path of pretrained encoder.")

    # reproducibility
    parser.add_argument('--seed', default=0, type=int, help="random seed for reproducibility.")

    # train parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_step', default=1000, type=int, help="step size for lr scheduler.")
    parser.add_argument('--w_decay', default=1e-4, type=float)
    parser.add_argument('--opt_betas', default=(0.9, 0.98), type=tuple, help="AdamW parameters.")

    parser.add_argument('--b_size', default=256, type=int)
    parser.add_argument('--epochs', default=150, type=int)

    parser.add_argument('--device', default="cuda", type=str)

    # model parameters
    parser.add_argument('--input_dim', default=331, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)

    parser.add_argument('--fs_dim', default=256, type=int, help="size of feature and query vectors.")
    parser.add_argument('--n_queries', default=10, type=int)

    parser.add_argument('--n_dlayers', default=4, type=int, help="number of decoder blocks.")
    parser.add_argument('--n_multihead', default=4, type=int, help="number of attention heads.")

    parser.add_argument('--freeze_encoder', default=False, type=bool, help="freezing encoder weights.")

    # loss parameters
    parser.add_argument('--md_loss_weight', default=5.0, type=float, help="lamda weight of MD-loss.")
    parser.add_argument('--fa_loss_weight', default=2.0, type=float, help="lamda weight of FA-loss.")
    parser.add_argument('--dir_loss_weight', default=0.5, type=float, help="lamda weight of direction-loss.")
    parser.add_argument('--wt_loss_weight', default=1.0, type=float, help="lamda weight of weight-loss.")
    parser.add_argument('--exs_loss_weight', default=0.01, type=float, help="lamda weight of existence score-loss.")

    parser.add_argument('--aux_loss', default=False, type=bool, help="activating the auxiliary loss.")
    parser.add_argument('--aux_m', default=1.0, type=float,
                        help="linear auxiliary loss weighting increasing for later decoder blocks.")

    return parser


def main(args):
    start = time.time()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(args.seed)

    # model initialization
    model = dl_models.DWIdetr(args)

    # load weights of trained model if defined
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))

    model.to(args.device)

    # train initialization
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr,
                                  weight_decay=args.w_decay,
                                  betas=args.opt_betas)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step)

    criterion = match_loss.HungarianLoss(args)

    setup = TrainUtils(model=model,
                       optimizer=optimizer,
                       criterion=criterion,
                       args=args)

    if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(__file__), '../models', args.model_folder))):
        os.makedirs(os.path.normpath(os.path.join(os.path.dirname(__file__), '../models', args.model_folder)))
    best_model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../models', args.model_folder, "detr_model_best"))

    # loading dataset in dataloader
    train_data = setup.get_data(os.path.normpath(os.path.join(os.path.dirname(__file__), '../data', args.train_data_file)), args.b_size, train=True, seed=args.seed)
    val_data = setup.get_data(os.path.normpath(os.path.join(os.path.dirname(__file__), '../data', args.val_data_file)), args.b_size, train=False, seed=args.seed)
    test_data = setup.get_data(os.path.normpath(os.path.join(os.path.dirname(__file__), '../data', args.test_data_file)), args.b_size, train=False, seed=args.seed)

    # loss dictionary with all components
    losses = {"train_loss": [], "val_loss": [], "val_loss_md": [], "val_loss_fa": [], "val_loss_di": [],
              "val_loss_wt": [], "val_loss_extnc": []}

    # check for auxiliary loss
    print("Auxiliary loss activated: ", args.aux_loss)

    print("Start training")

    # start training
    best_val_loss = np.inf
    for epoch in range(args.epochs):

        # training and evaluation run for one epoch
        ep_loss, val_losses, best_val_loss = setup.train_epoch(
            train_data, val_data, best_val_loss, best_model_path)

        scheduler.step()

        # save aggregated results
        losses["train_loss"].append(ep_loss)
        losses["val_loss"].append(val_losses[0])
        losses["val_loss_md"].append(val_losses[1])
        losses["val_loss_fa"].append(val_losses[2])
        losses["val_loss_di"].append(val_losses[3])
        losses["val_loss_wt"].append(val_losses[4])
        losses["val_loss_extnc"].append(val_losses[5])

        # save model and log file every 20 epochs
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.normpath(os.path.join(os.path.dirname(__file__), '../models', args.model_folder,  f"detr_model_{epoch}ep")))
            np.save(os.path.normpath(os.path.join(os.path.dirname(__file__), '../models', args.model_folder, f"detr_logs_{epoch}ep")), losses)

        # epoch information
        print(f"-{str(datetime.datetime.now())} " +
              f"- Epoch: {epoch + 1:02d}/{args.epochs} " +
              f"- Train Loss: {ep_loss:.10f} " +
              f"- Val Loss: {val_losses[0]:.10f} " +
              f"- MD Loss: {val_losses[1]:.10f} " +
              f"- FA Loss: {val_losses[2]:.10f} " +
              f"- Direction Loss: {val_losses[3]:.10f} " +
              f"- Weight Loss: {val_losses[4]:.10f} " +
              f"- Existence Loss: {val_losses[5]:.10f}")

        # clearing gpu cache
        torch.cuda.empty_cache()

    with open(os.path.normpath(os.path.join(os.path.dirname(__file__), '../models', args.model_folder, 'detr_logs.pkl')), 'wb') as f:
        pickle.dump(losses, f)

    # calculating train duration
    end = time.time()
    duration = end - start
    struct_time = time.strftime("%H:%M:%S", time.gmtime(duration))

    print(f"Training finished in {struct_time}")

    # test evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    model.eval()

    all_predictions = []
    all_ground_truths = []
    all_n_comps = []

    with torch.no_grad():
        for data, label, n_comp in test_data:
            X = data.to(args.device)
            y = label.to(args.device)
            outputs = model(X)
            pred = outputs  # ['pred']
            all_predictions.append(pred.cpu().numpy())
            all_ground_truths.append(y.cpu().numpy())
            # n_comp can be a tensor or numpy array, convert to numpy
            if torch.is_tensor(n_comp):
                all_n_comps.append(n_comp.cpu().numpy())
            else:
                all_n_comps.append(np.array(n_comp))

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)
    all_n_comps = np.concatenate(all_n_comps, axis=0)

    individual_losses = []
    for idx, (pred_np, gt_np) in enumerate(zip(all_predictions, all_ground_truths)):
        pred_tensor = torch.tensor(pred_np, dtype=torch.float32, device=args.device).unsqueeze(0)
        gt_tensor = torch.tensor(gt_np, dtype=torch.float32, device=args.device).unsqueeze(0)
        n_comp = [all_n_comps[idx]]  # HungarianLoss expects a list/array per batch
        loss_components = criterion(pred_tensor, gt_tensor, n_comp)
        individual_losses.append([l.item() if hasattr(l, 'item') else float(l) for l in loss_components])

    results = {
        'predictions': all_predictions,
        'ground_truths': all_ground_truths,
        'losses': individual_losses,
        'n_comp': all_n_comps
    }

    with open(os.path.normpath(os.path.join(os.path.dirname(__file__), '../models', 'detr_output.pkl')),
              'wb') as f:
        pickle.dump(results, f)


# code initialization
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parserargs = parser.parse_args()

    # run training
    main(parserargs)
