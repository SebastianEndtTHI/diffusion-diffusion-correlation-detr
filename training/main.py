import torch

import argparse
import numpy as np
import time
import datetime

import dl_models as models
import match_loss as Loss
from train_utils import Train_Utils



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)


    # paths
    parser.add_argument('--train_data_path', default=None, type=str)
    parser.add_argument('--test_data_path', default=None, type=str)

    parser.add_argument('--model_save_path', default="detr_model", type=str)
    parser.add_argument('--log_save_path', default="detr_logs", type=str)

    parser.add_argument('--model_path', default=None, type=str, help="path of trained model.")
    parser.add_argument('--pretrain_path', default=None, type=str, help="path of pretrained encoder.")


    # train parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_step', default=1000, type=int, help="step size for lr scheduler.")
    parser.add_argument('--w_decay', default=1e-4, type=float)
    parser.add_argument('--opt_betas', default=(0.9,0.98), type=tuple, help="AdamW parameters.")

    parser.add_argument('--b_size', default=256, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    
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
    parser.add_argument('--exs_loss_weight', default=0.01, type=float, help="lamda weight of esxistence score-loss.")
    
    parser.add_argument('--aux_loss', default=False, type=bool, help="activating the auxiliary loss.")
    parser.add_argument('--aux_m', default=1.0, type=float, help="linear auxiliary loss weighting increasing for later decoder blocks.")


    return parser



def main(args):
    start = time.time()


    # model initialization
    model = models.DWI_DETR_Att(args)

    # load weights of trained model if defined
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))

    model.to(args.device)


    # train initialization
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr= args.lr, 
                                  weight_decay= args.w_decay, 
                                  betas= args.opt_betas)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step)
    
    criterion = Loss.HungarianLoss(args)

    setup = Train_Utils(model=model, 
                        optimizer=optimizer, 
                        criterion=criterion,
                        args=args)
    

    # loading dataset in dataloader
    train_data = setup.get_data(args.train_data_path, args.b_size, train=True)
    test_data = setup.get_data(args.test_data_path, args.b_size, train=False)
    

    # loss dictionary with all components
    losses = {}
    losses["train_loss"] = []
    losses["test_loss"] = []
    losses["test_loss_md"] = []
    losses["test_loss_fa"] = []
    losses["test_loss_di"] = []
    losses["test_loss_wt"] = []
    losses["test_loss_extnc"] = []


    # check for auxiliary loss
    print("Auxiliary loss activated: ", args.aux_loss)

    print("Start training")


    # start training
    for epoch in range(args.epochs):

        # training and evaluation run for one epoch
        ep_loss, test_losses = setup.train_epoch(train_data, test_data, epoch)

        scheduler.step()


        # save aggregated results
        losses["train_loss"].append(ep_loss)
        losses["test_loss"].append(test_losses[0])
        losses["test_loss_md"].append(test_losses[1])
        losses["test_loss_fa"].append(test_losses[2])
        losses["test_loss_di"].append(test_losses[3])
        losses["test_loss_wt"].append(test_losses[4])
        losses["test_loss_extnc"].append(test_losses[5])


        # save model and log file every 20 epochs
        if epoch % 20 == 0:

            torch.save(model.state_dict(), args.model_save_path + f"_{epoch}ep")
            np.save(args.log_save_path + f"_{epoch}ep", losses)


        # epoch informations
        print(f"-{str(datetime.datetime.now())} " +
                f"- Epoche: {epoch+1:02d}/{args.epochs} " +
                f"- Train Loss: {ep_loss:.10f} " +
                f"- Test Loss: {test_losses[0]:.10f} " +
                f"- MD Loss: {test_losses[1]:.10f} " +
                f"- FA Loss: {test_losses[2]:.10f} " +
                f"- Direction Loss: {test_losses[3]:.10f} " +
                f"- Weight Loss: {test_losses[4]:.10f} " +
                f"- Existence Loss: {test_losses[5]:.10f}")
        

        # clearing gpu cache
        torch.cuda.empty_cache() 


    # safe final model and losses
    torch.save(model.state_dict(), args.model_save_path)

    np.save(args.log_save_path, losses)


    # calcualting train duration
    end = time.time()
    duration = end-start
    struct_time = time.strftime("%H:%M:%S",time.gmtime(duration))

    print(f"Training finished in {struct_time}")



# code initialization
if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # run training
    main(args)
