import os
import time
import numpy as np
import torch
import torch.nn as nn
from lib.logger import get_logger
from torch.utils.tensorboard import SummaryWriter

def adjust_learning_rate(optimizer, epoch, learning_rate):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lradj = 'type2'
    if lradj=='type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif lradj=='type2':
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 10: 5e-6, 16: 1e-6
        # }
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        # lr_adjust = {
        #     4: 5e-5, 8: 1e-5, 12: 5e-6, 16: 1e-6, 
        #     20: 5e-7, 30: 1e-7, 40: 5e-8
        # }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def get_fr(epoch):
    if epoch <= 4:
        return 0.1, 0.9 # 0.2, 0.8
    else:
        return 0.9, 0.1


class Trainer:
    """Trainer class.

    :param model: model
    :param optimizer: Optimizer used to minimize the loss function
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param target_dims: dimension of input features to forecast and reconstruct
    :param n_epochs: Number of iterations/epochs
    :param batch_size: Number of windows in a single batch
    :param init_lr: Initial learning rate of the module
    :param forecast_criterion: Loss to be used for forecasting.
    :param recon_criterion: Loss to be used for reconstruction.
    :param boolean use_cuda: To be run on GPU or not
    :param dload: Download directory where models are to be dumped
    :param log_dir: Directory where SummaryWriter logs are written to
    :param print_every: At what epoch interval to print losses
    :param log_tensorboard: Whether to log loss++ to tensorboard
    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard
    """

    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        use_cuda=True,
        dload="",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
        patience=15,
        debug=False,
        run_mode='all'
    ):

        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard
        self.patience = patience
        self.run_mode=run_mode

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
            "val_best": float('inf')
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

        # log
        self.logger = get_logger(dload, name='model', debug=debug)
        # self.logger.info('use_cuda:', next(model.parameters()).is_cuda)
        self.logger.info('Experiment log path in: {}'.format(dload))
        self.logger.info("Argument: %s", args_summary)
        self.logger.info("use_cuda:{}".format(self.device))

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """

        # init_train_loss = self.evaluate(train_loader)
        # self.logger.info(f"Init total train loss: {init_train_loss[2]:5f}")

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            self.logger.info(f"Init total val loss: {init_val_loss[2]:.5f}")

        self.logger.info(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()

        not_improved_count = 0

        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []
            recon_b_losses = []

            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                preds, recons, mu, logvar = self.model(x)
                # if self.model.AE != 'VAE':
                if mu is None or logvar is None:
                    KLD = 0
                else:
                    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
                    KLD = torch.sum(KLD_element).mul_(-0.5)/mu.nelement()

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)
                    recons = recons[:, :, self.target_dims]
                    preds = preds[:, self.target_dims]


                if y.ndim == 3:
                    y = y.squeeze(1)

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))+KLD
                # forecast_loss = torch.zeros_like(forecast_loss)
                if self.run_mode == 'all':
                    a, b = get_fr(epoch)
                    loss = a*forecast_loss + b*recon_loss
                else:
                    loss = forecast_loss if self.run_mode == 'fore' else recon_loss

                loss.backward()
                self.optimizer.step()

                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())
                if (i+1) % 200==0:
                    self.logger.info("\titers: {0}/{1}, epoch: {2} | loss: {3:.7f}".format(i + 1, len(train_loader), epoch + 1, loss.item()))


            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            # Evaluate on validation set
            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"
            if val_loader is not None:
                forecast_val_loss, recon_val_loss, total_val_loss = self.evaluate(val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_total"].append(total_val_loss)

                if total_val_loss <= self.losses["val_best"]:
                    self.save(f"model.pt")
                    self.losses["val_best"] = total_val_loss
                    not_improved_count = 0
                else:
                    not_improved_count += 1
                if self.patience != 0:
                    if not_improved_count == self.patience:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.patience))
                        break


            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )

                s += f" [{epoch_time:.1f}s]"
                self.logger.info(s)
            adjust_learning_rate(self.optimizer, epoch+1, self.init_lr)

        # if val_loader is None:
        #     self.save(f"model.pt")

        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        self.logger.info(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):
        """Evaluate model

        :param data_loader: data loader of input data
        :return forecasting loss, reconstruction loss, total loss
        """

        self.model.eval()

        forecast_losses = []
        recon_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds, recons, mu, logvar = self.model(x)
                # if self.model.AE != 'VAE':
                #     KLD = 0
                if mu is None or logvar is None:
                    KLD = 0
                else:
                    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
                    # print(mu.shape)
                    KLD = torch.sum(KLD_element).mul_(-0.5)/mu.shape[0]/mu.shape[1]

                if preds.ndim == 3:
                    preds = preds.squeeze(1)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)
                    recons = recons[:, :, self.target_dims]
                    preds = preds[:, self.target_dims]

                if y.ndim == 3:
                    y = y.squeeze(1)

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))+KLD

                forecast_losses.append(forecast_loss.item())
                recon_losses.append(recon_loss.item())

        forecast_losses = np.array(forecast_losses)
        recon_losses = np.array(recon_losses)

        forecast_loss = np.sqrt((forecast_losses ** 2).mean())
        recon_loss = np.sqrt((recon_losses ** 2).mean())

        if self.run_mode == 'all':
            # total_loss = forecast_loss + recon_loss
            total_loss = forecast_loss
        else:
            total_loss = forecast_loss if self.run_mode == 'fore' else recon_loss

        return forecast_loss, recon_loss, total_loss

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        PATH = self.dload + "/" + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)
