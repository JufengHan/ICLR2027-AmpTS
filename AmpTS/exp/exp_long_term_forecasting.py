from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.AmpTS import Model as AmpTSModel
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim


class Exp_Long_Term_Forecast(Exp_Basic):
    def _build_model(self):
        if self.args.model != "AmpTS":
            raise ValueError("This release only supports model=AmpTS")
        model = AmpTSModel(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _forward_model(self, batch_x, batch_x_mark):
        return self.model(batch_x, batch_x_mark, None, None)

    def vali(self, vali_data, vali_loader, criterion):
        del vali_data
        losses = []
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                f_dim = -1 if self.args.features == "MS" else 0
                true = batch_y[:, -self.args.pred_len:, f_dim:]

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark)
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                if outputs.shape[-1] != true.shape[-1]:
                    outputs = outputs[:, :, f_dim:]
                losses.append(criterion(outputs.detach(), true.detach()).item())

        self.model.train()
        return float(np.average(losses))

    def train(self, setting):
        _, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            train_losses = []
            self.model.train()
            epoch_start = time.time()

            for batch_x, batch_y, batch_x_mark, _ in train_loader:
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                f_dim = -1 if self.args.features == "MS" else 0
                true = batch_y[:, -self.args.pred_len:, f_dim:]

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark)
                        outputs = outputs[:, -self.args.pred_len:, :]
                        if outputs.shape[-1] != true.shape[-1]:
                            outputs = outputs[:, :, f_dim:]
                        loss = criterion(outputs, true)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    if outputs.shape[-1] != true.shape[-1]:
                        outputs = outputs[:, :, f_dim:]
                    loss = criterion(outputs, true)
                    loss.backward()
                    model_optim.step()

                train_losses.append(loss.item())

            train_loss = float(np.average(train_losses))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            elapsed = time.time() - epoch_start
            current_lr = model_optim.param_groups[0]["lr"]

            print(
                "Epoch: {}/{} | Train Loss: {:.7f} | Vali Loss: {:.7f} | LR: {:.3e} | Time: {:.1f}s".format(
                    epoch + 1,
                    self.args.train_epochs,
                    train_loss,
                    vali_loss,
                    current_lr,
                    elapsed,
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.args.checkpoints, setting, "checkpoint.pth"),
                    map_location=self.device,
                )
            )

        preds = []
        trues = []
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, _ in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                f_dim = -1 if self.args.features == "MS" else 0

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark)
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = batch_y_np.shape

                    def inverse_forecast(array):
                        if array.shape[-1] != batch_y_np.shape[-1]:
                            repeat = int(batch_y_np.shape[-1] / array.shape[-1])
                            array = np.tile(array, [1, 1, repeat])
                        return test_data.inverse_transform(
                            array.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)

                    outputs_np = inverse_forecast(outputs_np)
                    batch_y_np = test_data.inverse_transform(
                        batch_y_np.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                outputs_np = outputs_np[:, :, f_dim:]
                batch_y_np = batch_y_np[:, :, f_dim:]
                preds.append(outputs_np)
                trues.append(batch_y_np)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, _, _, _ = metric(preds, trues)

        print("Test MSE: {:.6f} | Test MAE: {:.6f}".format(mse, mae))

        result_path = os.path.join("./results", setting)
        os.makedirs(result_path, exist_ok=True)
        np.save(os.path.join(result_path, "metrics.npy"), np.array([mse, mae]))
        np.save(os.path.join(result_path, "pred.npy"), preds)
        np.save(os.path.join(result_path, "true.npy"), trues)

        with open("result_long_term_forecast.txt", "a", encoding="utf-8") as file:
            file.write(setting + "\n")
            file.write("mse:{}, mae:{}\n\n".format(mse, mae))

        return mse, mae
