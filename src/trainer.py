import numpy as np
import torch
import time
from tqdm.notebook import tqdm


class Trainer(object):
    _device = None

    def __init__(
        self,
        args,
        model,
        train_dataloader,
        valid_dataloader,
        optimizer,
        loss_function,
        metric,
        early_stopping,
    ):
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metric = metric
        self.early_stopping = early_stopping

    @property
    def device(self):
        if self._device:
            return self._device
        raise ValueError

    def to(self, device):
        self._device = device

    def fit(self, n_epochs):
        self.model.to(self.device)
        self.loss_function.to(self.device)

        start_time = time.time()
        train_losses, eval_losses = [], []
        epoch_len = len(str(n_epochs))

        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.time()
            print(f"Start Epoch #{epoch}")

            train_loss = self.train_step()
            train_losses.append(train_loss)

            eval_loss, metrics = self.valid_step()
            eval_losses.append(eval_loss)

            print(
                f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] " +
                f"--train_loss: {train_loss:.5f} " +
                f"--valid_loss: {eval_loss:.5f} " +
                "".join([
                    f"--{metric}: {value:.5f} "
                    for metric, value in metrics.items()
                ]) +
                f"--time: {time.time() - epoch_start_time:0.1f} sec"
            )

            checkpoint = dict(
                epoch=epoch,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                loss=eval_loss,
            )
            checkpoint.update(metrics)

            is_early_stop = self.early_stopping.check(eval_loss, checkpoint)
            if is_early_stop:
                print("Early stopped")
                break

        try:
            best_checkpoint = self.early_stopping.load_checkpoint()
            self.model.load_state_dict(best_checkpoint["model"])
        except FileNotFoundError:
            pass

        print(f"time: {time.time() - start_time:0.1f} sec")
        return self.model, train_losses, eval_losses

    def train_step(self):
        self.model.train()
        self.loss_function.train()
        losses = []

        for batch in tqdm(self.train_dataloader, miniters=1000):
            self.optimizer.zero_grad()

            inputs = batch["inputs"].to(self.device)
            targets = batch["targets"].to(self.device)

            outputs, mean, log_var = self.model(inputs)

            loss = self.loss_function(outputs, targets, mean, log_var)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()

        return np.mean(losses)

    def valid_step(self):
        self.model.eval()
        self.loss_function.eval()
        losses = []
        metrics = dict()

        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader, miniters=1000):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(inputs)

                loss = self.loss_function(outputs, targets)
                losses.append(loss.item())

                eval_metrics = self.metric(outputs, targets)

                for metric, value in eval_metrics.items():
                    metrics[metric] = metrics.get(metric, []) + [value]

        mean_loss = np.mean(losses)
        mean_metrics = {metric: np.mean(value) for metric, value in metrics.items()}

        return mean_loss, mean_metrics
