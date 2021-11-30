import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from setup import set_env
from src.dataset import OneHotEncoding, Dataset
from src.model import MVAE
from src.loss_function import LossFunction
from src.optimizer import Optimizer
from src.metric import Metric
from src.callback import EarlyStopping
from src.trainer import Trainer


def train_step(dataloader: DataLoader):
    model.train()
    loss_function.train()
    losses = []

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)

        outputs, mean, log_var = model(inputs)

        loss = loss_function(outputs, targets, mean, log_var)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return np.mean(losses)


def valid_step(dataloader: DataLoader):
    model.eval()
    loss_function.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs)

            loss = loss_function(outputs, targets)
            losses.append(loss.item())

    return np.mean(losses)


if __name__ == "__main__":
    args = set_env()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(args.data_dir, "train.csv")
    valid_path = os.path.join(args.data_dir, "test.csv")
    item_path = os.path.join(args.data_dir, "item_map.csv")

    checkpoint_path = os.path.join(args.checkpoint_dir, args.model_name)
    model_path = os.path.join(args.model_dir, args.model_name)

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    item_map_df = pd.read_csv(item_path)

    input_size = n_items = len(item_map_df)

    model = MVAE(input_size=input_size, hidden_dim=args.hidden_dim)

    penalty = item_map_df["penalty"].values
    penalty_term = torch.Tensor(penalty).to(device)
    loss_function = LossFunction(penalty_term, anneal_rate=args.anneal_rate)

    optimizer = Optimizer(model.parameters(), optimizer_type=args.optimizer_type, lr=args.lr)

    transforms = [OneHotEncoding(n_items)]
    train_dataset = Dataset(train_df, transforms=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = Dataset(valid_df, transforms=transforms)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    early_stopping = EarlyStopping(args, checkpoint_path)
    metric = Metric(args)

    trainer = Trainer(
        args,
        model,
        train_dataloader,
        valid_dataloader,
        optimizer,
        loss_function,
        metric,
        early_stopping,
    )
    trainer.to(device)
    trainer.fit(args.n_epochs)

    model_state = dict(
        model=model.state_dict(),
        args=dict(input_size=n_items, hidden_dim=args.hidden_dim)
    )

    with open(model_path, "wb") as f:
        torch.save(model_state, model_path)
