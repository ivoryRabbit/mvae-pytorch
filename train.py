import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from setup import set_env
from src.dataset import OneHotEncoding, Dataset
from src.model import MVAE
from src.loss_function import LossFunction
from src.optimizer import Optimizer
from src.metric import Metric
from src.callback import EarlyStopping
from src.trainer import Trainer


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

    early_stopping = EarlyStopping(checkpoint_path, patience=args.patience, delta=args.delta)
    metric = Metric(eval_k=args.eval_k)

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

    model.save(model_path, args=vars(args))
