import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import OneHotEncoding, Dataset
from src.model import MVAE
from src.loss_function import LossFunction
from src.optimizer import Optimizer
from src.metric import Metric
from src.callback import EarlyStopping
from src.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--model-name", default="MVAE.pt", type=str)
    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--n-layers", default=1, type=int)
    parser.add_argument("--anneal-rate", default=0.3, type=float)
    parser.add_argument("--seed", default=777, type=int, help="seed for random initialization")

    # learning parameters
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=50, type=int)
    parser.add_argument("--lr", default=0.05, type=float, help="learning rate")

    # optimizer parameters
    parser.add_argument("--optimizer-type", default="Adam", type=str, help="Adam, RMSProp, SGD")
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--momentum", default=0.1, type=float)
    parser.add_argument("--eps", default=1e-6, type=float)

    # callback parameters
    parser.add_argument("--patience", default=5, type=int, help="early stopping patience")
    parser.add_argument("--delta", default=0.0, type=float, help="early stopping threshold")

    # inference
    parser.add_argument("--eval-k", default=25, type=int, help="how many items you recommend")
    parser.add_argument("--user-id", type=int, help="user id")

    # directions
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--checkpoint-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])

    args = parser.parse_args()

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
