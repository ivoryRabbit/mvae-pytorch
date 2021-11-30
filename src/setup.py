import os
import argparse


def get_args():
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
    return args


def set_env():
    args = get_args()
    return args
