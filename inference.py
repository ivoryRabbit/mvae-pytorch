import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model import MVAE
from src.dataset import OneHot, Dataset
from utils import upload_parquet_to_s3


item_meta = pd.read_csv("data/item_meta.csv")
target_user_hist = pd.read_csv("data/target_user_hist.csv")
item_map = pd.read_csv("data/item_map.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_state = torch.load("trained/mvae.pt", map_location=device)

model = MVAE(**model_state["args"])
model.load_state_dict(model_state["model"])

dataset = Dataset(target_user_hist, transforms=[OneHot(model_state["args"]["input_size"])])
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

item_info = (
    item_map
    .assign(weight=lambda df: df["pop"].apply(lambda x: 1 / np.sqrt(x)))
    .merge(item_meta[["item_id", "genre", "price"]], on="item_id")
    .sort_values(by="item_idx").reset_index(drop=True)
    .filter(items=["item_idx", "genre", "weight"])
)

genre_mask = np.array([1.0 if genre == "webtoon" else 0.0 for _, genre, _ in item_info.values])
weight_mask = np.array([weight for _, _, weight in item_info.values])
total_mask = genre_mask * weight_mask

K = 20
item_idxs = np.empty(shape=(0, K), dtype=int)
scores = np.empty(shape=(0, K), dtype=float)
model.eval()

with torch.no_grad():
    for batch in tqdm(dataloader):
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].cpu().numpy()

        score = model(inputs).detach().cpu().numpy()
        score = score * (1.0 - targets)
        score = score * total_mask
        score = score / np.sum(score, axis=1)[:, np.newaxis]
        indices = np.argpartition(score, -K, axis=1)[:, -K:]
        score = np.take_along_axis(score, indices, axis=1)

        item_idxs = np.append(item_idxs, indices, axis=0)
        scores = np.append(scores, score, axis=0)

idx2item_map = {idx: item for item, idx in item_map[["item_id", "item_idx"]].values}
item_ids = np.vectorize(idx2item_map.get)(item_idxs)

result = (
    dataset.df
    .assign(
        item_id=pd.Series(item_ids.tolist()),
        score=pd.Series(scores.tolist()),
    )
    .filter(items=["user_id", "item_id", "score"])
    .set_index("user_id")
    .apply(pd.Series.explode)
    .reset_index()
)

upload_parquet_to_s3(result, "marketing_by_personalize")
