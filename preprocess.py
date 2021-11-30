import pandas as pd
import numpy as np
from typing import Iterable
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split


def get_interaction(df: DataFrame) -> DataFrame:
    return (
        df
        .sort_values(by=["user_id", "timestamp"])
        .drop_duplicates(subset=["user_id", "item_id"], keep="last")
        .filter(items=["user_id", "item_id"])
        .reset_index(drop=True)
    )


def drop_sparse(df: DataFrame, trunc_min: int, trunc_max: int) -> DataFrame:
    user_hist_size = df.groupby(by="user_id")["item_id"].count()
    fine_user = user_hist_size.index[user_hist_size.between(trunc_min, trunc_max)]
    return df[df["user_id"].isin(fine_user)]


def split_by_user(df: DataFrame, test_size=0.1, shuffle=True) -> Iterable[DataFrame]:
    user_ids = df["user_id"].unique()
    train_user_ids, test_user_ids = train_test_split(user_ids, test_size=test_size, shuffle=shuffle)

    def filter_by_id(ids: np.ndarray) -> DataFrame:
        return df[df["user_id"].isin(ids)]

    return map(filter_by_id, [train_user_ids, test_user_ids])


def calculate_penalty(df: DataFrame, score: str) -> Series:
    # lambda = 1 - (score - min of score) / (max of score - min of score)
    return 1 - (df[score] - df[score].min()) / (df[score].max() - df[score].min())


# Preprocess purchase data
purchase = pd.read_csv("data/purchase.csv")

interaction = get_interaction(purchase)
interaction = drop_sparse(interaction, trunc_min=5, trunc_max=200)

train, test = split_by_user(interaction, test_size=0.01)

item_ids = train["item_id"].unique()
test = test[test["item_id"].isin(item_ids)]

n_item = len(item_ids)
item2idx_map = {item: idx for idx, item in enumerate(item_ids)}

train = train.assign(item_idx=lambda df: df["item_id"].map(item2idx_map))
test = test.assign(item_idx=lambda df: df["item_id"].map(item2idx_map))

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

# Preprocess item meta data
item_meta = pd.read_csv("data/item_meta.csv")

pop = interaction.groupby("item_id", as_index=False).agg(pop=("user_id", "count"))

item_map = (
    item_meta[["item_id"]][item_meta["item_id"].isin(item_ids)]
    .merge(pop, on="item_id", how="inner")
    .assign(
        item_idx=lambda df: df["item_id"].map(item2idx_map),
        penalty=lambda df: calculate_penalty(df, score="pop")
    )
)

item_map.to_csv("data/item_map.csv", index=False)

# Preprocess user candidate data
target_user = pd.read_csv("data/target_user.csv")

target_user_hist = interaction[interaction["user_id"].isin(target_user["u_idx"])]
target_user_hist = target_user_hist[target_user_hist["item_id"].isin(item_ids)]
target_user_hist = target_user_hist.assign(item_idx=lambda df: df["item_id"].map(item2idx_map))

target_user_hist.to_csv("data/target_user_hist.csv", index=False)
