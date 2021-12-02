import json
import logging
import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.model import MVAE
from src.dataset import OneHotEncoding, Dataset


JSON_CONTENT_TYPE = "application/json"

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Current device: {}".format(device))

    logger.info("Loading the model.")
    with open(os.path.join(model_dir, "MVAE.pt"), "rb") as f:
        model_state = torch.load(f)

    model = MVAE(
        input_size=model_state["args"]["input_size"],
        hidden_dim=model_state["args"]["hidden_dim"]
    )
    model.load_state_dict(model_state["weights"])
    model.to(device).eval()

    data_dir = os.environ["SM_CHANNEL_DATA_DIR"]
    item_map_df = pd.read_csv(os.path.join(data_dir, "item_map.csv"))
    idx2item_map = {idx: item for item, idx in item_map_df[["item_id", "item_idx"]].values}

    target_user_hist_df = pd.read_csv(os.path.join(data_dir, "target_user_hist.csv"))
    return dict(
        model=model,
        idx2item_map=idx2item_map,
        target_user_hist_df=target_user_hist_df,
    )


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info("Deserializing the input data.")
    if content_type == JSON_CONTENT_TYPE:
        return json.loads(serialized_input_data)
    raise Exception(f"Requested unsupported ContentType in content_type: {content_type}")


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info("Serializing the generated output.")
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception(f"Requested unsupported ContentType in Accept: {accept}")


def predict_fn(inputs: Dict[str, Any], model: Dict[str, Any]):
    idx2item_map = model["idx2item_map"]
    target_user_hist_df = model["target_user_hist_df"]
    model = model["model"]

    user_id = inputs["user_id"]
    rec_k = inputs["rec_k"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Current device: {}".format(device))

    target_user_hist = target_user_hist_df.query(f"user_id=={user_id}")
    dataset = Dataset(target_user_hist, transforms=[OneHotEncoding(model.input_size)])

    with torch.no_grad():
        batch = dataset[0]
        inputs = torch.from_numpy(batch["inputs"])
        inputs = torch.unsqueeze(inputs, 0)

        targets = batch["targets"]

        score = model(inputs).detach().cpu().numpy().flatten()
        score = score * (1.0 - targets)

        indices = np.argpartition(score, -rec_k, axis=-1)[-rec_k:]
        score = np.take_along_axis(score, indices, axis=-1)
        indices = indices[np.argsort(-score, axis=-1)]

        item_ids = np.vectorize(idx2item_map.get)(indices)

    return ",".join(item_ids)


