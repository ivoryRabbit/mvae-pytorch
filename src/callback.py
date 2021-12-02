import numpy as np
import torch
from typing import Dict, Any


class EarlyStopping(object):
    def __init__(
            self,
            checkpoint_path: str,
            patience: int,
            delta: float = 0.0,
            min_mode: bool = True
    ):
        self.checkpoint_path = checkpoint_path
        self.min_mode = min_mode
        self.patience = patience
        self.delta = delta

        self.is_early_stop = False
        self.count = 0
        self.best_eval_loss = np.Inf if min_mode else -np.Inf

    def check(self, eval_loss: float, checkpoint: Dict[str, Any]) -> bool:
        score = self._get_score(eval_loss)
        best_score = self._get_score(self.best_eval_loss)

        if score + self.delta <= best_score:
            print(f"Validation loss: {self.best_eval_loss:.4f} --> {eval_loss:.4f}")
            self.__init_count()
            self.__update_best(eval_loss)
            self.save_checkpoint(checkpoint)
        else:
            self.count += 1
            print(f"Early Stopping count: {self.count} is out of patience: {self.patience}")

        if self.count >= self.patience:
            return True
        return False

    def __init_count(self):
        self.count = 0

    def __update_best(self, eval_loss: float):
        self.best_eval_loss = eval_loss

    def _get_score(self, loss: float) -> float:
        return loss if self.min_mode else -loss

    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        with open(self.checkpoint_path, "wb") as f:
            torch.save(checkpoint, f)

    def load_checkpoint(self) -> Dict[str, Any]:
        with open(self.checkpoint_path, "rb") as f:
            checkpoint = torch.load(f)
        return checkpoint
