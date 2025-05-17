from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info

class ValConsoleLoggerEveryN(Callback):
    def __init__(self, every_n_steps: int = 5):
        self.every_n_steps = every_n_steps

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        if (batch_idx + 1) % self.every_n_steps == 0:
            # outputs = dict returned by validation_step
            f2   = outputs["f2"]
            prec = outputs["prec"]
            rec  = outputs["rec"]
            msg = (f"[VAL] batch {batch_idx+1:>3}: "
                   f"f2={f2:.4f}, prec={prec:.4f}, rec={rec:.4f}")
            rank_zero_info(msg)
