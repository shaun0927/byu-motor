from lightning.pytorch.callbacks import Callback

class FreezeBackbone(Callback):
    """Freeze backbone for first ``freeze_epochs`` epochs."""

    def __init__(self, freeze_epochs: int = 0):
        self.freeze_epochs = int(freeze_epochs)

    def on_train_start(self, trainer, pl_module):
        if self.freeze_epochs > 0:
            for p in pl_module.net.backbone.parameters():
                p.requires_grad = False

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        if trainer.current_epoch + 1 == self.freeze_epochs:
            for p in pl_module.net.backbone.parameters():
                p.requires_grad = True

