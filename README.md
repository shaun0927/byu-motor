
## MotorDataModule instance-crop training

The data module can optionally construct datasets that crop patches
centered on each annotated motor instance. Enable this behaviour with the
`use_instance_crop` flag and control the number of crops per instance and
the amount of extra background crops with `num_crops` and `neg_ratio`.

Example:

```python
from motor_det.data.module import MotorDataModule

dm = MotorDataModule(
    data_root="data",
    fold=0,
    batch_size=2,
    use_instance_crop=True,
    num_crops=3,
    neg_ratio=0.5,
)
```

When enabled the module uses `MotorInstanceCropDataset` for positive
patches and optionally concatenates `BackgroundRandomCropDataset` for
additional negative examples.
