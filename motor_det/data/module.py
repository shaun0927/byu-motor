from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Iterable

import lightning as L
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedGroupKFold

from motor_det.data.detection import (
    InstanceCropDataset,
    RandomCropDataset,
    SlidingWindowDataset,
)
from motor_det.utils.collate import collate_with_centers
from motor_det.utils.voxel import (
    voxel_spacing_map,
    read_train_centers,
    DEFAULT_TEST_SPACING,
)


class MotorDataModule(L.LightningDataModule):
    """Build BYU-Motor train / val DataLoaders (crop-based)."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        data_root: str,
        fold: int,
        *,
        batch_size: int = 2,
        num_workers: int = 12,
        persistent_workers: bool = True,
        positive_only: bool = False,
        train_num_instance_crops: int = 128,
        train_num_random_crops: int = 0,
        train_include_sliding_dataset: bool = False,
        train_crop_size: tuple[int, int, int] = (96, 128, 128),
        valid_crop_size: tuple[int, int, int] = (192, 128, 128),
        pin_memory: bool = False,
        prefetch_factor: int | None = 2,
        preload_volumes: bool = True,
        use_gpu_augment: bool = True,
        valid_use_gpu_augment: bool | None = None,
        mixup_prob: float = 0.0,
        cutmix_prob: float = 0.0,
        copy_paste_prob: float = 0.0,
        copy_paste_limit: int = 1,
        val_num_crops: int = 128,
    ):
        super().__init__()
        self.root = Path(data_root)
        self.fold = fold

        # dataloader 설정
        self.bs = batch_size
        self.nw = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.preload_volumes = preload_volumes

        # dataset 파라미터
        self.positive_only = bool(positive_only)
        self.train_num_instance_crops = int(train_num_instance_crops)
        self.train_num_random_crops = int(train_num_random_crops)
        self.train_include_sliding_dataset = bool(train_include_sliding_dataset)
        self.train_crop_size = train_crop_size
        self.valid_crop_size = valid_crop_size
        self.val_num_crops = int(val_num_crops)

        # augmentation
        self.use_gpu_augment = bool(use_gpu_augment)
        self.valid_use_gpu_augment = (
            self.use_gpu_augment if valid_use_gpu_augment is None else bool(valid_use_gpu_augment)
        )
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.copy_paste_prob = copy_paste_prob
        self.copy_paste_limit = copy_paste_limit

    # -------------------- fold split util ------------------------------ #
    @staticmethod
    def _split_folds(df, n_splits: int = 5, seed: int = 42):
        if "fold" in df.columns and df["fold"].nunique() > 1:
            return df

        df = df.copy()
        df["motor_present"] = (df["Motor axis 0"] >= 0).astype(int)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_arr = np.full(len(df), -1, dtype=int)
        for f, (_, val_idx) in enumerate(sgkf.split(df, y=df["motor_present"], groups=df["tomo_id"])):
            fold_arr[val_idx] = f
        df["fold"] = fold_arr
        return df

    # -------------------- dataset builder ------------------------------ #
    def _make_crop_ds(
        self,
        ids: Iterable[str],
        centers_df,
        spacing_map,
        *,
        training: bool,
        use_gpu: bool,
    ):
        crop_size = self.train_crop_size if training else self.valid_crop_size
        datasets = []

        for tid in ids:
            zarr_path = self.root / "processed" / "zarr" / "train" / f"{tid}.zarr"
            if not zarr_path.exists():
                continue

            sub = centers_df[centers_df["tomo_id"] == tid]
            sub = sub[sub["Motor axis 0"] >= 0]          # GT 필터
            centers = sub[["Motor axis 2", "Motor axis 1", "Motor axis 0"]].values.astype(np.float32)
            vx = spacing_map.get(tid, DEFAULT_TEST_SPACING)

            # positive-only 모드라면 GT 없는 tomo skip
            if self.positive_only and len(centers) == 0:
                continue

            if training:
                sub_datasets = [
                    InstanceCropDataset(
                        zarr_path,
                        centers,
                        vx,
                        crop_size=self.train_crop_size,
                        num_crops=self.train_num_instance_crops,
                        negative_ratio=0.0 if self.positive_only else 0.2,
                        preload_volume=self.preload_volumes,
                        use_gpu=use_gpu,
                        mixup_prob=self.mixup_prob,
                        cutmix_prob=self.cutmix_prob,
                        copy_paste_prob=self.copy_paste_prob,
                        copy_paste_limit=self.copy_paste_limit,
                    ),
                ]
                if self.train_num_random_crops > 0:
                    sub_datasets.append(
                        RandomCropDataset(
                            zarr_path,
                            centers,
                            vx,
                            crop_size=self.train_crop_size,
                            num_crops=self.train_num_random_crops,
                            preload_volume=self.preload_volumes,
                            use_gpu=use_gpu,
                            mixup_prob=self.mixup_prob,
                            cutmix_prob=self.cutmix_prob,
                            copy_paste_prob=self.copy_paste_prob,
                            copy_paste_limit=self.copy_paste_limit,
                        )
                    )
                if self.train_include_sliding_dataset:
                    sub_datasets.append(
                        SlidingWindowDataset(
                            zarr_path,
                            window=self.valid_crop_size,
                            stride=tuple(s // 2 for s in self.valid_crop_size),
                            voxel_spacing=vx,
                            preload_volume=self.preload_volumes,
                        )
                    )
                ds = ConcatDataset(sub_datasets)
            else:
                ds = InstanceCropDataset(
                    zarr_path,
                    centers,
                    vx,
                    crop_size=crop_size,
                    num_crops=self.val_num_crops,
                    negative_ratio=0.0 if self.positive_only else 0.2,
                    preload_volume=self.preload_volumes,
                    use_gpu=use_gpu,
                    mixup_prob=0.0,
                    cutmix_prob=0.0,
                    copy_paste_prob=0.0,
                    copy_paste_limit=self.copy_paste_limit,
                )
            datasets.append(ds)

        if not datasets:
            raise ValueError(f"No valid tomograms for ids={list(ids)}")
        return ConcatDataset(datasets)

    # -------------------- Lightning hooks ------------------------------ #
    def setup(self, stage: str | None = None):
        spacing_map = voxel_spacing_map(self.root)
        centers_df = self._split_folds(read_train_centers(self.root))

        val_ids = centers_df.loc[centers_df["fold"] == self.fold, "tomo_id"].unique()
        train_ids = centers_df.loc[centers_df["fold"] != self.fold, "tomo_id"].unique()

        if stage in (None, "fit"):
            self.ds_train = self._make_crop_ds(
                train_ids, centers_df, spacing_map, training=True, use_gpu=self.use_gpu_augment
            )
            self.ds_val = self._make_crop_ds(
                val_ids, centers_df, spacing_map, training=False, use_gpu=self.valid_use_gpu_augment
            )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.nw,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_with_centers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_with_centers,
        )
