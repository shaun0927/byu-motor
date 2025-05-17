import lightning as L
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from motor_det.data.dataset import (
    MotorTrainDataset,
    PositiveOnlyCropDataset,
)
from motor_det.utils.collate import collate_with_centers
from motor_det.utils.voxel import voxel_spacing_map, read_train_centers
from sklearn.model_selection import StratifiedGroupKFold


class MotorDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        fold: int,
        batch_size: int = 2,
        num_workers: int = 4,
        persistent_workers: bool = False,
        positive_only: bool = False,
    ):
        super().__init__()
        self.root = Path(data_root)
        self.fold = fold
        self.bs = batch_size
        self.nw = num_workers
        self.persistent_workers = persistent_workers
        self.positive_only = bool(positive_only)

    def setup(self, stage=None):
        # spacing map 과 train centers 데이터프레임 읽기
        spacing_map = voxel_spacing_map(self.root)
        centers_df = read_train_centers(self.root)

        # fold 정보가 없으면 StratifiedGroupKFold 로 자동 생성
        if centers_df["fold"].nunique() <= 1:
            centers_df["motor_present"] = (centers_df["Motor axis 0"] >= 0).astype(int)
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            fold_arr = np.full(len(centers_df), -1, dtype=int)
            for f, (_, val_idx) in enumerate(
                sgkf.split(
                    X=centers_df,
                    y=centers_df["motor_present"],
                    groups=centers_df["tomo_id"],
                )
            ):
                fold_arr[val_idx] = f
            centers_df["fold"] = fold_arr

        # train / val tomo_id 리스트 분리
        val_ids = centers_df.loc[centers_df["fold"] == self.fold, "tomo_id"].unique()
        train_ids = centers_df.loc[centers_df["fold"] != self.fold, "tomo_id"].unique()

        # 데이터셋 생성
        self.ds_train = self._build_ds(train_ids, centers_df, spacing_map, training=True)
        self.ds_val   = self._build_ds(val_ids,   centers_df, spacing_map, training=False)

    def _build_ds(self, ids, centers_df, spacing_map, training=True):
        datasets = []
        for tid in ids:
            # 빈 문자열이나 None 은 건너뜁니다
            if not tid:
                continue
            # zarr 경로 결정 (train 셋은 모두 processed/zarr/train)
            zarr_path = self.root / "processed" / "zarr" / "train" / f"{tid}.zarr"
            # 실제 파일이 존재하는지 확인
            if not zarr_path.exists():
                continue
            # 이 tomo 에 해당하는 GT 좌표 (X, Y, Z 순으로 정렬)
            sub_df = centers_df[centers_df["tomo_id"] == tid]
            # `Motor axis 0` 이 음수이면 해당 tomogram 에 motor 가 없음
            sub_df = sub_df[sub_df["Motor axis 0"] >= 0]

            centers = sub_df[["Motor axis 2", "Motor axis 1", "Motor axis 0"]].values.astype(np.float32)
            vx = spacing_map.get(tid, 15.0)

            if self.positive_only and len(centers) == 0:
                # skip tomograms without GT when using positive-only crops
                continue

            if self.positive_only:
                ds = PositiveOnlyCropDataset(zarr_path, centers, vx)
            else:
                ds = MotorTrainDataset(zarr_path, centers, vx)

            datasets.append(ds)

        if len(datasets) == 0:
            raise ValueError(f"Fold {self.fold}에 사용 가능한 Zarr 데이터가 없습니다: ids={ids}")
        return ConcatDataset(datasets)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.nw,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=collate_with_centers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=collate_with_centers,
        )
