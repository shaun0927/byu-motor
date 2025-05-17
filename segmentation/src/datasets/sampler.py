# =====================================================================
#  sampler.py – positive-prior (oversampling) sampler
# =====================================================================
from __future__ import annotations
import itertools, math, random
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler

__all__ = ["PosBgSampler"]


# ---------------------------------------------------------------------
# util – 메타만 보고 pos / bg 풀 만들기 (I/O 0)
# ---------------------------------------------------------------------
def _build_index_pools(dataset) -> tuple[List[int], List[int]]:
    """
    MotorDataset 처럼 `dataset.items` 리스트에
    ▸ volume 단위 메타(dict)와  
    ▸ `sub_ep`(volume 당 sub-epochs) 값이 있는 경우만 지원.
    """
    if not hasattr(dataset, "items"):
        raise RuntimeError(
            "`PosBgSampler` 는 `dataset.items` 메타가 있는 Dataset 에서만 "
            "사용할 수 있습니다."
        )

    pos, neg = [], []
    n_items  = len(dataset.items)
    sub_ep   = getattr(dataset, "sub_ep", 1)

    for item_idx, meta in enumerate(dataset.items):
        target = pos if meta["motors"].size else neg
        if sub_ep == 1:
            target.append(item_idx)
        else:
            # 같은 volume 에 대해 sub_ep 만큼 연달아 index 가 배정됨
            base = item_idx
            target.extend(base + i * n_items for i in range(sub_ep))

    if not pos:
        raise RuntimeError("Positive-patch pool이 비어 있습니다.")

    return pos, neg


# ---------------------------------------------------------------------
# main Sampler
# ---------------------------------------------------------------------
class PosBgSampler(Sampler[int]):
    """
    한 epoch 을  `pos_ratio` : `(1-pos_ratio)` 비율로
    **양성 / 배경** 패치로 구성해 주는 oversampling Sampler.
    """

    def __init__(
        self,
        dataset,
        *,
        pos_ratio: float = 0.8,
        shuffle: bool = True,
        epoch_size: Optional[int] = None,
        seed: int = 0,
    ):
        if not 0.0 < pos_ratio < 1.0:
            raise ValueError("pos_ratio must be in (0, 1).")

        self.dataset    = dataset
        self.pos_ratio  = pos_ratio
        self.shuffle    = shuffle
        self.rng        = random.Random(seed)
        self.epoch_size = epoch_size or len(dataset)

        # I/O 없이 인덱스 풀 생성
        self.pos_pool, self.neg_pool = _build_index_pools(dataset)

    # -----------------------------------------------------------------
    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            self.rng.shuffle(self.pos_pool)
            self.rng.shuffle(self.neg_pool)

        n_pos = math.ceil(self.epoch_size * self.pos_ratio)
        n_neg = self.epoch_size - n_pos

        pos_iter = itertools.islice(itertools.cycle(self.pos_pool), n_pos)
        neg_iter = itertools.islice(itertools.cycle(self.neg_pool), n_neg)

        # interleave → batch 내부 클래스 비율 안정
        mixed = list(itertools.chain.from_iterable(zip(pos_iter, neg_iter)))
        # zip 이 짧은 쪽에서 끝나면 나머지 이어붙이기
        if len(mixed) < self.epoch_size:
            rest = (list(pos_iter) + list(neg_iter))[: self.epoch_size - len(mixed)]
            mixed.extend(rest)

        return iter(mixed[: self.epoch_size])

    # -----------------------------------------------------------------
    def __len__(self) -> int:     # type: ignore[override]
        return self.epoch_size
