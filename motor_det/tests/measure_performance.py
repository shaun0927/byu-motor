import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from motor_det.data.module import MotorDataModule
from motor_det.engine.lit_module import LitMotorDet
from motor_det.metrics.det_metric import fbeta_score
from motor_det.postprocess.decoder import decode_with_nms


def evaluate(weights: Path, data_root: Path, fold: int = 0) -> dict[str, float]:
    dm = MotorDataModule(
        data_root=str(data_root),
        fold=fold,
        batch_size=1,
        num_workers=12,
        persistent_workers=True,
        use_gpu_augment=False,
        valid_use_gpu_augment=False,
    )
    dm.setup()

    model = LitMotorDet.load_from_checkpoint(str(weights))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    f2_scores = []
    prec_scores = []
    rec_scores = []
    with torch.no_grad():
        for batch in tqdm(dm.val_dataloader(), desc="Validate"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(batch["image"])
            centers = decode_with_nms(
                out["cls"],
                out["offset"],
                stride=2,
                prob_thr=0.6,
                sigma=60.0,
                iou_thr=0.25,
            )[0]
            spacing = float(batch["spacing_Å_per_voxel"][0])
            centers = centers * spacing
            gt = batch["centers_Å"][0].to(device)
            f2, prec, rec, *_ = fbeta_score(centers, gt, beta=2, dist_thr=1000.0)
            f2_scores.append(f2)
            prec_scores.append(prec)
            rec_scores.append(rec)

    mean = lambda x: float(torch.tensor(x).mean())
    return {
        "f2": mean(f2_scores),
        "precision": mean(prec_scores),
        "recall": mean(rec_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    metrics = evaluate(Path(args.weights), Path(args.data_root), args.fold)
    print(
        f"F2: {metrics['f2']:.4f}  "
        f"Precision: {metrics['precision']:.4f}  "
        f"Recall: {metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
