# BYU 모터 탐지

BYU 데이터셋의 모터 결함을 탐지하기 위한 프로젝트입니다.

## 설치

`environment.yml` 파일을 이용해 conda 환경을 생성합니다:

```bash
conda env create -f environment.yml
conda activate byu-motor
```

`pyproject.toml` 을 직접 사용하여 설치할 수도 있습니다:

```bash
pip install -e .
```

## 데이터 준비

Kaggle CLI로 BYU 모터 데이터셋을 다운로드한 뒤 `data/` 폴더 아래에 배치합니다. 디렉터리 구조는 다음과 같습니다:

```
<DATA_ROOT>/
  raw/
    train_labels.csv
    train/
    test/
  processed/
    zarr/
      train/<tomo_id>.zarr
      test/<tomo_id>.zarr
```

데이터 위치는 `--data_root` 옵션이나 환경 변수 `BYU_DATA_ROOT` 로 지정합니다. Windows에서 하위 폴더에서 실행할 경우에는 절대 경로를 사용해야 합니다.

## 학습

전체 학습을 수행하려면 다음 명령을 실행합니다:

```bash
python -m motor_det.engine.train \
  --data_root D:\project\Kaggle\BYU\byu-motor\data \
  --batch_size 2 --epochs 10
```

검증 단계에서 사용할 NMS 알고리즘은 `--nms_algorithm` 옵션으로 조절할 수 있습니다. 기본값인 `vectorized` 모드는 검출 수가 `--nms_switch_thr`를 초과하면 자동으로 `greedy` 모드로 전환됩니다.

`--cpu_augment` 플래그로 CPU에서 증강을 수행할 수 있으며, 이 경우 `--pin_memory` 옵션을 함께 주면 데이터 전송이 빨라집니다. `--num_workers`와 `--persistent_workers` 옵션을 활용하면 DataLoader 초기화 시간을 줄일 수 있습니다.

학습 로그와 체크포인트는 `runs/motor_fold<fold>` 디렉터리에 저장됩니다. 진행 상황은 다음과 같이 확인합니다:

```bash
tensorboard --logdir runs
```

### 빠른 테스트

모델을 간단히 확인하기 위해 작은 규모의 학습을 실행할 수 있습니다. `--max_steps`, `--limit_val_batches`, `--val_check_interval` 인자를 조정하여 학습과 검증 횟수를 줄일 수 있습니다.

```bash
python -m motor_det.engine.train \
  --data_root D:\project\Kaggle\BYU\byu-motor\data \
  --batch_size 1 \
  --max_steps 1500 \
  --limit_val_batches 0.1 \
  --val_check_interval 1500
```

## 추론

학습 완료 후 다음 명령으로 예측 결과를 생성할 수 있습니다:

```bash
python -m motor_det.engine.infer \
  --weights runs/motor_fold0/best.ckpt \
  --data_root data \
  --out_csv predictions.csv
```

`--batch`와 `--num_workers` 값으로 처리 속도를 조절할 수 있으며 GPU가 있으면 자동으로 사용됩니다.

