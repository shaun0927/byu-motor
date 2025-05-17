# BYU 모터 결함 탐지

BYU Motor 데이터셋을 이용해 모터의 결함을 검출하는 프로젝트입니다.

## 환경 설정

`environment.yml` 파일을 사용해 conda 환경을 만들 수 있습니다:

```bash
conda env create -f environment.yml
conda activate byu-motor
```

또는 `pyproject.toml`을 통해 설치할 수도 있습니다:

```bash
pip install -e .
```

## 데이터 준비

Kaggle CLI를 통해 BYU Motor 데이터를 다운로드한 뒤 `data/` 폴더 아래에 배치합니다. 디렉터리 구조는 다음과 같습니다.

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

데이터 위치는 `--data_root` 인자(또는 `BYU_DATA_ROOT` 환경 변수)로 지정합니다. 하위 디렉터리에서 실행할 경우 Windows 환경에서는 절대 경로를 사용해 주세요.

## 학습

전체 학습 예시는 다음과 같습니다.

```bash
python -m motor_det.engine.train \
  --data_root D:\project\Kaggle\BYU\byu-motor\data \
  --batch_size 2 --epochs 10
```

`nms_algorithm` 옵션은 검증 시 사용되는 NMS 방식을 결정합니다. 기본값인 `vectorized` 모드는 검출 수가 `--nms_switch_thr` 값을 넘으면 자동으로 `greedy` 로 변경됩니다.

`--cpu_augment`를 사용하면 증강을 CPU에서 수행합니다. 이 경우 `--pin_memory`를 함께 지정하면 데이터 전송 속도를 높일 수 있습니다. 데이터 로더 초기화를 줄이고 싶다면 `--persistent_workers` 플래그를 켜고 `--num_workers` 값도 조정하세요.

학습 로그와 체크포인트는 `runs/motor_fold<fold>` 하위에 저장됩니다. 진행 상황은 다음과 같이 확인합니다.

```bash
tensorboard --logdir runs
```

### 간단한 품질 확인

짧은 학습으로 모델 성능을 빠르게 점검할 수 있습니다.

```bash
python -m motor_det.engine.train \
  --data_root D:\project\Kaggle\BYU\byu-motor\data \
  --batch_size 1 \
  --max_steps 1500 \
  --limit_val_batches 0.1 \
  --val_check_interval 1500
```

`--max_steps`, `--limit_val_batches`, `--val_check_interval` 값을 조절해 짧은 실험을 수행할 수 있습니다. 검증 속도가 느리다면 `valid_use_gpu_augment=False` 로 설정하여 GPU 증강을 끄는 것이 도움이 됩니다.

## 추론

학습 후 예측은 다음과 같이 생성합니다.

```bash
python -m motor_det.engine.infer \
  --weights runs/motor_fold0/best.ckpt \
  --data_root data \
  --out_csv predictions.csv
```

`--batch`와 `--num_workers` 인자로 추론 속도를 조절할 수 있으며, 가능한 경우 GPU가 자동으로 사용됩니다.

간단한 실험을 위해 `quick_train_val.py`와 `motor_det/tests/test_quick_train.py` 스크립트를 참고할 수 있습니다.

---

다음은 라이브러리 함수를 직접 사용해 추론을 수행하는 예시입니다.

```python
from motor_det.engine.infer import run_inference, InferConfig

run_inference(
    "weights/best.ckpt",
    data_root="data",
    out_csv="preds.csv",
    cfg=InferConfig(),  # 필요하면 설정 수정
)
```

동일한 기능은 CLI에서도 다음과 같이 사용할 수 있습니다.

```bash
python -m motor_det.engine.infer \
    --weights weights/best.ckpt \
    --data_root data \
    --out_csv preds.csv
```
