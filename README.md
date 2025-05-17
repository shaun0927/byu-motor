# BYU 모터 결함 탐지


BYU Motor 데이터셋을 이용해 Bacteria Flagellar 모터의 존재 유무와 좌표를 검출하는 프로젝트입니다.

## 설치

`environment.yml` 파일을 사용해 conda 환경을 구성할 수 있습니다:



```bash
conda env create -f environment.yml
conda activate byu-motor
```


또는 `pyproject.toml`을 이용해 바로 설치할 수도 있습니다:


```bash
pip install -e .
```

## 데이터 준비

Kaggle CLI를 사용하여 BYU Motor 데이터셋을 내려받은 뒤 `data/` 폴더 아래에 배치합니다. 디렉터리 구조 예시는 다음과 같습니다:

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



전체 학습 예시는 다음과 같습니다.

`--data_root` 인자(또는 `BYU_DATA_ROOT` 환경 변수)로 데이터 위치를 지정합니다. 하위 디렉터리에서 명령을 실행할 때는 Windows에서 경로가 올바르게 해석되도록 **절대 경로**를 사용하세요.

## 학습

전체 학습을 실행하려면 다음과 같이 명령을 입력합니다:



```bash
python -m motor_det.engine.train \
  --data_root D:\project\Kaggle\BYU\byu-motor\data \
  --batch_size 2 --epochs 10

```


검증 단계에서 사용할 NMS 방식은 `nms_algorithm` 옵션으로 결정하며 기본값인 `vectorized` 모드는 탐지 개수가 `--nms_switch_thr`를 넘으면 자동으로 `greedy`로 전환됩니다.

`--cpu_augment`를 사용하면 증강을 CPU에서 수행합니다. 이 경우 `--pin_memory`를 함께 지정하면 데이터 전송 속도를 높일 수 있습니다. 데이터 로더 초기화를 줄이고 싶다면 `--persistent_workers` 플래그를 켜고 `--num_workers` 값도 조정하세요.

학습 로그와 체크포인트는 `runs/motor_fold<fold>` 하위에 저장됩니다. 진행 상황은 다음과 같이 확인합니다.

`persistent_workers=True`로 설정하면 각 에폭마다 데이터로더 작업자를 재생성하지 않아 속도가 향상됩니다. 본 스크립트는 `--persistent_workers` 플래그로 이 기능을 켤 수 있습니다.

학습 로그와 체크포인트는 `runs/motor_fold<fold>` 아래에 저장되며 TensorBoard로 모니터링할 수 있습니다:


```bash
tensorboard --logdir runs
```

## 추론

학습 후 예측은 다음과 같이 생성합니다.

### 간단한 검증 실행

모델 동작을 빠르게 확인하려면 짧은 학습을 수행할 수 있습니다. `--max_steps`, `--limit_val_batches`, `--val_check_interval` 값은 필요에 따라 조절하세요:

```bash
python -m motor_det.engine.train \
  --data_root D:\\project\\Kaggle\\BYU\\byu-motor\\data \
  --batch_size 1 \
  --max_steps 1500 \
  --limit_val_batches 0.1 \
  --val_check_interval 1500 \
  --persistent_workers


```

위 예시는 약 1500 스텝 동안 학습하며 검증 세트의 10%만 사용해 성능을 빠르게 확인합니다.

## 추론

학습 후에는 다음과 같이 예측을 생성할 수 있습니다:



```bash
python -m motor_det.engine.infer \
  --weights runs/motor_fold0/best.ckpt \
  --data_root data \
  --out_csv predictions.csv
```

`--batch`와 `--num_workers` 인자로 추론 속도를 조절할 수 있으며, 가능한 경우 GPU가 자동으로 사용됩니다.

간단한 실험을 위해 `quick_train_val.py`와 `motor_det/tests/test_quick_train.py` 스크립트를 참고할 수 있습니다.


다음은 라이브러리 함수를 직접 사용해 추론을 수행하는 예시입니다.

`--batch`와 `--num_workers` 인자로 추론 속도를 조절할 수 있으며, 가능할 경우 자동으로 GPU를 사용합니다.

`quick_train_val.py`나 `motor_det/tests/test_quick_train.py` 등의 스크립트는 소규모 실험을 재현하기 위해 제공됩니다.

CPU 기반 증강을 사용할 경우 `--pin_memory` 옵션을 적용하면 유용합니다. 기본적으로는 CUDA 증강을 사용하므로 해당 모드를 끄려면 먼저 `--cpu_augment`를 지정해야 합니다.

## 추가 추론 예시

`run_inference` 함수를 이용하면 체크포인트를 불러와 특정 폴더의 토모그램에 대해 모터 위치를 예측할 수 있습니다:


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
    cfg=InferConfig(),
)
```

동일한 설정을 CLI로도 사용할 수 있습니다:

```bash
python -m motor_det.engine.infer \
    --weights weights/best.ckpt \
    --data_root data \
    --out_csv preds.csv
```

