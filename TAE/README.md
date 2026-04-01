# Topological Autoencoder (TAE) 프로젝트

TCGA-BRCA 유전자 발현 데이터를 잠재 공간(Latent Space)으로 압축하면서도, 데이터의 고유한 위상 구조(Topological Structure)를 보존하기 위한 모델입니다.

## 주요 파일
- `models/`: 모델 아키텍처 및 다양한 손실 함수 (Sinkhorn OT 포함)
- `training/train.py`: 모델 학습 메인 스크립트
- `training/latent_vis.py`: Latent 추출 및 SMOTE 증강, UMAP 시각화
- `training/visualization.py`: 학습 결과 차트 생성
- `run_pipeline.sh`: 전처리부터 시각화까지 한 번에 실행하는 스크립트

## 핵심 기능 (Sinkhorn Topo Loss)
기존의 MSE 기반 위상 손실 함수 대신 **Optimal Transport (Sinkhorn)** 기반의 로스를 추가했습니다. 
- **메모리 최적화**: Envelope Theorem을 적용해 VRAM 사용량을 대폭 줄였습니다.
- **수치적 안정성**: Log-domain 연산을 통해 작은 epsilon에서도 안정적으로 학습됩니다.
- **Debiased Divergence**: 로스 값이 음수로 떨어지지 않도록 보정하여 Adaptive Weighting과 잘 맞물리게 설계했습니다.

## 실행 방법

### 1. 전체 파이프라인
```bash
# 전과정 자동 실행
bash run_pipeline.sh

# 전처리는 건너뛰고 학습/시각화만 실행
bash run_pipeline.sh --skip-preprocess
```

### 2. 개별 모델 학습
```bash
# 기본 학습
python TAE/training/train.py --dimension 16 --distance-metric cosine

# Sinkhorn OT 로스 사용
python TAE/training/train.py --sinkhorn --topo-multiplier 1000.0
```

### 3. 결과 확인
학습이 끝나면 `TAE/results/` 폴더에 다음 파일들이 생성됩니다:
- `clf_..._cmp.png`: 분류 성능 비교 (Accuracy, AUC 등)
- `best_val_..._cmp.png`: 최적 손실 값 비교
- `umap_..._mXXX.png`: 설정별 UMAP 시각화 결과
- `wSMOTE/`: SMOTE 증강된 데이터 (CSV)

## 실험 팁
- **Batch Size**: 위상 구조를 잡으려면 데이터 간의 관계 정보가 많이 필요합니다. 최소 64 이상의 배치를 권장합니다.
- **Topo Multiplier**: Sinkhorn 로스는 MSE에 비해 값이 매우 작습니다. 보통 `10^2` ~ `10^4` 정도의 multiplier를 주어야 학습이 제대로 일어납니다.
- **Metric**: 유전자 데이터 특성상 `cosine`이나 `pearson` 거리가 `euclidean`보다 구조를 더 잘 포착하는 경향이 있습니다.
