# Topological Autoencoder (TAE)

이 디렉토리는 TCGA-BRCA RNA-seq 전처리 데이터를 기반으로 잠재 공간에 위상 구조를 보존하며 데이터를 압축하는 TAE 모델의 핵심 뼈대입니다.

## 파일 구조
- `models/model.py`: TAE 네트워크 구조 (Encoder/Decoder)
- `models/loss.py`: Topological Loss — Euclidean distance 기반 (Moor et al.의 Distance Matrix 매칭 아이디어 구현)
- `models/loss_alternative.py`: 대안 Topological Loss — Pearson correlation distance, Cosine distance 기반
- `training/train.py`: 학습 루프 (distance metric 선택 지원)
- `training/latent_vis.py`: Latent 추출, Borderline-SMOTE, UMAP 시각화
- `training/visualization.py`: 학습 곡선 및 성능 비교 시각화

## 실험 가이드 (황도현 님, 황선준 님)
1. **`topo_weight` 하이퍼파라미터 조율:** 위상 손실 페널티가 너무 크면 데이터 압축(Reconstruction) 자체를 망칠 수 있습니다. 학습 로그를 보며 `0.1` ~ `1.0` 사이에서 적절히 조율하세요.
2. **미니배치 크기(Batch Size):** TDA 로스는 '배치 안의 데이터들 간의 거리'를 계산합니다. 배치가 너무 작으면 위상 구조를 그리기 위한 정보가 부족해지므로, 64 이상의 넉넉한 Batch Size 사용을 권장합니다.
3. **거리 메트릭 선택:** 고차원 유전자 발현 데이터에서 Euclidean 거리는 curse of dimensionality로 인해 위상 구조를 제대로 포착하지 못할 수 있습니다. Pearson correlation distance(패턴 유사도)나 Cosine distance(방향 유사도)를 `--distance-metric` 옵션으로 선택하여 비교 실험하세요.

## 사용법 (CLI)
프로젝트 루트에서 실행:
```bash
# 기본값 (latent_dim=16, epochs=100, euclidean)
python TAE/training/train.py

# 차원 및 거리 메트릭 지정
python TAE/training/train.py --dimension 32 --distance-metric pearson --output TAE/models/tae_dim32_pearson.pth

# 전체 옵션
python TAE/training/train.py --dimension 64 --epochs 200 --batch-size 128 --topo-weight 1.0 --distance-metric cosine --output TAE/models/tae_dim64_cosine.pth
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--dimension` | Latent space 차원 | 16 |
| `--epochs` | 학습 에폭 수 | 100 |
| `--batch-size` | 미니배치 크기 | 64 |
| `--topo-weight` | Topological loss 가중치 | 0.5 |
| `--distance-metric` | 거리 메트릭 (`euclidean`, `pearson`, `cosine`) | `euclidean` |
| `--output` | 모델 저장 경로 | `TAE/models/tae_dim16.pth` |

## 전체 파이프라인 실행
프로젝트 루트에서 한 번에 전처리부터 시각화까지 실행:
```bash
bash run_pipeline.sh
```
이 스크립트는 다음을 순차적으로 수행합니다:
1. BRCA 환자 필터링 + GPU ComBat 전처리
2. 9개 모델 학습 (3 dims x 3 metrics)
3. Latent 추출 + Borderline-SMOTE + UMAP 시각화
4. 학습 곡선 및 성능 비교 시각화

## Latent 추출 & SMOTE & UMAP
```bash
# 전체 메트릭
python TAE/training/latent_vis.py

# 특정 메트릭만
python TAE/training/latent_vis.py --metric pearson cosine
```

## 학습 시각화
```bash
python TAE/training/visualization.py
```
모든 (차원 x 메트릭) 조합에 대한 learning curves, classifier 성능 비교, validation loss 비교 차트를 생성합니다.

## Python API 사용 예시
```python
import pandas as pd
import torch
from TAE.training.train import train_tae

df = pd.read_csv('data_preprocessing/cleaned_tcga_tpm_for_TAE.csv', index_col=0)
X = df.drop(columns=['Target']).values
y = df['Target'].values
X_tensor = torch.tensor(X, dtype=torch.float32)

# Pearson distance metric으로 학습
trained_model, history = train_tae(X_tensor, input_dim=X.shape[1], latent_dim=16,
                                   epochs=100, topo_weight=0.5, labels=y,
                                   distance_metric='pearson')
```
