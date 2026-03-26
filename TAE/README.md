# Topological Autoencoder (TAE)

이 디렉토리는 RNA-seq 전처리 데이터를 기반으로 잠재 공간에 위상 구조를 보존하며 데이터를 압축하는 TAE 모델의 핵심 뼈대입니다.

## 파일 구조
- `models/model.py`: TAE 네트워크 구조 (Encoder/Decoder)
- `models/loss.py`: Topological Loss (Moor et al.의 Distance Matrix 매칭 아이디어 구현)
- `training/train.py`: 학습 루프 뼈대

## 실험 가이드 (황도현 님, 황선준 님)
1. **`topo_weight` 하이퍼파라미터 조율:** 위상 손실 페널티가 너무 크면 데이터 압축(Reconstruction) 자체를 망칠 수 있습니다. 학습 로그를 보며 `0.1` ~ `1.0` 사이에서 적절히 조율하세요.
2. **미니배치 크기(Batch Size):** TDA 로스는 '배치 안의 데이터들 간의 거리'를 계산합니다. 배치가 너무 작으면 위상 구조를 그리기 위한 정보가 부족해지므로, 64 이상의 넉넉한 Batch Size 사용을 권장합니다.

## 사용법 (CLI)
프로젝트 루트에서 실행:
```bash
# 기본값 (latent_dim=16, epochs=100)
python TAE/training/train.py

# 차원 및 저장 경로 지정
python TAE/training/train.py --dimension 32 --output TAE/tae_dim32.pth

# 전체 옵션
python TAE/training/train.py --dimension 8 --epochs 200 --batch-size 128 --topo-weight 1.0
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--dimension` | Latent space 차원 | 16 |
| `--epochs` | 학습 에폭 수 | 100 |
| `--batch-size` | 미니배치 크기 | 64 |
| `--topo-weight` | Topological loss 가중치 | 0.5 |
| `--output` | 모델 저장 경로 | `TAE/tae_trained.pth` |

## Python API 사용 예시
```python
import pandas as pd
import torch
from TAE.training.train import train_tae

df = pd.read_csv('data_preprocessing/cleaned_tcga_tpm_for_TAE.csv', index_col=0)
X = df.drop(columns=['Target']).values
y = df['Target'].values
X_tensor = torch.tensor(X, dtype=torch.float32)

# labels를 전달하면 stratified train/val split + 분류 성능 지표 자동 계산
trained_model, history = train_tae(X_tensor, input_dim=X.shape[1], latent_dim=16,
                                   epochs=100, topo_weight=0.5, labels=y)
```