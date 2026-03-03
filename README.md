# i3d-shoplifting

Subpacote para fine-tuning do modelo **I3D (Inflated 3D ConvNet)** como classificador binário de **Shoplifting** (vs. Normal) a partir de blocos de 64 frames e, opcionalmente, **fluxo ótico** (late fusion RGB+Flow).

---

## 1) Overview

### O que este subpacote faz

- Fornece módulos de **preprocessamento** que convertem datasets públicos (clipes/vídeos) em **blocos de evento** (frames extraídos com FFmpeg) e os transformam em inputs do I3D:
  - **RGB**: 64 frames reamostrados e recortados para 224×224
  - **Optical Flow**: TV-L1 denso, armazenado como `flow_x_*.jpg` e `flow_y_*.jpg`
- Implementa a **arquitetura I3D** (Inception V1 inflada para 3D) em PyTorch puro.
- Treina (fine-tune) o I3D para **classificação binária** com `BCEWithLogitsLoss` e `pos_weight` (para lidar com desbalanceamento de classes).

### Estrutura do pacote

```
src/i3d_shoplifting/
├── dataset/
│   └── dataset.py          # Dataset PyTorch (RGB + Optical Flow)
├── inference/               # (reservado para uso futuro)
├── models/
│   └── i3d_pytorch.py       # Arquitetura I3D completa
├── preprocessing/
│   ├── extract_DSCASS.py    # Extração de frames do DCSASS
│   ├── extract_others.py    # Extração de frames de datasets simples (MNNIT, Dataset 2.0)
│   ├── gen_optical_flow.py  # Geração de fluxo ótico TV-L1
│   └── sample_event_blocks_i3d.py  # Amostragem de 64 frames + resize
└── training/
    └── train.py             # Fine-tuning do I3D
```

### Estrutura de dados (padrão)

- `data/event_blocks_frames/` → blocos brutos (frames extraídos de clipes/vídeos)
- `data/i3d_inputs/rgb/` → blocos RGB prontos para o I3D (64 frames 224×224)
- `data/i3d_inputs/optical_flow/` → blocos de fluxo ótico correspondentes
- `checkpoints/pretrained/` → pesos pré-treinados ImageNet (RGB e Flow)
- `checkpoints/experiments_i3d/` → resultados de treino (logs/curvas/weights)

> **Nota:** quando executado via scripts centralizados (raiz do monorepo), os caminhos de dados são controlados pelo `scripts/config.yaml`.

---

## 2) Preprocessamento

O preprocessamento converte fontes heterogêneas (clipes .mp4 em diferentes resoluções e durações) em entradas consistentes para o I3D.

> **Recomendado:** use o script centralizado `scripts/preprocess_i3d.py` na raiz do monorepo, que orquestra todas as etapas abaixo com caminhos configuráveis via `config.yaml`.

### 2.1 Extração de frames dos datasets

#### A) DCSASS (DCSASS_Dataset)

Script: `src/i3d_shoplifting/preprocessing/extract_DSCASS.py`

O DCSASS contém:

- `situations/` com múltiplas "situações", cada uma com clipes `.mp4` curtos e numerados
- `Shoplifting.csv` com anotações por clipe (0/1)

O script:

1. Lê as anotações (`Shoplifting.csv`).
2. Percorre cada situação e identifica **blocos contíguos** de clipes com o mesmo rótulo.
3. Para blocos positivos (Shoplifting), adiciona **contexto** (clipe anterior e posterior, quando existir).
4. Concatena os clipes de cada bloco com FFmpeg, reamostra para `25 FPS` e extrai frames.

**Argumentos CLI:**

| Argumento | Descrição |
|---|---|
| `--dataset-root` | Diretório raiz do dataset DCSASS |
| `--annotations-path` | Caminho para o `Shoplifting.csv` |
| `--output-dir` | Diretório de saída dos blocos de frames |
| `--config` | Caminho para `config.yaml` (alternativa aos args acima) |

#### B) MNNIT / Shoplifting Dataset 2.0

Script: `src/i3d_shoplifting/preprocessing/extract_others.py`

Trata diretórios com vídeos curtos individuais como **um bloco por vídeo**, extraindo frames com FFmpeg. Continua a numeração sequencial existente no diretório de saída.

**Argumentos CLI:**

| Argumento | Descrição |
|---|---|
| `--input-dir` | Diretório com os vídeos de entrada |
| `--output-dir` | Diretório de saída dos blocos de frames |
| `--dataset-type` | Prefixo da classe: `Normal` ou `Shoplifting` |
| `--config` | Caminho para `config.yaml` |

### 2.2 Converter blocos em inputs I3D

Depois de gerar `event_blocks_frames/<Classe>_<id>/frame_%06d.jpg`, os inputs de treino são criados em duas etapas:

#### Passo 1 — RGB (64 frames em 224×224)

Script: `src/i3d_shoplifting/preprocessing/sample_event_blocks_i3d.py`

- Amostragem por segmentos para obter **exatamente 64 frames** por bloco
- Resize do lado menor para 256 + center crop para 224×224

**Argumentos CLI:**

| Argumento | Descrição |
|---|---|
| `--source-blocks-dir` | Diretório com os blocos de frames brutos |
| `--output-rgb-dir` | Diretório de saída dos inputs RGB |
| `--num-frames` | Número de frames a amostrar (padrão: 64) |
| `--overwrite` | Sobrescrever blocos já processados |
| `--config` | Caminho para `config.yaml` |

#### Passo 2 — Optical Flow (TV-L1)

Script: `src/i3d_shoplifting/preprocessing/gen_optical_flow.py`

Calcula fluxo ótico denso TV-L1 entre frames consecutivos (u,v), normaliza para 0..255 (clip ±20) e salva `flow_x_%06d.jpg` e `flow_y_%06d.jpg`.

**Argumentos CLI:**

| Argumento | Descrição |
|---|---|
| `--source-rgb-dir` | Diretório com os inputs RGB processados |
| `--output-flow-dir` | Diretório de saída do fluxo ótico |
| `--overwrite` | Sobrescrever blocos já processados |

**Por que o fluxo ótico?**
RGB captura aparência; fluxo captura movimento. Em eventos como shoplifting, motion cues podem ser determinantes.

---

## 3) Modelo

### Arquitetura (`models/i3d_pytorch.py`)

Implementação da rede **Inception I3D** em PyTorch puro:

- `Unit3D` — Conv3D + BatchNorm3D + ReLU
- `InceptionModule` — Módulo Inception adaptado para 3D (4 branches)
- `InceptionI3d` — Rede completa. Saída: 1 logit (classificação binária com `BCEWithLogitsLoss`). Suporta `in_channels=3` (RGB) ou `in_channels=2` (fluxo ótico).

### Dataset (`dataset/dataset.py`)

- `VideoAugmentation` — Horizontal flip (50%) + color jitter. Inverte automaticamente o canal x do fluxo ótico no flip.
- `ShopliftingDataset` — Retorna `(rgb_before, flow_before, rgb_after, flow_after, label)`. Label inferido pelo nome da pasta.

---

## 4) Treinamento

Script: `src/i3d_shoplifting/training/train.py`

### O que o treino faz

- Carrega inputs RGB e (se `rgb_optical`) fluxo ótico
- Split estratificado: 70% treino, 15% validação, 15% teste
- `BCEWithLogitsLoss(pos_weight=...)` para balanceamento de classes
- Data augmentation no RGB (flip + color jitter)
- Salva:
  - Top-3 checkpoints por AUC-ROC em `<output-dir>/<run>/model_weights/`
  - Métricas em CSV
  - Curvas ROC por época (npz)
  - Vídeos de visualização (before/after augmentation)

### Argumentos CLI

| Argumento | Default | Descrição |
|---|---|---|
| `--model-mode` | `rgb_optical` | `rgb_optical` (late fusion) ou `rgb_only` |
| `--unfreeze-full-model` | `False` | Fine-tuning completo (descongela todas as camadas) |
| `--epochs` | `70` | Número de épocas |
| `--batch-size` | `1` | Batch size |
| `--learning-rate` | `1e-3` | Learning rate |
| `--seed` | `42` | Seed para reprodutibilidade |
| `--rgb-dir` | `data/i3d_inputs/rgb` | Diretório com inputs RGB |
| `--flow-dir` | `data/i3d_inputs/optical_flow` | Diretório com inputs de fluxo ótico |
| `--rgb-checkpoint` | `checkpoints/pretrained/rgb_imagenet.pt` | Checkpoint pré-treinado RGB |
| `--flow-checkpoint` | `checkpoints/pretrained/flow_imagenet.pt` | Checkpoint pré-treinado Flow |
| `--output-dir` | `checkpoints/experiments_i3d` | Diretório de saída dos experimentos |
| `--steps-to-visualize-per-epoch` | `2` | Batches por época para salvar em vídeo |

### Exemplos

RGB + Optical Flow (late fusion):

```bash
uv run python src/i3d_shoplifting/training/train.py \
    --model-mode rgb_optical \
    --unfreeze-full-model \
    --epochs 70 \
    --batch-size 1
```

Apenas RGB:

```bash
uv run python src/i3d_shoplifting/training/train.py \
    --model-mode rgb_only \
    --unfreeze-full-model \
    --epochs 70 \
    --batch-size 1
```

### Dependências

```bash
uv sync --all-extras
```

- **Runtime:** `torch`, `torchvision`, `numpy`, `opencv-contrib-python`, `natsort`
- **Dev (treino):** `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tqdm`
- **Sistema:** `ffmpeg` (necessário para os scripts de extração)

---

## Notas importantes

- **Labels** são inferidos pelo nome da pasta: `Shoplifting` no nome → label=1, caso contrário → label=0.
- Para o modo `rgb_optical`, o checkpoint salvo contém `model_flow_state_dict` além do RGB.
- Os scripts de preprocessamento podem ser executados individualmente (com args CLI) ou via o pipeline centralizado `scripts/preprocess_i3d.py` na raiz do monorepo.
