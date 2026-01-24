# i3d-shoplifting

Projeto para fine-tuning do modelo **I3D (Inflated 3D ConvNet)** como classificador binário de **Shoplifting** (vs. Normal) a partir de clips curtos (64 frames) e, opcionalmente, **fluxo ótico** (late fusion RGB+Flow).

---

## 1) Overview

### O que este repositório faz

- Converte datasets públicos (clipes/vídeos) em **blocos de evento** baseados em rótulos (Shoplifting/Normal), extraindo frames com **FFmpeg**.
- Transforma esses blocos em **inputs I3D**:
  - RGB: 64 frames reamostrados/recortados para 224×224
  - Optical Flow: TV-L1 denso, armazenado como imagens `flow_x_*.jpg` e `flow_y_*.jpg`
- Treina (fine-tune) um I3D para **classificação binária** com `BCEWithLogitsLoss` e `pos_weight` (para lidar com desbalanceamento).

### Estrutura de pastas (visão rápida)

- `data/event_blocks_frames/` -> blocos brutos (frames extraídos de clipes/vídeos)
- `data/i3d_inputs/rgb/` -> blocos RGB já no formato do I3D (64 frames 224×224)
- `data/i3d_inputs/optical_flow/` -> blocos de fluxo ótico correspondentes
- `checkpoints/pretrained/` -> pesos ImageNet (RGB e Flow)
- `checkpoints/experiments_i3d/` -> resultados de treino (logs/curvas/weights)

---

## 2) Data preprocess

A ideia do preprocessamento é converter fontes heterogêneas (clipes .mp4 em diferentes resoluções e durações) em entradas consistentes para o I3D.

### 2.1 Datasets públicos suportados pelos scripts

#### A) DCSASS (DCSASS_Dataset)

Script: `src/preprocessing/extract_DSCASS.py`

O DCSASS costuma vir com:

- `situations/` contendo múltiplas “situações”, cada uma com clipes `.mp4` curtos e numerados
- `Shoplifting.csv` com anotações por clipe (0/1)

O script:

1. Lê as anotações (`Shoplifting.csv`).
2. Percorre cada situação e identifica **blocos contíguos** de clipes com o mesmo rótulo.
3. Para blocos positivos (Shoplifting), adiciona **contexto** (clipe anterior e posterior, quando existir).
4. Concatena os clipes de cada bloco com FFmpeg, reamostra para `25 FPS` e extrai frames para pastas:

- `Normal_0/`, `Normal_1/`, ...
- `Shoplifting_0/`, `Shoplifting_1/`, ...

**Por que isso existe?**
- O “evento” shoplifting pode começar/terminar perto dos limites dos clipes; adicionar contexto ajuda o modelo a capturar o padrão temporal completo.

Exemplo (ajuste os caminhos conforme seu disco):

```bash
uv run python src/preprocessing/extract_DSCASS.py
```

> Observação: o script atualmente tem caminhos hardcoded no final do arquivo. Edite-os conforme seu dataset.

#### B) Shoplifting Dataset 2.0 (vídeos curtos)

Script: `src/preprocessing/extract_others.py`

Esse script trata um diretório com vídeos curtos (ex.: “Shoplifting Dataset 2.0”) como **um bloco por vídeo**, extraindo frames com FFmpeg e adicionando na sequência numérica existente.

- Você escolhe o prefixo com `DATASET_TYPE` (`Shoplifting` ou `Normal`).

Exemplo:

```bash
uv run python src/preprocessing/extract_others.py
```

> Observação: também usa caminhos hardcoded no final do arquivo.

### 2.2 Converter blocos extraídos em inputs I3D

Depois de gerar `data/event_blocks_frames/<Classe>_<id>/frame_%06d.jpg`, você cria os inputs usados no treino.

#### Passo 1 — RGB (64 frames em 224×224)

Script: `src/preprocessing/sample_event_blocks_i3d.py`

Ele faz:

- Amostragem por segmentos para obter **exatamente 64 frames** por bloco
- Resize do lado menor para 256
- Center crop para 224×224
- Saída em `data/i3d_inputs/rgb/<Classe>_<id>/frame_%06d.jpg`

Comando (recomendado):

```bash
uv run python src/preprocessing/sample_event_blocks_i3d.py --overwrite
```

#### Passo 2 — Optical Flow (TV-L1)

Script: `src/preprocessing/gen_optical_flow.py`

Ele calcula fluxo ótico denso TV-L1 entre frames consecutivos (u,v), normaliza para 0..255 e salva:

- `flow_x_%06d.jpg`
- `flow_y_%06d.jpg`

em `data/i3d_inputs/optical_flow/<Classe>_<id>/`.

Comando:

```bash
uv run python src/preprocessing/gen_optical_flow.py --overwrite
```

**Por que o fluxo ótico?**
- RGB captura aparência; fluxo captura movimento. Em eventos como shoplifting, motion cues podem ser determinantes.

---

## 3) Training

Script principal: `src/training/train.py`

O treino:

- Carrega `data/i3d_inputs/rgb` e (se `rgb_optical`) `data/i3d_inputs/optical_flow`
- Faz split estratificado: 70% treino, 15% validação, 15% teste
- Treina com `BCEWithLogitsLoss(pos_weight=...)` para lidar com desbalanceamento
- (Opcional) aplica augmentation no RGB (flip + color jitter)
- Salva:
  - melhores checkpoints (top-3 por AUC-ROC) em `checkpoints/experiments_i3d/<nome>/model_weights/`
  - métricas em CSV
  - curvas ROC por época (npz)
  - vídeos de visualização (before/after augmentation)

### Dependências

- Recomendado instalar extras de dev (treino usa `scikit-learn`, `tqdm`, `natsort`, etc.):

```bash
uv sync --all-extras
```

- Necessário ter `ffmpeg` disponível no sistema para os scripts de extração.

### Executar treino

RGB + Optical Flow (late fusion):

```bash
uv run python src/training/train.py --model-mode rgb_optical --unfreeze-full-model --epochs 70 --batch-size 1
```

Apenas RGB:

```bash
uv run python src/training/train.py --model-mode rgb_only --unfreeze-full-model --epochs 70 --batch-size 1
```

> Dica: se sua GPU estoura memória, teste `src/training/find_max_batch_size.py` e ajuste `--batch-size`.

---

## Notas importantes

- Labels são inferidos pelo nome da pasta: se contiver `Shoplifting` → label=1, caso contrário label=0.
- Para `rgb_optical`, o checkpoint precisa conter também `model_flow_state_dict`.

testing