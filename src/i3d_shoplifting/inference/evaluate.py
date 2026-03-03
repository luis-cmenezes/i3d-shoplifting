"""
Script de inferência e avaliação no conjunto de TESTE para o modelo I3D.

===========================================================================
PASSO A PASSO — o que este script faz e por quê:
===========================================================================

1. CARREGAR O MELHOR CHECKPOINT DO TREINAMENTO
   ─ O loop de treino salva checkpoints nomeados como
     ``epoch_001_auc_0.5368.pt``.  Esse script encontra automaticamente
     o arquivo com o **maior AUC-ROC** no diretório ``model_weights/``
     do experimento escolhido.
   ─ O checkpoint contém ``model_rgb_state_dict`` (e opcionalmente
     ``model_flow_state_dict`` para o modo ``rgb_optical``).

2. RECRIAR O SPLIT DE TESTE COM A MESMA SEED
   ─ O treino usa ``train_test_split`` com ``random_state=seed`` e os
     mesmos parâmetros (test_size, val_test_ratio, stratify).
   ─ Reproduzimos **exatamente** os mesmos ``test_indices`` aqui,
     garantindo que nenhuma amostra de treino ou validação vaze para
     a avaliação.

3. INFERÊNCIA COM THRESHOLD CALIBRADO
   ─ O I3D usa ``BCEWithLogitsLoss`` → a saída é um **logit escalar**.
   ─ Para converter logits em probabilidades aplicamos ``torch.sigmoid``.
   ─ O threshold padrão de 0.5 nem sempre é ótimo quando as classes são
     desbalanceadas.  Este script calcula o **threshold ótimo via
     curva ROC (ponto mais próximo de (0,1))** sobre o conjunto de
     validação, e depois aplica esse threshold para binarizar as
     predições no conjunto de teste.

4. GERAÇÃO DE MÉTRICAS COMPLETAS
   ─ Accuracy, Precision, Recall, F1, AUC-ROC
   ─ Matriz de Confusão e Classification Report (texto + CSV)
   ─ Curva ROC salva como imagem PNG e dados brutos (.npz)
   ─ Tudo é exportado para um diretório ``test_evaluation/`` dentro do
     diretório de resultados do experimento.

===========================================================================
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from i3d_shoplifting.dataset.dataset import ShopliftingDataset
from i3d_shoplifting.models.i3d_pytorch import InceptionI3d


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 1  # saída binária via sigmoid


@dataclass
class EvalConfig:
    """Parâmetros para a avaliação de um experimento I3D."""

    # Identificação do experimento (ex: "aug_head_unfreeze_rgb_only")
    experiment_dir: str

    # Modalidade: "rgb_only" ou "rgb_optical"
    model_mode: str = "rgb_only"

    # Caminhos dos dados preprocessados
    rgb_dir: str = ""
    flow_dir: str = ""

    # Seed e split idênticos ao treino
    seed: int = 42
    split_test_size: float = 0.3
    split_val_test_ratio: float = 0.5

    # Batch size para inferência (pode ser maior que no treino)
    batch_size: int = 4


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def find_best_checkpoint(model_weights_dir: str) -> Path:
    """Encontra o checkpoint com maior AUC-ROC no diretório de pesos.

    O treino salva arquivos no formato ``epoch_NNN_auc_X.XXXX.pt``.
    Parseamos o AUC do nome e retornamos o caminho com o maior valor.
    """
    pattern = re.compile(r"epoch_(\d+)_auc_(\d\.\d+)\.pt")
    best_path: Path | None = None
    best_auc = -1.0

    weights_dir = Path(model_weights_dir)
    for f in weights_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            file_auc = float(match.group(2))
            if file_auc > best_auc:
                best_auc = file_auc
                best_path = f

    if best_path is None:
        raise FileNotFoundError(
            f"Nenhum checkpoint encontrado em {model_weights_dir} "
            f"com o padrão 'epoch_NNN_auc_X.XXXX.pt'"
        )
    print(f"Melhor checkpoint: {best_path.name} (AUC-ROC = {best_auc:.4f})")
    return best_path


def load_i3d_model(checkpoint_path: Path, model_mode: str):
    """Carrega o(s) modelo(s) I3D a partir de um checkpoint.

    Returns:
        Tupla (model_rgb, model_flow).  ``model_flow`` é ``None`` se
        ``model_mode == "rgb_only"``.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Modelo RGB (sempre presente)
    model_rgb = InceptionI3d(num_classes=400, in_channels=3)
    model_rgb.replace_logits(NUM_CLASSES)
    model_rgb.load_state_dict(checkpoint["model_rgb_state_dict"])
    model_rgb.to(DEVICE)
    model_rgb.eval()

    # Modelo de Fluxo Ótico (somente para rgb_optical)
    model_flow = None
    if model_mode == "rgb_optical" and checkpoint.get("model_flow_state_dict") is not None:
        model_flow = InceptionI3d(num_classes=400, in_channels=2)
        model_flow.replace_logits(NUM_CLASSES)
        model_flow.load_state_dict(checkpoint["model_flow_state_dict"])
        model_flow.to(DEVICE)
        model_flow.eval()

    return model_rgb, model_flow


def reproduce_splits(dataset, seed: int, test_size: float, val_test_ratio: float):
    """Reproduz exatamente os splits de treino/validação/teste.

    Retorna (train_idx, val_idx, test_idx) — os mesmos usados pelo treino.
    """
    indices = list(range(len(dataset)))
    labels = [dataset.get_label(i).item() for i in indices]

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=temp_labels,
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model_rgb,
    model_flow,
    dataloader: DataLoader,
    model_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Executa inferência e retorna probabilidades e labels verdadeiros.

    Para o I3D, a saída do modelo é um logit escalar.  Aplicamos sigmoid
    para obter a probabilidade da classe positiva (Shoplifting).

    Quando ``model_mode == "rgb_optical"`` fazemos late-fusion: a média
    das probabilidades de RGB e Fluxo Ótico.

    Returns:
        (all_probs, all_labels) — arrays numpy 1-D.
    """
    all_probs: list[float] = []
    all_labels: list[int] = []

    for batch in tqdm(dataloader, desc="Inferência"):
        # O dataset retorna (rgb_before, flow_before, rgb_after, flow_after, label)
        # Para avaliação não usamos augmentation, então rgb_before == rgb_after,
        # mas consumimos o formato igualmente.
        rgb_before, flow_before, rgb_after, flow_after, labels = batch

        rgb_input = rgb_after.to(DEVICE)
        labels_np = labels.numpy()

        logits_rgb = model_rgb(rgb_input)  # (B, 1)
        probs_rgb = torch.sigmoid(logits_rgb).cpu().numpy().flatten()

        if model_mode == "rgb_optical" and model_flow is not None:
            flow_input = flow_after.to(DEVICE)
            logits_flow = model_flow(flow_input)
            probs_flow = torch.sigmoid(logits_flow).cpu().numpy().flatten()
            # Late fusion: média das probabilidades
            probs = (probs_rgb + probs_flow) / 2.0
        else:
            probs = probs_rgb

        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())

    return np.array(all_probs), np.array(all_labels)


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Encontra o threshold ótimo na curva ROC (ponto mais próximo de (0,1)).

    Isso é chamado de critério de Youden ou "closest to top-left corner".
    É útil quando as classes são desbalanceadas e o threshold 0.5 pode
    não ser ideal.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    # Distância euclidiana de cada ponto até (0, 1)
    distances = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    optimal_idx = np.argmin(distances)
    optimal_threshold = float(thresholds[optimal_idx])
    print(f"Threshold ótimo (validação): {optimal_threshold:.4f}  "
          f"(FPR={fpr[optimal_idx]:.4f}, TPR={tpr[optimal_idx]:.4f})")
    return optimal_threshold


# ---------------------------------------------------------------------------
# Métricas e relatórios
# ---------------------------------------------------------------------------

def compute_and_save_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    output_dir: Path,
):
    """Calcula todas as métricas e salva artefatos no ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = (probs >= threshold).astype(int)

    # --- Métricas escalares ---
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    auc_roc = roc_auc_score(labels, probs)

    metrics = {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc_roc,
        "num_samples": int(len(labels)),
        "num_positive": int(labels.sum()),
        "num_negative": int((labels == 0).sum()),
    }

    # Salva métricas como JSON
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMétricas salvas em {metrics_path}")

    # --- Classification Report ---
    report_str = classification_report(
        labels, preds,
        target_names=["Normal", "Shoplifting"],
        digits=4,
    )
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report_str)

    # --- Matriz de Confusão ---
    cm = confusion_matrix(labels, preds)
    cm_path = output_dir / "confusion_matrix.txt"
    cm_path.write_text(
        f"Confusion Matrix (threshold={threshold:.4f}):\n"
        f"               Pred Normal  Pred Shoplifting\n"
        f"True Normal     {cm[0, 0]:>10}  {cm[0, 1]:>16}\n"
        f"True Shoplift   {cm[1, 0]:>10}  {cm[1, 1]:>16}\n"
    )

    # --- Curva ROC (dados brutos) ---
    fpr, tpr, thresholds_roc = roc_curve(labels, probs)
    roc_auc_val = auc(fpr, tpr)
    np.savez(
        output_dir / "roc_curve_test.npz",
        fpr=fpr, tpr=tpr, thresholds=thresholds_roc, auc=roc_auc_val,
    )

    # --- Curva ROC (imagem PNG) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatório")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curva ROC — Conjunto de Teste (I3D)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve_test.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib não disponível — pulando geração do PNG da curva ROC.")

    # --- Print resumo ---
    print("\n" + "=" * 60)
    print("RESULTADOS NO CONJUNTO DE TESTE (I3D)")
    print("=" * 60)
    print(f"  Threshold:  {threshold:.4f}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  AUC-ROC:    {auc_roc:.4f}")
    print(f"  Amostras:   {len(labels)} (Pos={int(labels.sum())}, Neg={int((labels==0).sum())})")
    print("=" * 60)
    print(f"\nRelatório completo:\n{report_str}")
    print(f"Matriz de Confusão:\n{cm}\n")

    return metrics


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def evaluate(cfg: EvalConfig) -> dict:
    """Executa o pipeline completo de avaliação no conjunto de teste.

    Args:
        cfg: instância de EvalConfig com todos os parâmetros.

    Returns:
        Dicionário com todas as métricas calculadas.
    """
    experiment_path = Path(cfg.experiment_dir)
    model_weights_dir = experiment_path / "model_weights"
    eval_output_dir = experiment_path / "test_evaluation"

    print(f"Dispositivo: {DEVICE}")
    print(f"Experimento: {experiment_path}")
    print(f"Modo: {cfg.model_mode}")

    # =====================================================================
    # PASSO 1 — Carregar o melhor modelo
    # =====================================================================
    best_ckpt = find_best_checkpoint(str(model_weights_dir))
    model_rgb, model_flow = load_i3d_model(best_ckpt, cfg.model_mode)
    print("Modelo(s) carregado(s) com sucesso.")

    # =====================================================================
    # PASSO 2 — Recriar o dataset completo (sem augmentation)
    # =====================================================================
    rgb_dir = Path(cfg.rgb_dir)
    flow_dir = Path(cfg.flow_dir)
    dataset_no_aug = ShopliftingDataset(rgb_dir=rgb_dir, flow_dir=flow_dir, transform=None)

    print(f"Dataset total: {len(dataset_no_aug)} amostras")

    # =====================================================================
    # PASSO 3 — Reproduzir splits (mesma seed do treino)
    # =====================================================================
    train_idx, val_idx, test_idx = reproduce_splits(
        dataset_no_aug,
        seed=cfg.seed,
        test_size=cfg.split_test_size,
        val_test_ratio=cfg.split_val_test_ratio,
    )

    print(f"Split — Treino: {len(train_idx)}, Validação: {len(val_idx)}, Teste: {len(test_idx)}")

    # Verificar distribuição de classes do teste
    test_labels_preview = [dataset_no_aug.get_label(i).item() for i in test_idx]
    print(f"Teste — Normal: {sum(1 for lb in test_labels_preview if lb == 0)}, "
          f"Shoplifting: {sum(1 for lb in test_labels_preview if lb == 1)}")

    # =====================================================================
    # PASSO 4A — Inferência no conjunto de VALIDAÇÃO para calibrar threshold
    # =====================================================================
    print("\n--- Inferência no conjunto de VALIDAÇÃO (para calibrar threshold) ---")
    val_dataset = Subset(dataset_no_aug, val_idx)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    val_probs, val_labels = run_inference(model_rgb, model_flow, val_loader, cfg.model_mode)
    optimal_threshold = find_optimal_threshold(val_labels, val_probs)

    # =====================================================================
    # PASSO 4B — Inferência no conjunto de TESTE
    # =====================================================================
    print("\n--- Inferência no conjunto de TESTE ---")
    test_dataset = Subset(dataset_no_aug, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    test_probs, test_labels = run_inference(model_rgb, model_flow, test_loader, cfg.model_mode)

    # =====================================================================
    # PASSO 5 — Calcular e salvar métricas
    # =====================================================================
    metrics = compute_and_save_metrics(
        labels=test_labels,
        probs=test_probs,
        threshold=optimal_threshold,
        output_dir=eval_output_dir,
    )

    # Salvar também probabilidades brutas para análises futuras
    np.savez(
        eval_output_dir / "test_predictions.npz",
        probs=test_probs,
        labels=test_labels,
        threshold=optimal_threshold,
        test_indices=np.array(test_idx),
    )
    print(f"Predições brutas salvas em {eval_output_dir / 'test_predictions.npz'}")

    return metrics


# ---------------------------------------------------------------------------
# Entrypoint standalone (uso direto via CLI)
# ---------------------------------------------------------------------------

def _parse_args() -> EvalConfig:
    import argparse

    parser = argparse.ArgumentParser(
        description="Avaliação I3D no conjunto de teste",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Caminho do diretório do experimento (ex: results/i3d/aug_head_unfreeze_rgb_only)",
    )
    parser.add_argument(
        "--model-mode",
        type=str,
        default="rgb_only",
        choices=["rgb_only", "rgb_optical"],
    )
    parser.add_argument("--rgb-dir", type=str, required=True)
    parser.add_argument("--flow-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-test-size", type=float, default=0.3)
    parser.add_argument("--split-val-test-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()
    return EvalConfig(
        experiment_dir=args.experiment_dir,
        model_mode=args.model_mode,
        rgb_dir=args.rgb_dir,
        flow_dir=args.flow_dir,
        seed=args.seed,
        split_test_size=args.split_test_size,
        split_val_test_ratio=args.split_val_test_ratio,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    evaluate(cfg)
