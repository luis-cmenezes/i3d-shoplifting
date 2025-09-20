import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import natsorted
from pathlib import Path

TRAINING = "new_changes_data_aug"
TRAINING_DIR = Path(f"/home/luis/tcc/code/checkpoints/{TRAINING}") 
RESULTS_DIR = TRAINING_DIR / "plots"
LOG_FILE = TRAINING_DIR / "training_log.csv"
OUTPUT_METRICS_IMG = RESULTS_DIR / "all_metrics_vs_epoch.png"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(LOG_FILE)

    # Seleciona todas as colunas de métricas, incluindo a nova 'val_auc_roc'
    metrics = df.columns[1:]
    n = len(metrics)
    cols = 2
    rows = (n + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        axs[i].plot(df["epoch"], df[metric], marker="o")
        axs[i].set_title(f"{metric} x Epoch")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel(metric)
        axs[i].set_ylim(0, 1)
        axs[i].grid(True)

    # Remove subplots não usados
    for i in range(len(metrics), len(axs)):
        fig.delaxes(axs[i])  

    plt.tight_layout()
    plt.savefig(TRAINING_DIR / "all_metrics_vs_epoch.png")
    plt.close()
except FileNotFoundError:
    print(f"Arquivo de log não encontrado em '{LOG_FILE}'. Pulando o gráfico de métricas.")

# --- GRÁFICO 2: CURVAS ROC SOBREPOSTAS ---

# Encontra todos os arquivos de dados da curva ROC
roc_dir = Path(os.path.join(TRAINING_DIR,"roc_curves"))
roc_files = natsorted([f for f in os.listdir(roc_dir) if f.startswith('roc_data_epoch_') and f.endswith('.npz')])

if roc_files:
    plt.figure(figsize=(10, 8))

    # Adiciona a linha de referência (classificador aleatório)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório (AUC = 0.50)')

    for i, roc_file in enumerate(roc_files):
        epoch_num = i + 1
        data = np.load(roc_dir / roc_file)
        fpr = data['fpr']
        tpr = data['tpr']
        roc_auc = data['roc_auc']

        # Plota a curva da época atual
        plt.plot(fpr, tpr, lw=2, label=f'Época {epoch_num} (AUC = {roc_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)')
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR / Recall)')
        plt.title('Evolução da Curva ROC Durante o Treinamento')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"roc_curves_vs_epoch_{epoch_num}.png")
        plt.close()
        # print(f"Gráfico da curva ROC salvo em: {RESULTS_DIR / f"roc_curves_vs_epoch_{epoch_num}.png"}")
else:
    print(f"Nenhum arquivo 'roc_data_epoch_*.npz' encontrado em '{RESULTS_DIR}'. Pulando o gráfico ROC.")

