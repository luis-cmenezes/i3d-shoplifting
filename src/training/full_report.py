import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importa as classes necessárias do seu projeto
from src.models.i3d_pytorch import InceptionI3d
from src.common.dataset import ShopliftingDataset

### ALTERAÇÃO 1: Importa a função de avaliação do outro script ###
from src.training.evaluate_experiments import evaluate_experiments, get_predictions, get_best_checkpoint_path

# --- CONFIGURAÇÕES GLOBAIS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints' / 'experiments_i3d'
RGB_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'rgb'
FLOW_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'optical_flow'
OUTPUT_DIR = PROJECT_ROOT / 'full_report'
OUTPUT_DIR.mkdir(exist_ok=True)

# --- FUNÇÕES DE VISUALIZAÇÃO ---

def plot_detailed_learning_curves(results_df):
    """Plota curvas de aprendizado detalhadas (todas as métricas de validação) para todos os experimentos."""
    print("A gerar curvas de aprendizado detalhadas...")
    metrics_to_plot = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//2, i%2]
        for _, row in results_df.iterrows():
            log_path = BASE_CHECKPOINTS_DIR / row['Experimento'] / 'training_log.csv'
            if not log_path.exists(): continue
            log_df = pd.read_csv(log_path)
            ax.plot(log_df['epoch'], log_df[metric], label=row['Experimento'], lw=2)
        ax.set_title(f'{metric.replace("_", " ").title()} na Validação', fontsize=16)
        ax.set_xlabel('Época'); ax.set_ylabel(metric.split("_")[1].capitalize()); ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detailed_learning_curves.png', dpi=300)
    plt.close()

def plot_main_learning_curves(results_df):
    """Plota as curvas de aprendizado principais (val_auc_roc e train_loss)."""
    print("A gerar curvas de aprendizado principais...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    for _, row in results_df.iterrows():
        log_path = BASE_CHECKPOINTS_DIR / row['Experimento'] / 'training_log.csv'
        if not log_path.exists(): continue
        log_df = pd.read_csv(log_path)
        axes[0].plot(log_df['epoch'], log_df['val_auc_roc'], label=row['Experimento'], lw=2)
        axes[1].plot(log_df['epoch'], log_df['train_loss'], label=row['Experimento'], lw=2)
    axes[0].set_title('AUC-ROC na Validação Durante o Treinamento', fontsize=16)
    axes[0].set_xlabel('Época'); axes[0].set_ylabel('AUC-ROC'); axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(); axes[0].set_ylim(0.5, 1.0)
    axes[1].set_title('Perda (Loss) no Treino Durante o Treinamento', fontsize=16)
    axes[1].set_xlabel('Época'); axes[1].set_ylabel('BCEWithLogitsLoss'); axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'main_learning_curves.png', dpi=300); plt.close()

def plot_comparative_roc(results_df):
    """Plota a curva ROC da melhor época de cada experimento em um único gráfico."""
    print("A gerar curvas ROC comparativas (melhor época)...")
    plt.figure(figsize=(10, 8))
    for _, row in results_df.iterrows():
        exp_dir = BASE_CHECKPOINTS_DIR / row['Experimento']
        log_path = exp_dir / 'training_log.csv'
        if not log_path.exists(): continue
        log_df = pd.read_csv(log_path)
        best_epoch = log_df.loc[log_df['val_auc_roc'].idxmax()]
        roc_file = exp_dir / 'roc_curves' / f"roc_data_epoch_{int(best_epoch['epoch']):03d}.npz"
        if roc_file.exists():
            data = np.load(roc_file)
            plt.plot(data['fpr'], data['tpr'], lw=2, label=f"{row['Experimento']} (AUC = {best_epoch['val_auc_roc']:.3f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)'); plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curvas ROC Comparativas (Melhor Época de cada Experimento)', fontsize=16)
    plt.legend(loc="lower right"); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(OUTPUT_DIR / 'comparative_roc_curves_best_epoch.png', dpi=300); plt.close()

def plot_staged_roc_curves(results_df, total_epochs=70, num_stages=6):
    """Plota curvas ROC comparativas em diferentes estágios (épocas) do treinamento."""
    print("A gerar curvas ROC por estágio de treinamento...")
    epochs_to_plot = np.linspace(total_epochs / num_stages, total_epochs, num_stages, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    for i, epoch in enumerate(epochs_to_plot):
        ax = axes[i]
        for _, row in results_df.iterrows():
            exp_dir = BASE_CHECKPOINTS_DIR / row['Experimento']
            roc_file = exp_dir / 'roc_curves' / f'roc_data_epoch_{epoch:03d}.npz'
            if roc_file.exists():
                data = np.load(roc_file)
                ax.plot(data['fpr'], data['tpr'], lw=2, label=f"{row['Experimento'].replace('_', ' ').title()} (AUC={data['roc_auc']:.2f})")
        
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_title(f'Comparativo na Época {epoch}', fontsize=14)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.grid(True, linestyle='--', alpha=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Evolução das Curvas ROC Durante o Treinamento', fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'staged_roc_curves.png', dpi=300)
    plt.close()


def analyze_model_performance(model_name, threshold):
    """Gera Matriz de Confusão e Relatório de Classificação para um modelo específico."""
    print(f"A analisar o modelo: {model_name}")
    
    # Recria o split de teste para obter o data loader
    dataset_no_aug = ShopliftingDataset(rgb_dir=RGB_DIR, flow_dir=FLOW_DIR, transform=None)
    indices = list(range(len(dataset_no_aug)))
    labels = [dataset_no_aug.get_label(idx) for idx in indices]
    _, temp_indices = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    _, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42, stratify=[labels[i] for i in temp_indices])
    test_dataset = Subset(dataset_no_aug, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    exp_dir = BASE_CHECKPOINTS_DIR / model_name
    model_mode = 'rgb_optical' if 'rgb_optical' in model_name else 'rgb_only'
    checkpoint_path = get_best_checkpoint_path(exp_dir)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_rgb = InceptionI3d(num_classes=400, in_channels=3); model_rgb.replace_logits(1)
    model_rgb.load_state_dict(checkpoint['model_rgb_state_dict']); model_rgb.to(DEVICE)
    model_flow = None
    if model_mode == 'rgb_optical':
        model_flow = InceptionI3d(num_classes=400, in_channels=2); model_flow.replace_logits(1)
        model_flow.load_state_dict(checkpoint['model_flow_state_dict']); model_flow.to(DEVICE)

    test_probs, test_labels = get_predictions(model_rgb, model_flow, test_loader, model_mode)
    test_preds = (test_probs > threshold).astype(int)
    
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Shoplifting'], yticklabels=['Normal', 'Shoplifting'])
    plt.title(f'Matriz de Confusão - {model_name}\nThreshold = {threshold:.3f}', fontsize=16)
    plt.ylabel('Classe Verdadeira'); plt.xlabel('Classe Prevista')
    print(f"{model_name} confusion matrix:\n{cm}")
    plt.savefig(OUTPUT_DIR / f'confusion_matrix_{model_name}.png', dpi=300); plt.close()
    
    report = classification_report(test_labels, test_preds, target_names=['Normal', 'Shoplifting'])
    print(f"\n--- Relatório de Classificação para '{model_name}' ---\n"); print(report)
    with open(OUTPUT_DIR / f'classification_report_{model_name}.txt', 'w') as f:
        f.write(f"Relatório para: {model_name}\nThresh: {threshold}\n\n{report}")

### FUNÇÃO PRINCIPAL DO RELATÓRIO ###
def main():
    print("A iniciar geração de relatório completo...")
    
    results_df = evaluate_experiments()
    
    if results_df.empty:
        print("Nenhum resultado foi gerado pela avaliação. Verifique os logs.")
        return
        
    print("\n\n--- Tabela Comparativa de Resultados Finais (Gerada Dinamicamente) ---")
    print(results_df.to_string())
    results_df.to_csv(OUTPUT_DIR / 'final_results_summary.csv', index=False)
    
    # 2. Gera os gráficos comparativos
    plot_main_learning_curves(results_df)
    plot_detailed_learning_curves(results_df)
    plot_comparative_roc(results_df)
    plot_staged_roc_curves(results_df) # Novo gráfico
    
    # 3. Analisa TODOS os modelos encontrados
    print("\n--- A gerar relatórios detalhados para cada modelo ---")
    for index, row in results_df.iterrows():
        analyze_model_performance(row['Experimento'], row['Threshold Ideal'])
    
    print(f"\nAnálise completa. Todos os gráficos e relatórios foram salvos em: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()

