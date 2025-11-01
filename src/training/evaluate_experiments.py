# evaluate_models.py

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from tqdm import tqdm
from pathlib import Path
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importa as classes necessárias do seu projeto
from src.models.i3d_pytorch import InceptionI3d
from src.common.dataset import ShopliftingDataset

# --- CONFIGURAÇÕES ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints' / 'experiments_i3d'
RGB_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'rgb'
FLOW_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'optical_flow'
BATCH_SIZE = 8 # Pode aumentar o batch size para avaliação, pois não há backpropagation

def get_data_splits():
    """
    Recria exatamente os mesmos splits de dados usados no treinamento.
    É crucial usar o mesmo random_state.
    """
    dataset_no_aug = ShopliftingDataset(rgb_dir=RGB_DIR, flow_dir=FLOW_DIR, transform=None)
    indices = list(range(len(dataset_no_aug)))
    labels = [dataset_no_aug.get_label(idx) for idx in indices]

    train_indices, temp_indices = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels)
    
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42, stratify=[labels[i] for i in temp_indices])

    val_dataset = Subset(dataset_no_aug, val_indices)
    test_dataset = Subset(dataset_no_aug, test_indices)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return val_loader, test_loader

def get_best_checkpoint_path(experiment_dir):
    """Encontra o caminho do checkpoint com o maior AUC-ROC no nome."""
    weights_dir = experiment_dir / 'model_weights'
    checkpoints = os.listdir(weights_dir)
    
    best_auc = -1
    best_checkpoint = None
    
    pattern = re.compile(r'epoch_(\d+)_auc_(\d\.\d+)\.pt')
    for f in checkpoints:
        match = pattern.match(f)
        if match:
            auc = float(match.group(2))
            if auc > best_auc:
                best_auc = auc
                best_checkpoint = f
                
    if not best_checkpoint:
        raise FileNotFoundError(f"Nenhum checkpoint válido encontrado em {weights_dir}")
    
    return weights_dir / best_checkpoint

def get_predictions(model_rgb, model_flow, data_loader, model_mode):
    """Executa o modelo no data_loader e retorna probabilidades e rótulos."""
    model_rgb.eval()
    if model_flow:
        model_flow.eval()

    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for _, _, rgb_data, flow_data, labels in tqdm(data_loader, desc="Avaliando"):
            rgb_data = rgb_data.to(DEVICE)
            
            rgb_logits = model_rgb(rgb_data)
            
            if model_mode == 'rgb_optical':
                flow_data = flow_data.to(DEVICE)
                flow_logits = model_flow(flow_data)
                final_logits = (rgb_logits + flow_logits) / 2
            else: # rgb_only
                final_logits = rgb_logits

            probs = torch.sigmoid(final_logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
    return np.array(all_probs), np.array(all_labels)

def find_optimal_threshold(probs, labels):
    """Encontra o threshold que maximiza o F2-score."""
    best_f2 = -1
    best_threshold = 0.5
    
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (probs > threshold).astype(int)
        f2 = fbeta_score(labels, preds, beta=2, zero_division=0)
        
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold
            
    return best_threshold, best_f2

def evaluate_experiments():
    print("Iniciando avaliação dos modelos...")
    val_loader, test_loader = get_data_splits()
    
    experiment_folders = [d for d in BASE_CHECKPOINTS_DIR.iterdir() if d.is_dir()]
    
    results = []

    for experiment_dir in experiment_folders:
        print(f"\n--- Processando Experimento: {experiment_dir.name} ---")
        
        # Determina o modo do modelo a partir do nome da pasta
        model_mode = 'rgb_optical' if 'rgb_optical' in experiment_dir.name else 'rgb_only'
        
        # 1. Encontrar e carregar o melhor checkpoint
        best_checkpoint_path = get_best_checkpoint_path(experiment_dir)
        print(f"Melhor checkpoint encontrado: {best_checkpoint_path.name}")
        checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
        
        model_rgb = InceptionI3d(num_classes=400, in_channels=3)
        model_rgb.replace_logits(1)
        model_rgb.load_state_dict(checkpoint['model_rgb_state_dict'])
        model_rgb.to(DEVICE)
        
        model_flow = None
        if model_mode == 'rgb_optical':
            if 'model_flow_state_dict' not in checkpoint or checkpoint['model_flow_state_dict'] is None:
                print(f"AVISO: Modo é {model_mode}, mas não foi encontrado 'model_flow_state_dict' no checkpoint.")
                continue
            model_flow = InceptionI3d(num_classes=400, in_channels=2)
            model_flow.replace_logits(1)
            model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
            model_flow.to(DEVICE)
            
        # 2. Obter predições no CONJUNTO DE VALIDAÇÃO
        val_probs, val_labels = get_predictions(model_rgb, model_flow, val_loader, model_mode)
        
        # 3. Encontrar o threshold ideal usando o conjunto de validação
        optimal_threshold, best_f2_val = find_optimal_threshold(val_probs, val_labels)
        print(f"Threshold ideal (max F2-score na validação): {optimal_threshold:.4f}")
        
        # 4. AVALIAÇÃO FINAL: Usar o modelo e o threshold ideal no CONJUNTO DE TESTE
        print("Avaliando no conjunto de teste com o threshold ideal...")
        test_probs, test_labels = get_predictions(model_rgb, model_flow, test_loader, model_mode)
        
        # Aplicar o threshold para obter as predições finais
        test_preds = (test_probs > optimal_threshold).astype(int)
        
        # Calcular métricas finais no conjunto de teste
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
        f2 = fbeta_score(test_labels, test_preds, beta=2, zero_division=0)
        
        # Carregar log de treino para obter a melhor AUC-ROC da validação
        log_df = pd.read_csv(experiment_dir / 'training_log.csv')
        best_val_auc = log_df['val_auc_roc'].max()
        
        results.append({
            'Experimento': experiment_dir.name,
            'Melhor AUC Validação': best_val_auc,
            'Threshold Ideal': optimal_threshold,
            'F2-Score Teste': f2,
            'Precisão Teste': precision,
            'Revocação Teste': recall,
            'F1-Score Teste': f1
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(by='F2-Score Teste', ascending=False).reset_index(drop=True)

if __name__ == '__main__':
    results_df = evaluate_experiments()
    print("\n\n--- Tabela Comparativa de Resultados Finais (Gerada Dinamicamente) ---")
    print(results_df.to_string())