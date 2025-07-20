import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Importa as classes que criamos
from src.models.i3d_pytorch import InceptionI3d
from src.common.dataset import ShopliftingDataset

# --- 1. CONFIGURAÇÃO ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Caminhos para os dados
RGB_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'rgb'
FLOW_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'optical_flow'

# --- IMPORTANTE: ATUALIZE ESTE CAMINHO ---
# Aponte para o seu melhor checkpoint salvo pelo script de treinamento
CHECKPOINT_PATH = PROJECT_ROOT / 'checkpoints' / 'shoplifting-training' / 'epoch_40_recall_0.8286.pt' 
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'evaluation_results'

SEED = 42 # Deve ser o MESMO usado no treinamento para garantir o mesmo split

def load_models_from_checkpoint(checkpoint_path, num_classes=2):
    """Carrega os modelos RGB e de Fluxo a partir de um único arquivo de checkpoint."""
    model_rgb = InceptionI3d(num_classes=num_classes, in_channels=3)
    model_flow = InceptionI3d(num_classes=num_classes, in_channels=2)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model_rgb.load_state_dict(checkpoint['model_rgb_state_dict'])
        model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
        print(f"Modelos carregados com sucesso do checkpoint da época {checkpoint.get('epoch', 'N/A')}")
    except FileNotFoundError:
        print(f"ERRO: Checkpoint não encontrado em '{checkpoint_path}'. Abortando.")
        return None, None
    except Exception as e:
        print(f"ERRO ao carregar o checkpoint: {e}")
        return None, None
        
    model_rgb.to(DEVICE)
    model_flow.to(DEVICE)
    model_rgb.eval()
    model_flow.eval()
    
    return model_rgb, model_flow

def evaluate():
    """Função principal que orquestra a avaliação no conjunto de teste."""
    print(f"Usando dispositivo: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. REPRODUÇÃO DO SPLIT DE TESTE ---
    full_dataset = ShopliftingDataset(rgb_dir=RGB_DIR, flow_dir=FLOW_DIR)
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.get_label(idx) for idx in indices]

    # Usa o mesmo random_state para garantir que os splits sejam idênticos
    _, temp_indices, _, temp_labels = train_test_split(
        indices, labels, test_size=0.3, random_state=SEED, stratify=labels)
    _, test_indices, _, test_labels = train_test_split(
        temp_indices, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels)

    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch size 1 para visualização

    print(f"Conjunto de teste carregado com {len(test_dataset)} amostras.")

    # # --- 3. CARREGAMENTO DO MODELO TREINADO ---
    model_rgb, model_flow = load_models_from_checkpoint(CHECKPOINT_PATH)
    if model_rgb is None:
        return

    # # --- 4. AVALIAÇÃO E VISUALIZAÇÃO ---
    all_preds = []
    all_labels = []

    class_map = {0: "Normal", 1: "Shoplifting"}
    color_map = {True: (0, 255, 0), False: (0, 0, 255)} # Verde para correto, Vermelho para incorreto

    with torch.no_grad():
        for i, (rgb_data, flow_data, label) in enumerate(tqdm(test_loader, desc="Avaliando no Teste")):

            original_dataset_index = test_indices[i]
            block_name = full_dataset.samples[original_dataset_index]

            rgb_data, flow_data, label = rgb_data.to(DEVICE), flow_data.to(DEVICE), label.to(DEVICE)
            
            # Inferência
            rgb_logits = torch.mean(model_rgb(rgb_data), dim=2)
            flow_logits = torch.mean(model_flow(flow_data), dim=2)
            final_logits = (rgb_logits + flow_logits) / 2
            
            pred_prob = torch.softmax(final_logits, dim=1)
            pred_class = torch.argmax(pred_prob, dim=1)
            
            true_label = label.item()
            predicted_label = pred_class.item()
            
            all_preds.append(predicted_label)
            all_labels.append(true_label)
            
            # --- 5. OUTPUT VISUAL ---
            is_correct = (predicted_label == true_label)
            result_prefix = "CORRETO" if is_correct else "INCORRETO"
            video_filename = f"{result_prefix}_{block_name}.mp4"
            video_save_path = OUTPUT_DIR / video_filename

            display_tensor = rgb_data.squeeze(0).permute(1, 2, 3, 0) 
            display_numpy = (display_tensor.cpu().numpy() * 255).astype(np.uint8)

            height, width, _ = display_numpy[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_save_path), fourcc, 25.0, (1280, 720))

            text_true = f"Real: {class_map[true_label]}"
            text_pred = f"Previsto: {class_map[predicted_label]} ({pred_prob[0][predicted_label]:.2f})"

            for frame_numpy in display_numpy:
                # Converte de RGB (PyTorch) para BGR (OpenCV) para exibição correta das cores
                display_frame = cv2.resize(cv2.cvtColor(frame_numpy, cv2.COLOR_RGB2BGR), (1280,720))
                
                cv2.putText(display_frame, text_true, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display_frame, text_pred, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[is_correct], 2)
                video_writer.write(display_frame)
                
            video_writer.release()
                
    # --- 6. MÉTRICAS FINAIS ---
    print("\n--- Relatório Final de Métricas no Conjunto de Teste ---")
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Revocação (Recall): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nMatriz de Confusão:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_map.values(), 
                yticklabels=class_map.values())
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão no Conjunto de Teste')
    
    confusion_matrix_path = OUTPUT_DIR / 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"\nImagem da Matriz de Confusão salva em: {confusion_matrix_path}")

if __name__ == '__main__':
    evaluate()
