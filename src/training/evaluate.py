import torch
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.i3d_pytorch import InceptionI3d
from src.common.dataset import ShopliftingDataset

# --- 1. CONFIGURAÇÃO ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Caminhos dos dados
RGB_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'rgb'
FLOW_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'optical_flow'

# --- IMPORTANTE: ATUALIZE ESTE CAMINHO ---
TRAINING = 'new_changes_data_aug'
CHECKPOINT_FILENAME = 'epoch_032_recall_0.8286.pt'

CHECKPOINT_PARENT_DIR = PROJECT_ROOT / 'checkpoints' / TRAINING / 'model_weights'
CHECKPOINT_PATH = CHECKPOINT_PARENT_DIR / CHECKPOINT_FILENAME

# Saídas
OUTPUT_DIR = PROJECT_ROOT / 'evaluation_results' / CHECKPOINT_FILENAME.replace('.pt', '')
OUTPUT_DIR_VIDEOS = OUTPUT_DIR / 'videos'

# Seed DEVE ser 42 para replicar o split do script de treino
SEED = 42
NUM_CLASSES_MODEL = 1 # Nosso modelo foi treinado com 1 saída para BCE Loss

def load_models_from_checkpoint(checkpoint_path, num_classes_output):
    """Carrega os modelos RGB e de Fluxo a partir do nosso checkpoint de treino."""
    
    # 1. Inicializa a arquitetura base (pré-treinada no Kinetics-400)
    model_rgb = InceptionI3d(num_classes=400, in_channels=3)
    model_flow = InceptionI3d(num_classes=400, in_channels=2)
    
    # 2. Substitui os logits para corresponder à NOSSA tarefa (1 saída)
    #    Isso é crucial para que a arquitetura do modelo corresponda às chaves do state_dict
    model_rgb.replace_logits(num_classes_output)
    model_flow.replace_logits(num_classes_output)

    try:
        print(f"Carregando checkpoint de: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        model_rgb.load_state_dict(checkpoint['model_rgb_state_dict'])
        model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
        print(f"Modelos carregados com sucesso! Checkpoint da época {checkpoint.get('epoch', 'N/A')+1} (Recall: {checkpoint.get('recall', 'N/A'):.4f})")
    
    except FileNotFoundError:
        print(f"ERRO: Checkpoint não encontrado em '{checkpoint_path}'. Abortando.")
        return None, None
    except Exception as e:
        print(f"ERRO ao carregar o checkpoint: {e}")
        print("Certifique-se de que NUM_CLASSES_MODEL está correto (deve ser 1).")
        return None, None
        
    model_rgb.to(DEVICE)
    model_flow.to(DEVICE)
    model_rgb.eval()
    model_flow.eval()
    
    return model_rgb, model_flow

def plot_confusion_matrix(cm, class_map, save_path):
    """Salva a imagem da matriz de confusão a partir de uma matriz cm calculada."""
    print("\nGerando imagem da Matriz de Confusão...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_map.values(), 
                yticklabels=class_map.values())
    plt.xlabel('Previsto pelo Modelo')
    plt.ylabel('Rótulo Real')
    plt.title('Matriz de Confusão no Conjunto de Teste')
    
    plt.savefig(save_path)
    plt.close()
    print(f"Imagem da Matriz de Confusão salva em: {save_path}")

def evaluate():
    """Função principal que orquestra a avaliação no conjunto de teste."""
    print(f"Usando dispositivo: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_VIDEOS.mkdir(parents=True, exist_ok=True)
    print(f"Resultados (vídeos e métricas) serão salvos em: {OUTPUT_DIR}")

    # --- 2. REPRODUÇÃO EXATA DO SPLIT DE TESTE (do train.py) ---
    full_dataset = ShopliftingDataset(rgb_dir=RGB_DIR, flow_dir=FLOW_DIR, transform=None)
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.get_label(idx) for idx in indices]

    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, labels, test_size=0.3, random_state=SEED, stratify=labels)
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels)

    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Conjunto de teste (15%) carregado com {len(test_dataset)} amostras.")

    # --- 3. CARREGAMENTO DO MODELO TREINADO ---
    model_rgb, model_flow = load_models_from_checkpoint(CHECKPOINT_PATH, NUM_CLASSES_MODEL)
    if model_rgb is None:
        return

    # --- 4. AVALIAÇÃO E GERAÇÃO DE VÍDEO ---
    all_preds = []
    all_labels = []
    all_probs = []

    class_map = {0: "Normal", 1: "Shoplifting"}
    color_map = {"correto": (0, 255, 0), "incorreto": (0, 0, 255), "texto_overlay": (255, 220, 100)}
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    line_type = 2

    with torch.no_grad():
        for i, (rgb_before, flow_before, rgb_after, flow_after, label) in enumerate(tqdm(test_loader, desc="Avaliando no Teste")):
            original_dataset_index = test_indices[i]
            block_name = full_dataset.samples[original_dataset_index]
            rgb_data, flow_data = rgb_after.to(DEVICE), flow_after.to(DEVICE)
            
            rgb_logits = model_rgb(rgb_data)
            flow_logits = model_flow(flow_data)
            final_logits = (rgb_logits + flow_logits) / 2 
            pred_prob = torch.sigmoid(final_logits) 
            pred_class_tensor = (pred_prob > 0.5).long()
            
            true_label = label.item()
            predicted_label = pred_class_tensor.item()
            probability_score = pred_prob.item() 
            
            all_preds.append(predicted_label)
            all_labels.append(true_label)
            all_probs.append(probability_score)
            
            # --- 5. Geração do Vídeo de Output ---
            is_correct = (predicted_label == true_label)
            result_prefix = "CORRETO" if is_correct else "INCORRETO"
            video_filename = f"{result_prefix}_{block_name}.mp4"
            video_save_path = OUTPUT_DIR_VIDEOS / video_filename

            display_tensor = rgb_before.squeeze(0).permute(1, 2, 3, 0) 
            display_numpy = (display_tensor.cpu().numpy() * 255).astype(np.uint8)
            H, W, _ = display_numpy[0].shape 
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_save_path), fourcc, 10.0, (W, H))

            text_true = f"Real: {class_map[true_label]}"
            prob_display = probability_score if predicted_label == 1 else (1.0 - probability_score)
            text_pred = f"Previsto: {class_map[predicted_label]} ({prob_display:.2f})"
            text_color = color_map["correto"] if is_correct else color_map["incorreto"]

            for frame_numpy in display_numpy:
                display_frame_bgr = cv2.cvtColor(frame_numpy, cv2.COLOR_RGB2BGR)
                cv2.putText(display_frame_bgr, text_true, (5, H - 25), font, font_scale, color_map["texto_overlay"], line_type)
                cv2.putText(display_frame_bgr, text_pred, (5, H - 10), font, font_scale, text_color, line_type)
                video_writer.write(display_frame_bgr)
            video_writer.release()
                
    # --- 6. MÉTRICAS FINAIS (COM ARQUIVO DE RELATÓRIO) ---
    print("\n--- Relatório Final de Métricas no Conjunto de Teste ---")
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds) # Calcula a Matriz de Confusão

    # Imprime no console
    print(f"Total de Amostras de Teste: {len(all_labels)}")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (p/ Shoplifting): {precision:.4f}")
    print(f"Revocação (Recall p/ Shoplifting): {recall:.4f}")
    print(f"F1-Score (p/ Shoplifting): {f1:.4f}")
    print("\nMatriz de Confusão (Linha=Real, Coluna=Previsto):")
    print(cm)

    # --- NOVO: Salva o relatório de métricas em um arquivo .txt ---
    metrics_file_path = OUTPUT_DIR / 'teste_report_metrics.txt'
    try:
        with open(metrics_file_path, 'w') as f:
            f.write("--- Relatório Final de Métricas no Conjunto de Teste ---\n")
            f.write(f"Checkpoint Avaliado: {CHECKPOINT_FILENAME}\n")
            f.write(f"Total de Amostras de Teste: {len(all_labels)}\n\n")
            f.write(f"Acurácia: {accuracy:.4f}\n")
            f.write(f"Precisão (Classe 1 - Shoplifting): {precision:.4f}\n")
            f.write(f"Revocação (Recall Classe 1 - Shoplifting): {recall:.4f}\n")
            f.write(f"F1-Score (Classe 1 - Shoplifting): {f1:.4f}\n\n")
            f.write("--- Matriz de Confusão ---\n")
            f.write(f"Classes (na ordem): {list(class_map.values())}\n")
            f.write(f"(Linhas = Real, Colunas = Previsto)\n")
            f.write(np.array2string(cm))
            f.write("\n\nLegenda da Matriz:\n")
            f.write("[[Verdadeiro Normal,   Falso Shoplifting],\n")
            f.write(" [Falso Normal,        Verdadeiro Shoplifting]]\n")
        
        print(f"\nRelatório de métricas salvo com sucesso em: {metrics_file_path}")
    except Exception as e:
        print(f"ERRO ao salvar arquivo de métricas: {e}")
    # ----------------------------------------------------

    # Salva a imagem da Matriz de Confusão (passando a 'cm' calculada)
    matrix_plot_path = OUTPUT_DIR / 'teste_confusion_matrix.png'
    plot_confusion_matrix(cm, class_map, matrix_plot_path)

if __name__ == '__main__':
    evaluate()