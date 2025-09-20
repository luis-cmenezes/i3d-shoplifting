import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tqdm import tqdm
from pathlib import Path
import os, re, csv
import numpy as np
import cv2
import random
from pathlib import Path

# Importa as classes que criamos
from src.models.i3d_pytorch import InceptionI3d
from src.common.dataset import ShopliftingDataset, VideoAugmentation

# --- 1. CONFIGURAÇÃO E HIPERPARÂMETROS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Flag para controlar o modo do modelo. Opções: 'rgb_optical', 'rgb_only'
MODEL_MODE = 'rgb_optical' 
# Flag para descongelar todos os pesos do modelo para um fine-tuning completo
UNFREEZE_FULL_MODEL = True
TEST_NAME = f'aug_{"full_unfreeze" if UNFREEZE_FULL_MODEL else "head_unfreeze"}_{MODEL_MODE}'

# Caminhos para os dados e checkpoints
RGB_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'rgb'
FLOW_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'optical_flow'
RGB_CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'pretrained' /'rgb_imagenet.pt'
FLOW_CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'pretrained' /'flow_imagenet.pt'

# --- ESTRUTURA DE SAÍDA REFINADA ---
BASE_OUTPUT_DIR = PROJECT_ROOT / 'checkpoints' / TEST_NAME
CHECKPOINT_SAVE_DIR = BASE_OUTPUT_DIR / 'model_weights'
VISUALIZATION_DIR = BASE_OUTPUT_DIR / 'input_visualizations'
ROC_CURVE_DIR = BASE_OUTPUT_DIR / 'roc_curves'
LOG_FILE_PATH = BASE_OUTPUT_DIR / 'training_log.csv'

CHECKPOINT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
ROC_CURVE_DIR.mkdir(parents=True, exist_ok=True)

# Hiperparâmetros de treinamento
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
EPOCHS = 70
NUM_CLASSES = 1 # A saída única representa a probabilidade de ser Shoplifting
STEPS_TO_VISUALIZE_PER_EPOCH = 2

def save_video_comparison(rgb_before_batch, rgb_after_batch, label_batch, output_path, fps=10):
    """
    Salva um vídeo .mp4 comparando os clipes RGB (Antes e Depois) para TODOS os itens em um batch.
    
    O vídeo final será um stack vertical (uma grade) de todas as comparações 
    lado a lado de cada item no batch.
    """
    try:
        rgb_before_batch = rgb_before_batch.cpu()
        rgb_after_batch = rgb_after_batch.cpu()
        label_batch = label_batch.cpu()

        # Extrai as dimensões do batch de entrada
        # rgb_before_batch tem shape (B, C, T, H, W)
        B, C, num_frames, H, W = rgb_before_batch.shape

        # Define as dimensões do vídeo de SAÍDA:
        # Largura = Dobro da largura original (para [Antes | Depois])
        # Altura = Altura original * Número de itens no batch (para empilhar verticalmente)
        output_width = W * 2
        output_height = H * B
        
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color_white = (255, 255, 255) # Branco
        line_type = 2

        # Itera por cada TIMESTEP (frame) do vídeo
        for t in range(num_frames):
            
            # Lista para armazenar os frames de comparação de cada item do batch (para este timestep t)
            frames_do_batch_para_empilhar = []
            
            # Itera por cada ITEM 'i' dentro do batch
            for i in range(B):
                
                # --- Configuração do Rótulo para o item 'i' ---
                label_int = label_batch[i].item()
                if label_int == 1:
                    text_label = "LABEL: SHOPLIFTING"
                    label_color = (0, 0, 255) # Vermelho em BGR
                else:
                    text_label = "LABEL: NORMAL"
                    label_color = (0, 255, 0) # Verde em BGR

                # --- Processa o Frame "Antes" (Batch item i, tempo t) ---
                frame_b_tensor = rgb_before_batch[i, :, t, :, :] # (C, H, W)
                frame_b_np = (frame_b_tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
                frame_b_bgr = cv2.cvtColor(frame_b_np, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_b_bgr, 'Before (Original)', (10, 25), font, font_scale, font_color_white, line_type)

                # --- Processa o Frame "Depois" (Batch item i, tempo t) ---
                frame_a_tensor = rgb_after_batch[i, :, t, :, :] # (C, H, W)
                frame_a_np = (frame_a_tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
                frame_a_bgr = cv2.cvtColor(frame_a_np, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_a_bgr, 'After (Augmented)', (10, 25), font, font_scale, font_color_white, line_type)

                # Concatena horizontalmente [Antes | Depois] para o item 'i'
                comparison_frame_i = np.concatenate((frame_b_bgr, frame_a_bgr), axis=1) # Shape (H, W*2, C)

                # Adiciona o rótulo e o índice do batch neste frame de comparação
                cv2.putText(comparison_frame_i, text_label, (10, H - 20), font, font_scale, label_color, line_type, cv2.LINE_AA)
                cv2.putText(comparison_frame_i, f"Batch Item: {i}", (W*2 - 200, H - 20), font, font_scale, font_color_white, line_type)
                
                # Adiciona este frame de comparação (H, W*2, C) à nossa lista
                frames_do_batch_para_empilhar.append(comparison_frame_i)
            
            # --- Montagem do Frame Final ---
            # Após processar todos os itens do batch (i=0 até B-1) para o tempo 't',
            # empilhamos todos os frames de comparação verticalmente.
            # (axis=0 faz o empilhamento vertical)
            final_frame_t = np.concatenate(frames_do_batch_para_empilhar, axis=0) # Shape final: (H*B, W*2, C)

            # Escreve o frame "alto" combinado no vídeo
            out.write(final_frame_t)

        out.release()
    except Exception as e:
        print(f" (!) Erro ao salvar vídeo de comparação em '{output_path}': {e}")
        
def load_pretrained_weights(model, checkpoint_path):
    """Carrega pesos pré-treinados, ignorando a camada de logits se as dimensões não baterem."""
    try:
        pretrained_dict = torch.load(checkpoint_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Carregados {len(pretrained_dict)} tensores de pesos de '{checkpoint_path}'")
    except Exception as e:
        print(f"Erro ao carregar o checkpoint '{checkpoint_path}': {e}. Continuando com pesos aleatórios.")

def manage_top_checkpoints(checkpoint_dir, current_auc_roc, epoch, model_rgb, model_flow, optimizer, model_mode):
    """Mantém os 3 melhores checkpoints (baseados em AUC-ROC) no diretório."""
    existing_checkpoints = []
    pattern = re.compile(r'epoch_(\d+)_auc_(\d\.\d+)\.pt')
    for f in os.listdir(checkpoint_dir):
        match = pattern.match(f)
        if match:
            existing_checkpoints.append((float(match.group(2)), f))
            
    existing_checkpoints.sort(key=lambda x: x[0])
    
    if len(existing_checkpoints) < 3 or current_auc_roc > existing_checkpoints[0][0]:
        checkpoint_name = f"epoch_{epoch+1:03d}_auc_{current_auc_roc:.4f}.pt"
        save_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        print(f"Novo melhor AUC-ROC de validação: {current_auc_roc:.4f}. Salvando modelo em '{checkpoint_name}'...")
        model_flow_state = model_flow.state_dict() if model_mode == 'rgb_optical' else None
        
        torch.save({
            'epoch': epoch,
            'model_rgb_state_dict': model_rgb.state_dict(),
            'model_flow_state_dict': model_flow_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'auc_roc': current_auc_roc
        }, save_path)

        if len(existing_checkpoints) >= 3:
            worst_checkpoint_path = os.path.join(checkpoint_dir, existing_checkpoints[0][1])
            print(f"Removendo o pior checkpoint antigo: '{existing_checkpoints[0][1]}'")
            os.remove(worst_checkpoint_path)

def log_metrics(epoch, train_loss, val_metrics):
    """Salva as métricas da época no arquivo CSV global 'LOG_FILE_PATH'."""
    file_exists = os.path.isfile(LOG_FILE_PATH)
    with open(LOG_FILE_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_auc_roc'])
        
        writer.writerow([
            epoch + 1,
            train_loss,
            val_metrics['accuracy'],
            val_metrics['precision'],
            val_metrics['recall'],
            val_metrics['f1'],
            val_metrics['auc_roc']
        ])

def train():
    """Função principal que orquestra todo o processo de treinamento e visualização."""
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Todos os outputs serão salvos em: {BASE_OUTPUT_DIR}")

    # --- 2. PREPARAÇÃO DO DATASET E DIVISÃO ---
    transform_train = VideoAugmentation(
        p_flip=0.5,
        color_jitter_params=dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    )

    dataset_with_aug = ShopliftingDataset(rgb_dir=RGB_DIR, flow_dir=FLOW_DIR, transform=transform_train)
    dataset_no_aug = ShopliftingDataset(rgb_dir=RGB_DIR, flow_dir=FLOW_DIR, transform=None)

    indices = list(range(len(dataset_with_aug)))
    labels = [dataset_with_aug.get_label(idx) for idx in indices]

    # Divide em 70% treino, 15% validação, 15% teste
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, labels, test_size=0.3, random_state=42, stratify=labels)
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    # Cria Subsets apontando para os Datasets corretos
    train_dataset = Subset(dataset_with_aug, train_indices)
    val_dataset = Subset(dataset_no_aug, val_indices)

    print(f"Tamanho do dataset: {len(indices)} amostras - {sum(labels)} Shoplifting | {len(labels) - sum(labels)} Normal")
    print(f"Treino: {len(train_dataset)} amostras - {sum(train_labels)} Shoplifting | {len(train_labels) - sum(train_labels)} Normal")
    print(f"Validação: {len(val_dataset)} amostras - {sum(val_labels)} Shoplifting | {len(val_labels) - sum(val_labels)} Normal")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Seleciona os steps (índices de batch) que iremos salvar para visualização
    num_train_steps = len(train_loader)
    steps_to_visualize = random.sample(
        range(num_train_steps), 
        min(STEPS_TO_VISUALIZE_PER_EPOCH, num_train_steps) # Garante que não tentemos amostrar mais do que existe
    )
    print(f"Visualizações de treino serão salvas nos steps (batches): {steps_to_visualize}")

    # --- 3. INICIALIZAÇÃO DO MODELO ---
    model_rgb = InceptionI3d(num_classes=400, in_channels=3)
    load_pretrained_weights(model_rgb, RGB_CHECKPOINT)
    model_rgb.replace_logits(NUM_CLASSES)
    model_rgb.to(DEVICE)
    
    model_flow = None
    if MODEL_MODE == 'rgb_optical':
        model_flow = InceptionI3d(num_classes=400, in_channels=2)
        load_pretrained_weights(model_flow, FLOW_CHECKPOINT)
        model_flow.replace_logits(NUM_CLASSES)
        model_flow.to(DEVICE)

    # Lógica de congelamento/descongelamento baseada em Flag
    if UNFREEZE_FULL_MODEL:
        print("Modo de Treino: Fine-tuning COMPLETO (todos os pesos descongelados).")
        for param in model_rgb.parameters():
            param.requires_grad = True
        if MODEL_MODE == 'rgb_optical':
            for param in model_flow.parameters():
                param.requires_grad = True
    else:
        print("Modo de Treino: Apenas CABEÇA de classificação (logits descongelados).")
        for param in model_rgb.parameters():
            param.requires_grad = False
        for param in model_rgb.logits.parameters():
            param.requires_grad = True
        
        if MODEL_MODE == 'rgb_optical':
            for param in model_flow.parameters():
                param.requires_grad = False
            for param in model_flow.logits.parameters():
                param.requires_grad = True

    model_rgb.to(DEVICE)
    if MODEL_MODE == 'rgb_optical':
        model_flow.to(DEVICE)

    # --- 4. OTIMIZADOR E FUNÇÃO DE PERDA ---
    trainable_params = [p for p in model_rgb.parameters() if p.requires_grad]
    if MODEL_MODE == 'rgb_optical':
        trainable_params.extend([p for p in model_flow.parameters() if p.requires_grad])
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
    
    # Calcula o peso para a classe positiva (Shoplifting) para a BCEWithLogitsLoss
    num_normal = len(train_labels) - sum(train_labels)
    num_shoplifting = sum(train_labels)
    weight_shoplifting = num_normal / num_shoplifting
    pos_weight = torch.tensor([weight_shoplifting], device=DEVICE)
    print(f"Peso da classe positiva (Shoplifting): {weight_shoplifting:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- 5. LOOP DE TREINAMENTO E VALIDAÇÃO ---
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        epoch_viz_dir = VISUALIZATION_DIR / f'epoch_{epoch+1:03d}'
        epoch_viz_dir.mkdir(exist_ok=True)

        model_rgb.train()
        if MODEL_MODE == 'rgb_optical':
            model_flow.train()

        running_loss = 0.0
        
        for step_idx, (rgb_before, flow_before, rgb_after, flow_after, labels) in enumerate(tqdm(train_loader, desc="Treinando")):
            rgb_after, labels = rgb_after.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            
            if MODEL_MODE == 'rgb_optical':
                flow_after = flow_after.to(DEVICE)
            
            if step_idx in steps_to_visualize:
                save_path = epoch_viz_dir / f'step_{step_idx:04d}_rgb_comparison.mp4'
                save_video_comparison(rgb_before, rgb_after, labels, save_path)
            
            optimizer.zero_grad()
            
            rgb_logits = model_rgb(rgb_after)
            if MODEL_MODE == 'rgb_optical':
                flow_logits = model_flow(flow_after)
                final_logits = (rgb_logits + flow_logits) / 2
            else: # 'rgb_only'
                final_logits = rgb_logits

            loss = criterion(final_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)
        print(f"Epoch train loss: {epoch_train_loss:.4f}")

        # Validação
        model_rgb.eval()
        if MODEL_MODE == 'rgb_optical':
            model_flow.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for rgb_before, flow_before, rgb_after, flow_after, labels in tqdm(val_loader, desc="Validando"):
                rgb_after, labels = rgb_after.to(DEVICE), labels.to(DEVICE)
                if MODEL_MODE == 'rgb_optical':
                    flow_after = flow_after.to(DEVICE)
                
                rgb_logits = model_rgb(rgb_after)
                if MODEL_MODE == 'rgb_optical':
                    flow_logits = model_flow(flow_after)
                    final_logits = (rgb_logits + flow_logits) / 2
                else: # 'rgb_only'
                    final_logits = rgb_logits

                probs = torch.sigmoid(final_logits)
                preds = (probs > 0.5).long().squeeze(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.squeeze(1).cpu().numpy())
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        # Salva dados da curva ROC na pasta organizada
        roc_data_path = ROC_CURVE_DIR / f'roc_data_epoch_{epoch + 1:03d}.npz'
        np.savez(roc_data_path, fpr=fpr, tpr=tpr, roc_auc=roc_auc)

        val_metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc_roc': roc_auc
        }
        
        print(f"Validação - Acurácia: {val_metrics['accuracy']:.4f}, Precisão: {val_metrics['precision']:.4f}, Revocação: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, AUC-ROC: {val_metrics['auc_roc']:.4f}")
        manage_top_checkpoints(CHECKPOINT_SAVE_DIR, val_metrics['auc_roc'], epoch, model_rgb, model_flow, optimizer, MODEL_MODE)
        log_metrics(epoch, epoch_train_loss, val_metrics)

if __name__ == '__main__':
    train()

