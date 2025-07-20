import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from pathlib import Path
import os, re, csv

# Importa as classes que criamos
from src.models.i3d_pytorch import InceptionI3d
from src.common.dataset import ShopliftingDataset

# --- 1. CONFIGURAÇÃO E HIPERPARÂMETROS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Caminhos para os dados e checkpoints
RGB_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'rgb'
FLOW_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'optical_flow'
RGB_CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'pretrained' /'rgb_imagenet.pt'
FLOW_CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'pretrained' /'flow_imagenet.pt'
OUTPUT_CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints' / 'testing'
LOG_FILE_PATH = OUTPUT_CHECKPOINT_DIR / 'training_log.csv'

# Hiperparâmetros de treinamento
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
EPOCHS = 1
NUM_CLASSES = 2 # Shoplifting (1) e Normal (0)

def load_pretrained_weights(model, checkpoint_path):
    """Carrega pesos pré-treinados, ignorando a camada de logits se as dimensões não baterem."""
    try:
        pretrained_dict = torch.load(checkpoint_path)
        model_dict = model.state_dict()
        
        # Filtra os pesos da camada de logits se ela for diferente
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Carregados {len(pretrained_dict)} tensores de pesos de '{checkpoint_path}'")
    except Exception as e:
        print(f"Erro ao carregar o checkpoint '{checkpoint_path}': {e}")
        print("Continuando com pesos inicializados aleatoriamente.")

def manage_top_checkpoints(checkpoint_dir, current_recall, epoch, model_rgb, model_flow, optimizer):
    """Mantém os 3 melhores checkpoints com base na revocação."""
    
    # Encontra os checkpoints existentes e seus scores de recall
    existing_checkpoints = []
    for f in os.listdir(checkpoint_dir):
        match = re.match(r'epoch_(\d+)_recall_(\d\.\d+)\.pt', f)
        if match:
            existing_checkpoints.append((float(match.group(2)), f))
            
    # Ordena do pior para o melhor
    existing_checkpoints.sort(key=lambda x: x[0])
    
    # Verifica se o recall atual é melhor que o pior dos top 3, ou se há menos de 3
    if len(existing_checkpoints) < 3 or current_recall > existing_checkpoints[0][0]:
        # Salva o novo checkpoint
        checkpoint_name = f"epoch_{epoch+1:02d}_recall_{current_recall:.4f}.pt"
        save_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        print(f"Novo melhor Recall de validação: {current_recall:.4f}. Salvando modelo em '{checkpoint_name}'...")
        torch.save({
            'epoch': epoch,
            'model_rgb_state_dict': model_rgb.state_dict(),
            'model_flow_state_dict': model_flow.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'recall': current_recall
        }, save_path)
        
        # Se agora temos mais de 3 checkpoints, remove o pior
        if len(existing_checkpoints) >= 3:
            worst_checkpoint_path = os.path.join(checkpoint_dir, existing_checkpoints[0][1])
            print(f"Removendo o pior checkpoint antigo: '{existing_checkpoints[0][1]}'")
            os.remove(worst_checkpoint_path)

def log_metrics(epoch, train_loss, val_metrics):
    """Salva as métricas da época em um arquivo CSV."""
    file_exists = os.path.isfile(LOG_FILE_PATH)
    with open(LOG_FILE_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1'])
        
        writer.writerow([
            epoch + 1,
            train_loss,
            val_metrics['accuracy'],
            val_metrics['precision'],
            val_metrics['recall'],
            val_metrics['f1']
        ])

def train():
    """Função principal que orquestra todo o processo de treinamento."""
    print(f"Usando dispositivo: {DEVICE}")

    # --- 2. PREPARAÇÃO DO DATASET E DIVISÃO ---
    full_dataset = ShopliftingDataset(rgb_dir=RGB_DIR, flow_dir=FLOW_DIR)    
    indices = list(range(len(full_dataset)))
    labels = []
    for idx in indices:
        labels.append(full_dataset.get_label(idx))

    # Divide em 70% treino, 15% validação, 15% teste
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, labels, test_size=0.3, random_state=42, stratify=labels)
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Tamanho do dataset: {len(full_dataset)} amostras - \
         {sum(1 for label in labels if label == 1)} Shoplifting | {sum(1 for label in labels if label == 0)} Normal")
    print(f"Treino: {len(train_dataset)} amostras - \
          {sum(1 for label in train_labels if label == 1)} Shoplifting | {sum(1 for label in train_labels if label == 0)} Normal")
    print(f"Validação: {len(val_dataset)} amostras - \
          {sum(1 for label in val_labels if label == 1)} Shoplifting | {sum(1 for label in val_labels if label == 0)} Normal")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. INICIALIZAÇÃO DO MODELO ---
    # Modelo RGB
    model_rgb = InceptionI3d(num_classes=400, in_channels=3)
    load_pretrained_weights(model_rgb, RGB_CHECKPOINT)
    model_rgb.replace_logits(NUM_CLASSES)

    # Modelo de Fluxo Ótico
    model_flow = InceptionI3d(num_classes=400, in_channels=2)
    load_pretrained_weights(model_flow, FLOW_CHECKPOINT)
    model_flow.replace_logits(NUM_CLASSES)

    # Congela todas camadas com exceção de classificador
    for param in model_rgb.parameters(): param.requires_grad = False
    for param in model_flow.parameters(): param.requires_grad = False
    for param in model_rgb.logits.parameters(): param.requires_grad = True
    for param in model_flow.logits.parameters(): param.requires_grad = True

    model_rgb.to(DEVICE)
    model_flow.to(DEVICE)

    # --- 4. OTIMIZADOR E FUNÇÃO DE PERDA ---
    trainable_params = list(model_rgb.logits.parameters()) + list(model_flow.logits.parameters())
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
    
    # Calcula os pesos das classes para a perda
    num_normal = sum(1 for label in train_labels if label == 0)
    num_shoplifting = sum(1 for label in train_labels if label == 1)
    weight_shoplifting = num_normal / num_shoplifting
    class_weights = torch.tensor([1.0, weight_shoplifting], device=DEVICE)
    print(f"Pesos das classes: Normal=1.0, Shoplifting={weight_shoplifting:.2f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- 5. LOOP DE TREINAMENTO E VALIDAÇÃO ---
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        model_rgb.train()
        model_flow.train()
        running_loss = 0.0
        
        for rgb_data, flow_data, labels in tqdm(train_loader, desc="Treinando"):
            rgb_data, flow_data, labels = rgb_data.to(DEVICE), flow_data.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            final_logits = (torch.mean(model_rgb(rgb_data), dim=2) + torch.mean(model_flow(flow_data), dim=2)) / 2

            loss = criterion(final_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)

        print(f"Epoch train loss: {epoch_train_loss:.4f}")

        # Validação
        model_rgb.eval()
        model_flow.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for rgb_data, flow_data, labels in tqdm(val_loader, desc="Validando"):
                rgb_data, flow_data, labels = rgb_data.to(DEVICE), flow_data.to(DEVICE), labels.to(DEVICE)
                final_logits = (torch.mean(model_rgb(rgb_data), dim=2) + torch.mean(model_flow(flow_data), dim=2)) / 2
                preds = torch.argmax(final_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0)
        }

        
        print(f"Validação - Acurácia: {val_metrics['accuracy']:.4f}, Precisão: {val_metrics['precision']:.4f}, Revocação: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        manage_top_checkpoints(OUTPUT_CHECKPOINT_DIR, val_metrics['recall'], epoch, model_rgb, model_flow, optimizer)
        log_metrics(epoch, epoch_train_loss, val_metrics)

if __name__ == '__main__':
    train()

