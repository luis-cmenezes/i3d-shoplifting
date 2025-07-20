import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import gc

# Importa as classes que já criamos
from src.models.i3d_pytorch import InceptionI3d
from src.common.dataset import ShopliftingDataset

def try_batch_size(batch_size, device, rgb_dir, flow_dir):
    """
    Tenta executar um único passo de treinamento com um determinado tamanho de lote.
    Retorna True se for bem-sucedido, False se ocorrer um erro de falta de memória.
    """
    print(f"--- Testando BATCH_SIZE = {batch_size}... ", end="")
    
    try:
        # 1. Configuração do modelo e dados (dentro da função para liberar memória depois)
        model_rgb = InceptionI3d(num_classes=2, in_channels=3).to(device)
        model_flow = InceptionI3d(num_classes=2, in_channels=2).to(device)
        
        # Otimizador simples apenas para o teste do passo de gradiente
        optimizer = torch.optim.Adam(list(model_rgb.parameters()) + list(model_flow.parameters()))
        criterion = nn.CrossEntropyLoss()

        # Usamos o dataset completo, pois só vamos carregar um lote
        dataset = ShopliftingDataset(rgb_dir=rgb_dir, flow_dir=flow_dir)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        # Pega um único lote
        rgb_data, flow_data, labels = next(iter(loader))
        rgb_data, flow_data, labels = rgb_data.to(device), flow_data.to(device), labels.to(device)

        # 2. Simula um passo de treinamento completo
        optimizer.zero_grad()
        
        rgb_logits = torch.mean(model_rgb(rgb_data), dim=2)
        flow_logits = torch.mean(model_flow(flow_data), dim=2)
        
        final_logits = (rgb_logits + flow_logits) / 2
        
        loss = criterion(final_logits, labels)
        loss.backward() # O cálculo do gradiente é uma das partes que mais consome memória
        optimizer.step()

        print("OK")
        
        # 3. Limpeza explícita da memória
        del model_rgb, model_flow, optimizer, criterion, dataset, loader, rgb_data, flow_data, labels, loss
        gc.collect()
        torch.cuda.empty_cache()
        
        return True

    except torch.cuda.OutOfMemoryError:
        print("FALHOU (Out of Memory)")
        
        # Limpeza em caso de erro
        gc.collect()
        torch.cuda.empty_cache()
        
        return False
    except Exception as e:
        print(f"FALHOU (Erro inesperado: {e})")
        return False


def find_max_batch_size():
    """
    Encontra o maior tamanho de lote que cabe na memória da GPU.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == 'cpu':
        print("Nenhuma GPU detectada. Não é possível encontrar o batch size máximo para CUDA.")
        return

    print(f"Iniciando busca pelo BATCH_SIZE máximo no dispositivo: {DEVICE}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    print(PROJECT_ROOT)
    RGB_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'rgb'
    FLOW_DIR = PROJECT_ROOT / 'data' / 'i3d_inputs' / 'optical_flow'

    batch_size = 2
    max_safe_batch_size = 1

    while True:
        success = try_batch_size(batch_size, DEVICE, RGB_DIR, FLOW_DIR)
        if success:
            max_safe_batch_size = batch_size
            batch_size *= 2 # Dobra para o próximo teste
        else:
            break # Para a busca ao primeiro erro de memória

    print("\n--- Busca Concluída ---")
    print(f"O maior BATCH_SIZE que funcionou foi: {max_safe_batch_size}")
    print(f"Recomendação: Use um BATCH_SIZE de {max_safe_batch_size} ou um pouco menor no seu script 'train.py' para ter uma margem de segurança.")


if __name__ == '__main__':
    find_max_batch_size()
