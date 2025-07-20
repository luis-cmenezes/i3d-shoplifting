import torch
from pathlib import Path

# Importação do nosso modelo corrigido
from models.i3d_pytorch import InceptionI3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_test():
    """Encapsula a lógica de teste em uma função."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Dispositivo selecionado: {device} ---")

    # --- TESTANDO O STREAM RGB ---
    print("--- Testando o modelo RGB ---")
    i3d_rgb = InceptionI3d(num_classes=400, in_channels=3)

    checkpoint_path_rgb = PROJECT_ROOT / 'checkpoints' / 'rgb_imagenet.pt'
    print(f"Carregando pesos de: {checkpoint_path_rgb}")
    i3d_rgb.load_state_dict(torch.load(checkpoint_path_rgb))
    print("Pesos RGB carregados com sucesso!")

    i3d_rgb.to(device)
    print(f"Modelo RGB movido para o dispositivo: {device}")

    dummy_input_rgb = torch.randn(1, 3, 64, 224, 224).to(device)
    output_logits_temporal = i3d_rgb(dummy_input_rgb)

    print(f"Formato da saída temporal do modelo RGB: {output_logits_temporal.shape}")
    # A saída esperada aqui é torch.Size([1, 400, 7])

    # --- INÍCIO DA CORREÇÃO ---
    # Tira a média das previsões ao longo da dimensão temporal (dim=2)
    output_logits = torch.mean(output_logits_temporal, dim=2)
    # --- FIM DA CORREÇÃO ---

    print(f"Formato da saída após média temporal: {output_logits.shape}")
    # A saída esperada agora é torch.Size([1, 400])
    assert output_logits.shape == (1, 400)
    print("Teste do modelo RGB bem-sucedido!\n")

    # --- TESTANDO O STREAM DE FLUXO ÓTICO ---
    print("--- Testando o modelo de Fluxo Ótico ---")
    i3d_flow = InceptionI3d(num_classes=400, in_channels=2)

    checkpoint_path_flow = PROJECT_ROOT / 'checkpoints' / 'flow_imagenet.pt'
    print(f"Carregando pesos de: {checkpoint_path_flow}")
    i3d_flow.load_state_dict(torch.load(checkpoint_path_flow))
    print("Pesos de Fluxo Ótico carregados com sucesso!")

    i3d_flow.to(device)
    print(f"Modelo de Fluxo Ótico movido para o dispositivo: {device}")

    dummy_input_flow = torch.randn(1, 2, 64, 224, 224).to(device)
    output_logits_flow_temporal = i3d_flow(dummy_input_flow)
    
    print(f"Formato da saída temporal do modelo de Fluxo: {output_logits_flow_temporal.shape}")

    # --- APLICA A MESMA CORREÇÃO AQUI ---
    output_logits_flow = torch.mean(output_logits_flow_temporal, dim=2)
    
    print(f"Formato da saída do Fluxo após média temporal: {output_logits_flow.shape}")
    assert output_logits_flow.shape == (1, 400)
    print("Teste do modelo de Fluxo Ótico bem-sucedido!")

if __name__ == '__main__':
    run_test()
