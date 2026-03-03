"""
Determinação automática do batch_size máximo para o modelo I3D.

Realiza uma busca binária tentando forward + backward passes com tensores
sintéticos de tamanho crescente até detectar um erro de CUDA OOM.
O batch_size seguro retornado é ``int(0.8 * max_batch_size)``.
"""

import gc
import torch
import torch.nn as nn
from pathlib import Path

from i3d_shoplifting.models.i3d_pytorch import InceptionI3d

# Dimensões padrão usadas pelo pipeline de treino I3D
_NUM_CLASSES = 1
_NUM_FRAMES = 64
_HEIGHT = 224
_WIDTH = 224


def _try_batch_size(
    batch_size: int,
    model_mode: str,
    model_rgb: nn.Module,
    model_flow: nn.Module | None,
    device: torch.device,
) -> bool:
    """Tenta executar um forward + backward com *batch_size* amostras sintéticas.

    Retorna ``True`` se bem-sucedido, ``False`` se ocorrer OOM.
    """
    try:
        # Tensores sintéticos no formato esperado: (B, C, T, H, W)
        rgb_input = torch.randn(
            batch_size, 3, _NUM_FRAMES, _HEIGHT, _WIDTH,
            device=device, dtype=torch.float32,
        )
        labels = torch.randint(
            0, 2, (batch_size, 1),
            device=device, dtype=torch.float32,
        )

        criterion = nn.BCEWithLogitsLoss()

        # Forward RGB
        rgb_logits = model_rgb(rgb_input)

        if model_mode == "rgb_optical" and model_flow is not None:
            flow_input = torch.randn(
                batch_size, 2, _NUM_FRAMES, _HEIGHT, _WIDTH,
                device=device, dtype=torch.float32,
            )
            flow_logits = model_flow(flow_input)
            final_logits = (rgb_logits + flow_logits) / 2
        else:
            final_logits = rgb_logits

        loss = criterion(final_logits, labels)
        loss.backward()

        # Se chegou aqui, batch_size cabe na GPU
        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False
        raise  # Re-lança erros não relacionados a memória
    finally:
        # Limpa tudo para a próxima tentativa
        del rgb_input, labels
        if "flow_input" in dir():
            del flow_input
        gc.collect()
        torch.cuda.empty_cache()


def find_max_batch_size(
    model_mode: str = "rgb_only",
    unfreeze_full_model: bool = False,
    rgb_checkpoint: str = "",
    flow_checkpoint: str = "",
    start_batch_size: int = 128,
    device: torch.device | None = None,
) -> int:
    """Encontra o batch_size máximo que cabe na GPU para o modelo I3D.

    Usa busca binária entre 1 e *start_batch_size*.

    Args:
        model_mode: ``"rgb_only"`` ou ``"rgb_optical"`` (late fusion).
        unfreeze_full_model: Se ``True``, todos os pesos ficam treináveis
            (consome mais memória por causa dos gradientes).
        rgb_checkpoint: Caminho do checkpoint pré-treinado RGB (opcional).
        flow_checkpoint: Caminho do checkpoint pré-treinado de fluxo (opcional).
        start_batch_size: Limite superior inicial para a busca binária.
        device: Dispositivo CUDA. Se ``None``, usa ``cuda:0``.

    Returns:
        Batch_size seguro (``int(0.8 * max_batch_size)``), mínimo 1.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("[find_max_batch_size] Sem GPU detectada — retornando batch_size=1.")
        return 1

    print(f"[find_max_batch_size] Procurando batch_size máximo para I3D "
          f"(mode={model_mode}, unfreeze={unfreeze_full_model}) ...")

    # --- Instancia modelos ---
    model_rgb = InceptionI3d(num_classes=400, in_channels=3)
    if rgb_checkpoint and Path(rgb_checkpoint).exists():
        _load_weights(model_rgb, rgb_checkpoint)
    model_rgb.replace_logits(_NUM_CLASSES)

    if unfreeze_full_model:
        for p in model_rgb.parameters():
            p.requires_grad = True
    else:
        for p in model_rgb.parameters():
            p.requires_grad = False
        for p in model_rgb.logits.parameters():
            p.requires_grad = True

    model_rgb.to(device)
    model_rgb.train()

    model_flow = None
    if model_mode == "rgb_optical":
        model_flow = InceptionI3d(num_classes=400, in_channels=2)
        if flow_checkpoint and Path(flow_checkpoint).exists():
            _load_weights(model_flow, flow_checkpoint)
        model_flow.replace_logits(_NUM_CLASSES)

        if unfreeze_full_model:
            for p in model_flow.parameters():
                p.requires_grad = True
        else:
            for p in model_flow.parameters():
                p.requires_grad = False
            for p in model_flow.logits.parameters():
                p.requires_grad = True

        model_flow.to(device)
        model_flow.train()

    # --- Busca binária ---
    low, high = 1, start_batch_size
    max_ok = 0  # Maior batch_size que passou

    while low <= high:
        mid = (low + high) // 2
        print(f"  Tentando batch_size={mid} ...", end=" ", flush=True)

        # Zera gradientes antes de testar
        model_rgb.zero_grad()
        if model_flow is not None:
            model_flow.zero_grad()

        success = _try_batch_size(mid, model_mode, model_rgb, model_flow, device)

        if success:
            print("OK")
            max_ok = mid
            low = mid + 1
        else:
            print("OOM")
            high = mid - 1

    # --- Cleanup completo ---
    del model_rgb
    if model_flow is not None:
        del model_flow
    gc.collect()
    torch.cuda.empty_cache()

    safe_batch_size = max(1, int(0.8 * max_ok))
    print(f"[find_max_batch_size] max_batch_size={max_ok} → "
          f"safe batch_size (×0.8) = {safe_batch_size}")
    return safe_batch_size


def _load_weights(model: nn.Module, checkpoint_path: str) -> None:
    """Carrega pesos pré-treinados ignorando camadas incompatíveis."""
    try:
        pretrained = torch.load(checkpoint_path, map_location="cpu")
        model_dict = model.state_dict()
        filtered = {
            k: v for k, v in pretrained.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
    except Exception as e:
        print(f"  Aviso: não foi possível carregar '{checkpoint_path}': {e}")
