import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from natsort import natsorted
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------------------------
# Detecção de backend: CUDA (GPU) vs CPU
# ---------------------------------------------------------------------------

def _has_cuda_optflow() -> bool:
    """Retorna True se o OpenCV foi compilado com suporte a CUDA optical flow."""
    try:
        obj = cv2.cuda.OpticalFlowDual_TVL1.create()  # type: ignore[attr-defined]
        return obj is not None
    except (AttributeError, cv2.error):
        return False

_USE_CUDA = _has_cuda_optflow()


def _create_tvl1():
    """Cria o estimador TV-L1 usando GPU (CUDA) ou CPU."""
    if _USE_CUDA:
        return cv2.cuda.OpticalFlowDual_TVL1.create()  # type: ignore[attr-defined]
    return cv2.optflow.createOptFlow_DualTVL1()


def _calc_flow(tvl1, prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Calcula fluxo ótico usando GPU ou CPU de forma transparente."""
    if _USE_CUDA:
        gpu_prev = cv2.cuda_GpuMat()
        gpu_curr = cv2.cuda_GpuMat()
        gpu_prev.upload(prev_gray)
        gpu_curr.upload(gray)
        gpu_flow = tvl1.calc(gpu_prev, gpu_curr, None)  # type: ignore[arg-type]
        return gpu_flow.download()
    return tvl1.calc(prev_gray, gray, None)


# ---------------------------------------------------------------------------
# Processamento de um bloco individual
# ---------------------------------------------------------------------------

def generate_flow_for_block(source_block_path: str, output_block_path: str) -> str | None:
    """
    Calcula o fluxo ótico denso (TV-L1) para uma sequência de frames RGB
    e salva os componentes u e v como imagens separadas.

    Retorna o nome do bloco em caso de erro/skip, ou None se tudo ok.
    """
    block_name = os.path.basename(source_block_path)

    try:
        rgb_frames = sorted(
            os.path.join(source_block_path, f)
            for f in os.listdir(source_block_path)
            if f.endswith('.jpg')
        )
        if len(rgb_frames) < 2:
            return f"{block_name} (<2 frames)"
    except FileNotFoundError:
        return f"{block_name} (não encontrado)"

    # Pula se o bloco já foi processado (número esperado de arquivos de flow)
    expected_flow_files = (len(rgb_frames) - 1) * 2  # u + v por par
    if os.path.isdir(output_block_path):
        existing = sum(1 for f in os.listdir(output_block_path) if f.endswith('.jpg'))
        if existing >= expected_flow_files:
            return None  # já concluído

    os.makedirs(output_block_path, exist_ok=True)

    # Inicializa o algoritmo TV-L1 (um por processo/worker)
    tvl1 = _create_tvl1()

    # Carrega o primeiro frame e converte para escala de cinza
    prev_frame = cv2.imread(rgb_frames[0])
    if prev_frame is None:
        return f"{block_name} (frame 0 inválido)"
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Itera a partir do segundo frame para calcular o fluxo
    for i in range(1, len(rgb_frames)):
        frame = cv2.imread(rgb_frames[i])
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calcula o fluxo ótico denso
        flow = _calc_flow(tvl1, prev_gray, gray)

        # Separa os componentes horizontal (u) e vertical (v)
        u, v = cv2.split(flow)

        # --- Normalização para o intervalo 0-255 ---
        # Limita os valores para evitar que movimentos extremos dominem a imagem
        u = np.clip(u, -20, 20)
        v = np.clip(v, -20, 20)

        # Normaliza os valores para o intervalo de 8 bits (0-255)
        u = cv2.normalize(u, None, 0, 255, cv2.NORM_MINMAX)
        v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)

        # Converte para o tipo de dados de imagem padrão
        u = u.astype(np.uint8)
        v = v.astype(np.uint8)

        # --- Salvamento dos frames de fluxo ---
        flow_x_filename = f"flow_x_{i:06d}.jpg"
        flow_y_filename = f"flow_y_{i:06d}.jpg"

        cv2.imwrite(os.path.join(output_block_path, flow_x_filename), u)
        cv2.imwrite(os.path.join(output_block_path, flow_y_filename), v)

        # Atualiza o frame anterior para a próxima iteração
        prev_gray = gray

    return None


def _worker(args: tuple[str, str]) -> str | None:
    """Wrapper para multiprocessing — desempacota a tupla de argumentos."""
    return generate_flow_for_block(args[0], args[1])


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

def main(source_rgb_dir: str, output_flow_dir: str, num_workers: int = 0) -> None:
    """
    Gera fluxo ótico para todos os blocos em paralelo.

    Args:
        source_rgb_dir:  Diretório com inputs RGB já processados.
        output_flow_dir: Diretório de saída para o fluxo ótico.
        num_workers:     Número de processos paralelos.
                         0 = automático (cpu_count, ou 1 se usando CUDA).
    """
    backend = "CUDA (GPU)" if _USE_CUDA else "CPU"
    print("--- INICIANDO GERAÇÃO DE INPUT DE FLUXO ÓTICO PARA O MODELO I3D ---")
    print(f"    Backend: {backend}")

    os.makedirs(output_flow_dir, exist_ok=True)

    try:
        block_folders = natsorted(os.listdir(source_rgb_dir))
    except FileNotFoundError:
        print(f"ERRO: Diretório de input RGB '{source_rgb_dir}' não encontrado. Abortando.")
        return

    # Filtra apenas diretórios válidos
    work_items: list[tuple[str, str]] = []
    for block_name in block_folders:
        src = os.path.join(source_rgb_dir, block_name)
        if os.path.isdir(src):
            work_items.append((src, os.path.join(output_flow_dir, block_name)))

    if not work_items:
        print("Nenhum bloco encontrado para processar.")
        return

    # Determina o número de workers
    if num_workers <= 0:
        if _USE_CUDA:
            # GPU: um único processo (a GPU já paraleliza internamente)
            num_workers = 1
        else:
            # CPU: usa todos os cores disponíveis
            num_workers = cpu_count() or 1

    print(f"    Workers: {num_workers}")
    print(f"    Blocos:  {len(work_items)}\n")

    if num_workers == 1:
        # Execução sequencial — sem overhead de fork
        skipped = []
        for item in tqdm(work_items, desc="Processando Blocos para Fluxo Ótico"):
            result = _worker(item)
            if result:
                skipped.append(result)
    else:
        # Execução paralela com multiprocessing
        skipped = []
        with Pool(processes=num_workers) as pool:
            results = pool.imap_unordered(_worker, work_items, chunksize=4)
            for result in tqdm(results, total=len(work_items), desc="Processando Blocos para Fluxo Ótico"):
                if result:
                    skipped.append(result)

    if skipped:
        print(f"\n  Blocos pulados/com erro ({len(skipped)}):")
        for s in skipped[:20]:
            print(f"    - {s}")
        if len(skipped) > 20:
            print(f"    ... e mais {len(skipped) - 20}")

    print("\n--- PROCESSO CONCLUÍDO ---")
    print(f"Os inputs de fluxo ótico foram gerados com sucesso em: '{os.path.abspath(output_flow_dir)}'")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="Gera inputs de fluxo ótico (I3D) a partir dos inputs RGB.")
    parser.add_argument(
        "--source-rgb-dir",
        type=str,
        default=str(project_root / "data" / "i3d_inputs" / "rgb"),
        help="Diretório com inputs RGB já processados (64 frames 224x224).",
    )
    parser.add_argument(
        "--output-flow-dir",
        type=str,
        default=str(project_root / "data" / "i3d_inputs" / "optical_flow"),
        help="Diretório de saída para os inputs de fluxo ótico.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Número de processos paralelos (0 = automático: todos os cores CPU, ou 1 se GPU).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Se setado, apaga o diretório de saída antes de gerar novamente.",
    )
    args = parser.parse_args()

    if args.overwrite and os.path.exists(args.output_flow_dir):
        print(f"Aviso: Limpando diretório de saída '{args.output_flow_dir}' (--overwrite).")
        shutil.rmtree(args.output_flow_dir)

    main(args.source_rgb_dir, args.output_flow_dir, num_workers=args.workers)
