import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from natsort import natsorted
import argparse
from pathlib import Path

def generate_flow_for_block(source_block_path, output_block_path):
    """
    Calcula o fluxo ótico denso (TV-L1) para uma sequência de frames RGB
    e salva os componentes u e v como imagens separadas.

    Args:
        source_block_path (str): Caminho para a pasta do bloco com frames RGB.
        output_block_path (str): Caminho para a pasta de saída do fluxo ótico.
    """
    try:
        rgb_frames = sorted([os.path.join(source_block_path, f) for f in os.listdir(source_block_path) if f.endswith('.jpg')])
        if len(rgb_frames) < 2:
            print(f"Aviso: Bloco '{os.path.basename(source_block_path)}' tem menos de 2 frames. Pulando.")
            return
    except FileNotFoundError:
        print(f"Aviso: Diretório do bloco '{source_block_path}' não encontrado. Pulando.")
        return

    os.makedirs(output_block_path, exist_ok=True)

    # Inicializa o algoritmo TV-L1. É mais eficiente criar o objeto fora do loop.
    tvl1 = cv2.optflow.createOptFlow_DualTVL1()

    # Carrega o primeiro frame e converte para escala de cinza
    prev_frame = cv2.imread(rgb_frames[0])
    if prev_frame is None: return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Itera a partir do segundo frame para calcular o fluxo
    for i in range(1, len(rgb_frames)):
        frame = cv2.imread(rgb_frames[i])
        if frame is None: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calcula o fluxo ótico denso
        flow = tvl1.calc(prev_gray, gray, None)

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
        # O nome do arquivo corresponde ao frame final do par (i)
        flow_x_filename = f"flow_x_{i:06d}.jpg"
        flow_y_filename = f"flow_y_{i:06d}.jpg"

        cv2.imwrite(os.path.join(output_block_path, flow_x_filename), u)
        cv2.imwrite(os.path.join(output_block_path, flow_y_filename), v)

        # Atualiza o frame anterior para a próxima iteração
        prev_gray = gray

def main(source_rgb_dir, output_flow_dir):
    """
    Função principal para orquestrar a geração de fluxo ótico.
    """
    print("--- INICIANDO GERAÇÃO DE INPUT DE FLUXO ÓTICO PARA O MODELO I3D ---")
    os.makedirs(output_flow_dir, exist_ok=True)

    try:
        block_folders = natsorted(os.listdir(source_rgb_dir))
        print(block_folders)
    except FileNotFoundError:
        print(f"ERRO: Diretório de input RGB '{source_rgb_dir}' não encontrado. Abortando.")
        return

    for block_name in tqdm(block_folders, desc="Processando Blocos para Fluxo Ótico"):
        source_block_path = os.path.join(source_rgb_dir, block_name)
        if not os.path.isdir(source_block_path):
            continue
        
        output_block_path = os.path.join(output_flow_dir, block_name)
        
        generate_flow_for_block(source_block_path, output_block_path)

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
        "--overwrite",
        action="store_true",
        help="Se setado, apaga o diretório de saída antes de gerar novamente.",
    )
    args = parser.parse_args()

    if args.overwrite and os.path.exists(args.output_flow_dir):
        print(f"Aviso: Limpando diretório de saída '{args.output_flow_dir}' (--overwrite).")
        shutil.rmtree(args.output_flow_dir)

    main(args.source_rgb_dir, args.output_flow_dir)
