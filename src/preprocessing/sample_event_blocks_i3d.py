import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import shutil

def process_frame_for_i3d(frame_path, output_size=224):
    """
    Carrega um frame, redimensiona a menor dimensão para 256,
    e faz um corte central para o tamanho de saída desejado.

    Args:
        frame_path (str): Caminho para o arquivo de imagem do frame.
        output_size (int): O tamanho final da imagem (largura e altura).

    Returns:
        numpy.ndarray: O frame processado, ou None se ocorrer um erro.
    """
    # Carrega a imagem
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Aviso: Não foi possível ler o frame em {frame_path}")
        return None

    height, width, _ = frame.shape

    # --- Redimensionamento mantendo a proporção ---
    # Redimensiona a menor dimensão para 256
    scale = 256 / min(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # --- Corte Central (Center Crop) ---
    start_x = (new_width - output_size) // 2
    start_y = (new_height - output_size) // 2
    
    cropped_frame = resized_frame[start_y:start_y + output_size, start_x:start_x + output_size]

    return cropped_frame

def sample_and_process_block(block_path, output_path, num_frames_to_sample=64):
    """
    Aplica a amostragem por segmentos a um bloco de evento e processa
    os frames selecionados.

    Args:
        block_path (str): Caminho para a pasta do bloco de evento.
        output_path (str): Caminho para a pasta de saída do bloco amostrado.
        num_frames_to_sample (int): O número de frames a serem amostrados (64 para I3D).
    """
    # Lista e ordena todos os frames disponíveis no bloco
    try:
        available_frames = sorted([os.path.join(block_path, f) for f in os.listdir(block_path) if f.endswith('.jpg')])
        total_frames = len(available_frames)
        if total_frames == 0:
            print(f"Aviso: Bloco '{os.path.basename(block_path)}' está vazio. Pulando.")
            return
    except FileNotFoundError:
        print(f"Aviso: Diretório do bloco '{block_path}' não encontrado. Pulando.")
        return

    # --- Lógica de Amostragem por Segmentos ---
    sampled_frame_paths = []
    
    # Divide a lista de frames em 'num_frames_to_sample' segmentos
    segment_indices = np.linspace(0, total_frames, num_frames_to_sample + 1, dtype=int)

    for i in range(num_frames_to_sample):
        start = segment_indices[i]
        end = segment_indices[i+1]
        
        if start >= end: # Lida com blocos com menos frames que o necessário
            # Estratégia de looping: se o segmento está vazio, usa o último frame disponível
            if available_frames:
                frame_index_to_sample = (start) % total_frames
                sampled_frame_paths.append(available_frames[frame_index_to_sample])
        else:
            # Sorteia um frame aleatório dentro do segmento
            frame_index_to_sample = random.randint(start, end - 1)
            sampled_frame_paths.append(available_frames[frame_index_to_sample])

    # --- Processamento e Salvamento ---
    os.makedirs(output_path, exist_ok=True)
    for i, frame_path in enumerate(sampled_frame_paths):
        processed_frame = process_frame_for_i3d(frame_path)
        if processed_frame is not None:
            # Salva o frame processado com um nome sequencial
            output_frame_name = f"frame_{i+1:06d}.jpg"
            cv2.imwrite(os.path.join(output_path, output_frame_name), processed_frame)


def main(source_blocks_dir, output_rgb_dir):
    """
    Função principal para orquestrar o processo de amostragem e processamento.
    """
    print("--- INICIANDO GERAÇÃO DE INPUT RGB PARA O MODELO I3D ---")
    if os.path.exists(output_rgb_dir):
        print(f"Aviso: O diretório de saída '{output_rgb_dir}' já existe. Ele será limpo antes de começar.")
        shutil.rmtree(output_rgb_dir)
    os.makedirs(output_rgb_dir, exist_ok=True)
    
    try:
        block_folders = sorted(os.listdir(source_blocks_dir))
    except FileNotFoundError:
        print(f"ERRO: Diretório de origem '{source_blocks_dir}' não encontrado. Abortando.")
        return

    for block_name in tqdm(block_folders, desc="Processando Blocos de Eventos"):
        source_block_path = os.path.join(source_blocks_dir, block_name)
        if not os.path.isdir(source_block_path):
            continue
        
        output_block_path = os.path.join(output_rgb_dir, block_name)
        
        sample_and_process_block(source_block_path, output_block_path)

    print("\n--- PROCESSO CONCLUÍDO ---")
    print(f"Os inputs RGB foram gerados com sucesso em: '{os.path.abspath(output_rgb_dir)}'")


if __name__ == '__main__':
    # --- CONFIGURE OS CAMINHOS AQUI ---
    
    # 1. O diretório que contém as pastas de blocos de eventos (ex: Normal_0, Shoplifting_0)
    SOURCE_BLOCKS_DIR = '/home/luis/tcc/code/preprocessed/event_blocks_frames'
    
    # 2. O diretório de saída onde o input final para o modelo será salvo
    OUTPUT_I3D_INPUT_DIR = '/home/luis/tcc/code/preprocessed/i3d_inputs'
    
    # -----------------------------------

    main(SOURCE_BLOCKS_DIR, OUTPUT_I3D_INPUT_DIR)
