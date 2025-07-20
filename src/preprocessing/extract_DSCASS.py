import os
import pandas as pd
import subprocess
from tqdm import tqdm

def load_annotations(annotations_path):
    """
    Carrega o arquivo de anotações em um DataFrame do Pandas para buscas eficientes.
    """
    try:
        annotations_df = pd.read_csv(annotations_path, header=None, names=['clip_name_base', 'category', 'label'])
        annotations_df.set_index('clip_name_base', inplace=True)
        print(f"Arquivo de anotações '{annotations_path}' carregado com sucesso.")
        return annotations_df
    except FileNotFoundError:
        print(f"ERRO: Arquivo de anotações não encontrado em '{annotations_path}'. Abortando.")
        return None

def identify_event_blocks_with_context(dataset_root, annotations_df):
    """
    Itera sobre o dataset, identifica blocos de clipes contíguos com o mesmo rótulo
    e adiciona clipes de contexto (anterior/posterior) aos blocos de Shoplifting.
    """
    print("Identificando blocos de eventos e adicionando contexto...")
    all_blocks = []
    situation_folders = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])

    for situation_name in tqdm(situation_folders, desc="Analisando Situações"):
        situation_path = os.path.join(dataset_root, situation_name)
        
        # Coleta e ordena todos os clipes e seus rótulos para a situação atual
        situation_clip_data = []
        try:
            sorted_clip_names = sorted(
                [f for f in os.listdir(situation_path) if f.endswith('.mp4')],
                key=lambda name: int(os.path.splitext(name)[0].split('_')[-1])
            )
            for clip_filename in sorted_clip_names:
                clip_name_base = os.path.splitext(clip_filename)[0]
                label = annotations_df.loc[clip_name_base, 'label']
                situation_clip_data.append({
                    'path': os.path.join(situation_path, clip_filename),
                    'label': label
                })
        except (ValueError, IndexError, KeyError) as e:
            print(f"\nAviso: Problema ao processar clipes em '{situation_name}': {e}. Pulando situação.")
            continue
        
        if not situation_clip_data:
            continue

        # Itera sobre a lista de clipes da situação para identificar os blocos
        i = 0
        while i < len(situation_clip_data):
            start_index = i
            current_label = situation_clip_data[start_index]['label']
            
            # Encontra o final do bloco contíguo
            end_index = start_index
            while end_index + 1 < len(situation_clip_data) and situation_clip_data[end_index + 1]['label'] == current_label:
                end_index += 1
            
            # Extrai os caminhos dos clipes para o bloco atual
            block_clip_paths = [d['path'] for d in situation_clip_data[start_index : end_index + 1]]

            # Se o bloco for de Shoplifting, adiciona os clipes vizinhos (se existirem)
            if current_label == 1:
                # Adiciona o clipe anterior, se não for o primeiro da situação
                if start_index > 0:
                    context_before_path = situation_clip_data[start_index - 1]['path']
                    block_clip_paths.insert(0, context_before_path)
                
                # Adiciona o clipe posterior, se não for o último da situação
                if end_index < len(situation_clip_data) - 1:
                    context_after_path = situation_clip_data[end_index + 1]['path']
                    block_clip_paths.append(context_after_path)

            all_blocks.append({'label': current_label, 'clip_paths': block_clip_paths})
            
            # Pula para o início do próximo bloco
            i = end_index + 1
            
    print(f"Identificação concluída. Total de {len(all_blocks)} blocos de eventos encontrados.")
    return all_blocks

def process_and_extract_blocks(blocks_list, output_root_dir):
    """
    Recebe a lista de blocos, funde os clipes de cada um, reamostra para 25 FPS
    e extrai os frames para pastas nomeadas.
    """
    print("Processando blocos com FFmpeg (concatenação, reamostragem e extração)...")
    os.makedirs(output_root_dir, exist_ok=True)
    
    normal_counter = 0
    shoplifting_counter = 0

    for block in tqdm(blocks_list, desc="Processando Blocos"):
        block_label = block['label']
        clip_paths = block['clip_paths']

        if block_label == 1:
            block_output_name = f"Shoplifting_{shoplifting_counter}"
            shoplifting_counter += 1
        else:
            block_output_name = f"Normal_{normal_counter}"
            normal_counter += 1
        
        block_output_dir = os.path.join(output_root_dir, block_output_name)
        os.makedirs(block_output_dir, exist_ok=True)
        
        file_list_path = os.path.join(output_root_dir, 'temp_file_list.txt')
        with open(file_list_path, 'w') as f:
            for path in clip_paths:
                f.write(f"file '{os.path.abspath(path)}'\n")

        command = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', f'"{file_list_path}"',
            '-vf', 'fps=25', '-q:v', '2', '-y',
            f'"{os.path.join(block_output_dir, "frame_%06d.jpg")}"'
        ]

        try:
            subprocess.run(" ".join(command), shell=True, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\nERRO ao processar o bloco: {block_output_name}")
            print(f"Erro FFmpeg: {e.stderr}")
        finally:
            if os.path.exists(file_list_path):
                os.remove(file_list_path)

if __name__ == '__main__':
    # --- CONFIGURE OS CAMINHOS AQUI ---
    DATASET_ROOT_DIRECTORY = '/home/luis/tcc/datasets/DCSASS_Dataset/situations'
    ANNOTATIONS_CSV_PATH = '/home/luis/tcc/datasets/DCSASS_Dataset/Shoplifting.csv'
    OUTPUT_BLOCKS_DIR = '/home/luis/tcc/code/preprocessed/event_blocks_frames' # Nome de pasta sugerido
    # -----------------------------------

    annotations = load_annotations(ANNOTATIONS_CSV_PATH)

    if annotations is not None:
        event_blocks = identify_event_blocks_with_context(DATASET_ROOT_DIRECTORY, annotations)
        
        if event_blocks:
            process_and_extract_blocks(event_blocks, OUTPUT_BLOCKS_DIR)
            print("\n--- PROCESSO CONCLUÍDO ---")
        else:
            print("Nenhum bloco de evento foi identificado. Verifique seus dados.")
