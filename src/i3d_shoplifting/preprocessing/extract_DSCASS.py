import os
import pandas as pd
import subprocess
import argparse
import yaml
from pathlib import Path
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
        
        # Identifica blocos contíguos de rótulos iguais
        blocks = []
        current_block = [situation_clip_data[0]]
        current_label = situation_clip_data[0]['label']
        
        for clip_data in situation_clip_data[1:]:
            if clip_data['label'] == current_label:
                current_block.append(clip_data)
            else:
                # Finaliza bloco atual
                blocks.append((current_label, current_block))
                # Inicia novo bloco
                current_block = [clip_data]
                current_label = clip_data['label']
        
        # Adiciona último bloco
        if current_block:
            blocks.append((current_label, current_block))
        
        # Adiciona contexto aos blocos de Shoplifting
        for label, block in blocks:
            if label == 1:  # Shoplifting
                # Adiciona contexto antes e depois se disponível
                extended_block = add_context_to_shoplifting_block(
                    block, situation_clip_data
                )
                extended_clip_paths = [d['path'] for d in extended_block]
                all_blocks.append({'label': label, 'clip_paths': extended_clip_paths})
            else:
                block_clip_paths = [d['path'] for d in block]
                all_blocks.append({'label': label, 'clip_paths': block_clip_paths})
    
    print(f"Identificação concluída. Total de {len(all_blocks)} blocos de eventos encontrados.")
    return all_blocks

def add_context_to_shoplifting_block(shoplifting_block, all_clips):
    """
    Adiciona contexto (clipes anterior e posterior) ao bloco de shoplifting.
    """
    # Encontra índices do primeiro e último clipe do bloco
    first_clip_path = shoplifting_block[0]['path']
    last_clip_path = shoplifting_block[-1]['path']
    
    first_idx = None
    last_idx = None
    
    for i, clip_data in enumerate(all_clips):
        if clip_data['path'] == first_clip_path:
            first_idx = i
        if clip_data['path'] == last_clip_path:
            last_idx = i
    
    extended_block = shoplifting_block.copy()
    
    # Adiciona contexto anterior
    if first_idx > 0:
        extended_block.insert(0, all_clips[first_idx - 1])
    
    # Adiciona contexto posterior
    if last_idx < len(all_clips) - 1:
        extended_block.append(all_clips[last_idx + 1])
    
    return extended_block

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

def main_extract_dcsass(dataset_root, annotations_path, output_dir):
    """
    Função principal parametrizada para extração do DCSASS.
    
    Args:
        dataset_root: Caminho para o diretório raiz do dataset DCSASS
        annotations_path: Caminho para o arquivo de anotações CSV
        output_dir: Diretório de saída para os blocos extraídos
    """
    annotations = load_annotations(annotations_path)

    if annotations is not None:
        event_blocks = identify_event_blocks_with_context(dataset_root, annotations)
        
        if event_blocks:
            process_and_extract_blocks(event_blocks, output_dir)
            print("\n--- PROCESSO CONCLUÍDO ---")
            return True
        else:
            print(" Nenhum bloco de evento foi identificado. Verifique seus dados.")
            return False
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extrai blocos de evento do dataset DCSASS")
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Caminho para o diretório raiz do DCSASS (pasta 'situations')"
    )
    parser.add_argument(
        "--annotations-path",
        type=str,
        help="Caminho para o arquivo de anotações CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Diretório de saída para blocos extraídos"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Arquivo de configuração YAML (opcional)"
    )
    
    args = parser.parse_args()
    
    # Se config foi fornecido, carrega configurações de lá
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Use configurações do arquivo se argumentos não foram fornecidos
        dataset_root = args.dataset_root or config.get('data', {}).get('datasets', {}).get('dcsass', {}).get('root', '')
        annotations_path = args.annotations_path or config.get('data', {}).get('datasets', {}).get('dcsass', {}).get('annotations', '')
        output_dir = args.output_dir or str(Path(config.get('models', {}).get('i3d', {}).get('data_dir', '')) / "event_blocks_frames")
    else:
        # Fallback para argumentos obrigatórios
        if not all([args.dataset_root, args.annotations_path, args.output_dir]):
            print(" Argumentos --dataset-root, --annotations-path e --output-dir são obrigatórios")
            print("   Ou forneça um arquivo de configuração com --config")
            exit(1)
        
        dataset_root = args.dataset_root
        annotations_path = args.annotations_path  
        output_dir = args.output_dir
    
    # Verifica se diretórios existem
    if not Path(dataset_root).exists():
        print(f" Dataset root não encontrado: {dataset_root}")
        exit(1)
    
    if not Path(annotations_path).exists():
        print(f" Arquivo de anotações não encontrado: {annotations_path}")
        exit(1)
    
    # Cria diretório de saída se não existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    success = main_extract_dcsass(dataset_root, annotations_path, output_dir)
    exit(0 if success else 1)
