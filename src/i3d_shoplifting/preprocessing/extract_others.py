import os
import re
import subprocess
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

def get_next_start_index(output_dir, prefix="Shoplifting_"):
    """
    Verifica um diretório de saída para encontrar o maior índice numérico
    usado em nomes de pasta com um determinado prefixo e retorna o próximo índice.

    Args:
        output_dir (str): O diretório onde os blocos estão salvos.
        prefix (str): O prefixo a ser procurado (ex: "Shoplifting_").

    Returns:
        int: O próximo índice a ser usado para um novo bloco.
    """
    if not os.path.isdir(output_dir):
        print(f"Aviso: Diretório de saída '{output_dir}' não encontrado. Começando do índice 0.")
        return 0

    # Expressão regular para encontrar o número no final do nome da pasta
    p = re.compile(rf'^{re.escape(prefix)}(\d+)$')
    
    max_index = -1
    for dirname in os.listdir(output_dir):
        # Verifica se o item é um diretório e corresponde ao padrão
        if os.path.isdir(os.path.join(output_dir, dirname)):
            match = p.match(dirname)
            if match:
                try:
                    index = int(match.group(1))
                    if index > max_index:
                        max_index = index
                except ValueError:
                    continue # Ignora se a parte numérica não for um inteiro válido
    
    # O próximo índice é o maior encontrado + 1
    return max_index + 1

def augment_with_new_videos(new_videos_dir, output_blocks_dir, string_to_look_for):
    """
    Processa um diretório de novos vídeos de <string_to_look_for>, extraindo seus frames
    para o diretório de blocos existente, continuando a sequência de numeração.

    Args:
        new_videos_dir (str): O diretório contendo os novos vídeos curtos.
        output_blocks_dir (str): O diretório principal onde os blocos são salvos.
    """
    print("--- INICIANDO AUMENTO DO DATASET COM NOVOS VÍDEOS ---")

    # 1. Determina a partir de qual número começar a nomear as novas pastas
    curr_counter = get_next_start_index(output_blocks_dir, f"{string_to_look_for}_")
    print(f"Último índice de {string_to_look_for} encontrado. Novos blocos começarão em: {curr_counter}")

    # 2. Lista todos os arquivos de vídeo no diretório de entrada
    try:
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        new_videos = sorted([f for f in os.listdir(new_videos_dir) if f.lower().endswith(video_extensions)])
        if not new_videos:
            print(f"Nenhum vídeo encontrado em '{new_videos_dir}'. Abortando.")
            return
    except FileNotFoundError:
        print(f"ERRO: Diretório de novos vídeos '{new_videos_dir}' não encontrado. Abortando.")
        return

    print(f"Encontrados {len(new_videos)} novos vídeos para processar.")

    # 3. Itera sobre cada novo vídeo, tratando-o como um novo bloco
    for video_filename in tqdm(new_videos, desc="Processando Novos Vídeos"):
        input_video_path = os.path.join(new_videos_dir, video_filename)
        
        # Define o nome da pasta de saída para este novo bloco
        block_output_name = f"{string_to_look_for}_{curr_counter}"
        block_output_dir = os.path.join(output_blocks_dir, block_output_name)
        os.makedirs(block_output_dir, exist_ok=True)

        # Comando FFmpeg para reamostrar e extrair frames de um único vídeo
        command = [
            'ffmpeg',
            '-i', f'"{input_video_path}"',
            '-vf', 'fps=25',
            '-q:v', '2',
            '-y',
            f'"{os.path.join(block_output_dir, "frame_%06d.jpg")}"'
        ]

        try:
            subprocess.run(" ".join(command), shell=True, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\nERRO ao processar o vídeo: {input_video_path}")
            print(f"Erro FFmpeg: {e.stderr}")
        
        # Incrementa o contador para o próximo vídeo
        curr_counter += 1

    print("\n--- AUMENTO DO DATASET CONCLUÍDO ---")
    print("Todos os novos vídeos foram processados e adicionados ao diretório de blocos.")


def main_extract_others(input_dir, output_dir, dataset_type='Normal'):
    """
    Função principal parametrizada para extração de outros datasets.
    
    Args:
        input_dir: Diretório com vídeos de entrada
        output_dir: Diretório de saída para blocos
        dataset_type: Tipo do dataset ('Normal' ou 'Shoplifting')
    """
    augment_with_new_videos(input_dir, output_dir, dataset_type)
    print(f" Extração concluída! Blocos salvos em: {output_dir}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extrai frames de vídeos individuais")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Diretório com vídeos de entrada"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Diretório de saída para blocos"
    )
    parser.add_argument(
        "--dataset-type",
        choices=['Normal', 'Shoplifting'],
        default='Normal',
        help="Tipo do dataset (Normal ou Shoplifting)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Arquivo de configuração YAML (opcional)"
    )
    
    args = parser.parse_args()
    
    # Se config foi fornecido, carrega valores padrão
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Valores padrão baseados na config
        default_output = str(Path(config.get('models', {}).get('i3d', {}).get('data_dir', '')) / "event_blocks_frames")
        
        input_dir = args.input_dir
        output_dir = args.output_dir or default_output
        dataset_type = args.dataset_type
    else:
        if not all([args.input_dir, args.output_dir]):
            print(" Argumentos --input-dir e --output-dir são obrigatórios")
            exit(1)
        
        input_dir = args.input_dir
        output_dir = args.output_dir
        dataset_type = args.dataset_type
    
    # Verifica se diretório de entrada existe
    if not Path(input_dir).exists():
        print(f" Diretório de entrada não encontrado: {input_dir}")
        exit(1)
    
    # Cria diretório de saída
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    success = main_extract_others(input_dir, output_dir, dataset_type)
    exit(0 if success else 1)
