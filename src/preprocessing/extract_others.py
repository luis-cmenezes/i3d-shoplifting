import os
import re
import subprocess
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


if __name__ == '__main__':
    # --- CONFIGURE OS CAMINHOS AQUI ---
    
    # 1. O diretório que contém os novos vídeos curtos de shoplifting
    NEW_VIDEOS_INPUT_DIR = '/home/luis/tcc/datasets/Shoplifting Dataset 2.0/see and let'
    
    # 2. O diretório de saída principal, o mesmo usado no script anterior
    #    (ex: 'data_blocks_with_context')
    OUTPUT_BLOCKS_DIR = '/home/luis/tcc/code/preprocessed/event_blocks_frames'
    
    DATASET_TYPE = 'Normal'
    # -----------------------------------

    augment_with_new_videos(NEW_VIDEOS_INPUT_DIR, OUTPUT_BLOCKS_DIR, DATASET_TYPE)
