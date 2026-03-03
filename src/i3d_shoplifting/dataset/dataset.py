import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from natsort import natsorted
import random
import torchvision.transforms.v2 as T

class VideoAugmentation:
    def __init__(self, p_flip=0.5, color_jitter_params=None):
        self.p_flip = p_flip
        self.color_jitter = T.ColorJitter(**color_jitter_params) if color_jitter_params else None

    def __call__(self, rgb_tensor, flow_tensor):
        # Flip horizontal
        if random.random() < self.p_flip:
            rgb_tensor = torch.flip(rgb_tensor, dims=[-1])   # W axis
            flow_tensor = torch.flip(flow_tensor, dims=[-1])
            # Inverte canal x do fluxo óptico
            flow_tensor[0] = -flow_tensor[0]

        # Color jitter só no RGB
        if self.color_jitter:
            rgb_t = rgb_tensor.permute(1, 0, 2, 3)  # (T, C, H, W)
            rgb_t = self.color_jitter(rgb_t)        # aplica em todos os frames
            rgb_tensor = rgb_t.permute(1, 0, 2, 3)

        return rgb_tensor, flow_tensor
    
class ShopliftingDataset(Dataset):
    """
    Dataset customizado para carregar os frames RGB e de Fluxo Ótico
    para o treinamento do modelo I3D.

    Este dataset agora retorna os tensores ANTES e DEPOIS 
    da aplicação do data augmentation, para fins de visualização.

    """
    def __init__(self, rgb_dir, flow_dir, transform=None):
        """
        Args:
            rgb_dir (str): Caminho para o diretório com os blocos de frames RGB.
            flow_dir (str): Caminho para o diretório com os blocos de frames de fluxo ótico.
            transform (callable, optional): Transformações opcionais a serem aplicadas.
        """
        self.rgb_dir = rgb_dir
        self.flow_dir = flow_dir
        self.transform = transform

        # Usa natsort para garantir que 'Normal_10' venha depois de 'Normal_9'
        self.samples = natsorted(os.listdir(self.rgb_dir))

    def __len__(self):
        """Retorna o número total de amostras (blocos de eventos)."""
        return len(self.samples)

    def get_label(self, idx):
        """
        Carrega e retorna a label do dataset no índice `idx`.

        Retorna:
            tensor: label
        """
        block_name = self.samples[idx]

        label = 1 if 'Shoplifting' in block_name else 0

        return torch.tensor(label)

    def __getitem__(self, idx):
        """
        Carrega e retorna uma amostra do dataset no índice `idx`.

        Retorna:
            tuple: (rgb_tensor, flow_tensor, label)
        """
        block_name = self.samples[idx]
        
        # 1. Determina o rótulo a partir do nome da pasta
        label = 1 if 'Shoplifting' in block_name else 0
        
        # --- 2. Carrega e processa os frames RGB ---
        rgb_block_path = os.path.join(self.rgb_dir, block_name)
        rgb_frames = []
        # Carrega os 64 frames RGB
        for i in range(1, 65):
            frame_path = os.path.join(rgb_block_path, f"frame_{i:06d}.jpg")
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            # Converte de BGR (OpenCV) para RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(frame)
        
        # Converte a lista de frames para um array NumPy
        rgb_array = np.stack(rgb_frames, axis=0) # Shape: (64, 224, 224, 3)

        # --- 3. Carrega e processa os frames de Fluxo Ótico ---
        flow_block_path = os.path.join(self.flow_dir, block_name)
        flow_frames = []
        # Carrega os 63 pares de frames de fluxo
        for i in range(1, 64):
            flow_x_path = os.path.join(flow_block_path, f"flow_x_{i:06d}.jpg")
            flow_y_path = os.path.join(flow_block_path, f"flow_y_{i:06d}.jpg")
            
            flow_x = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE)
            flow_y = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE)
            
            if flow_x is None or flow_y is None:
                continue
            
            # Empilha os fluxos x e y para formar um "frame" de 2 canais
            flow_frames.append(np.stack([flow_x, flow_y], axis=-1))
        
        # Duplica o último frame de fluxo para corresponder à dimensão temporal de 64
        if flow_frames:
            flow_frames.append(flow_frames[-1])

        flow_array = np.stack(flow_frames, axis=0) # Shape: (64, 224, 224, 2)

        # --- 4. Converte para Tensores e aplica transformações ---
        # Converte para tensor float e normaliza para [0, 1]
        rgb_tensor = torch.from_numpy(rgb_array).float() / 255.0
        flow_tensor = torch.from_numpy(flow_array).float() / 255.0

        # Reorganiza as dimensões para o formato do PyTorch: (C, T, H, W)
        # Original: (T, H, W, C) -> (C, T, H, W)
        rgb_tensor = rgb_tensor.permute(3, 0, 1, 2)
        flow_tensor = flow_tensor.permute(3, 0, 1, 2)
        
        # 1. Clona os tensores originais (ANTES)
        rgb_before = rgb_tensor.clone()
        flow_before = flow_tensor.clone()

        # 2. Aplica a transformação. Os tensores (rgb_tensor, flow_tensor)
        #    serão modificados in-place ou substituídos pela augmentation.
        #    Estes são os tensores "DEPOIS".
        if self.transform:
            rgb_tensor, flow_tensor = self.transform(rgb_tensor, flow_tensor)

        # 3. Retorna ambos os conjuntos
        return rgb_before, flow_before, rgb_tensor, flow_tensor, torch.tensor(label)

