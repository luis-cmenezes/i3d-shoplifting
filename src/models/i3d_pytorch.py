import torch
import torch.nn as nn
import torch.nn.functional as F

class Unit3D(nn.Module):
    """O bloco de construção básico para I3D: uma convolução 3D seguida de batch norm e ReLU."""
    def __init__(self, in_channels, output_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, use_batch_norm=True, use_relu=True, use_bias=False):
        super(Unit3D, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.use_relu = use_relu
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        if self.use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels)

    def forward(self, x):
        x = self.conv3d(x)
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_relu:
            x = F.relu(x)
        return x

class InceptionModule(nn.Module):
    """Um módulo Inception, adaptado para 3D."""
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_size=(1, 1, 1))
        
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_size=(1, 1, 1))
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_size=(3, 3, 3), padding=1)
        
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_size=(1, 1, 1))
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_size=(3, 3, 3), padding=1)
        
        self.b3a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_size=(1, 1, 1))

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """A arquitetura I3D completa, baseada na Inception V1."""
    
    def __init__(self, num_classes=400, in_channels=3):
        super(InceptionI3d, self).__init__()

        self.Conv3d_1a_7x7 = \
            Unit3D(in_channels=in_channels, output_channels=64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=3)
        self.MaxPool3d_2a_3x3 = \
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Conv3d_2b_1x1 = \
            Unit3D(in_channels=64, output_channels=64, kernel_size=(1, 1, 1))
        self.Conv3d_2c_3x3 = \
            Unit3D(in_channels=64, output_channels=192, kernel_size=(3, 3, 3), padding=1)
        self.MaxPool3d_3a_3x3 = \
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.Mixed_3b = \
            InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.Mixed_3c = \
            InceptionModule(256, [128, 128, 192, 32, 96, 64])
        self.MaxPool3d_4a_3x3 = \
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)

        self.Mixed_4b = \
            InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.Mixed_4c = \
            InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.Mixed_4d = \
            InceptionModule(512, [128, 128, 256, 24, 64, 64])
        self.Mixed_4e = \
             InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.Mixed_4f = \
             InceptionModule(528, [256, 160, 320, 32, 128, 128])
        self.MaxPool3d_5a_2x2 = \
             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.Mixed_5b = \
            InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.Mixed_5c = \
            InceptionModule(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = \
             nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        self.dropout = \
             nn.Dropout(0.5)
        
        # A camada final de classificação (logits).
        # É esta camada que vamos substituir para o fine-tuning.
        self.logits = Unit3D(in_channels=1024, output_channels=num_classes,
                             kernel_size=(1, 1, 1),
                             use_batch_norm=False,
                             use_relu=False,
                             use_bias=True)

    def replace_logits(self, num_classes):
        """Função auxiliar para substituir a camada de classificação."""
        print(f"Substituindo a camada de classificação final. Nova quantidade de classes: {num_classes}")
        self.logits = Unit3D(in_channels=1024, output_channels=num_classes,
                             kernel_size=(1, 1, 1),
                             use_batch_norm=False,
                             use_relu=False,
                             use_bias=True)
    
    def forward(self, x):
        # Forward pass através da rede
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        
        # O formato da saída do logits é (batch_size, num_classes, 1, 1, 1)
        # Removemos as dimensões extras para ter (batch_size, num_classes)
        return x.squeeze(3).squeeze(3).squeeze(2)
