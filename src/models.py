"""
Neural network models for image restoration tasks
U-Net architecture with residual blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutions"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        return out


class UNetBlock(nn.Module):
    """U-Net block with skip connections"""
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class UNetEncoder(nn.Module):
    """Encoder part of U-Net"""
    def __init__(self, in_channels, channels):
        super(UNetEncoder, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.block1 = UNetBlock(in_channels, channels)
        self.block2 = UNetBlock(channels, channels * 2)
        self.block3 = UNetBlock(channels * 2, channels * 4)
        self.block4 = UNetBlock(channels * 4, channels * 8)
    
    def forward(self, x):
        x1 = self.block1(x)
        x = self.pool(x1)
        
        x2 = self.block2(x)
        x = self.pool(x2)
        
        x3 = self.block3(x)
        x = self.pool(x3)
        
        x4 = self.block4(x)
        
        return x1, x2, x3, x4


class UNetDecoder(nn.Module):
    """Decoder part of U-Net"""
    def __init__(self, channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=2, stride=2)
        self.block1 = UNetBlock(channels * 8, channels * 4)
        
        self.upconv2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)
        self.block2 = UNetBlock(channels * 4, channels * 2)
        
        self.upconv3 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
        self.block3 = UNetBlock(channels * 2, channels)
        
        self.upconv4 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.block4 = UNetBlock(channels * 2, channels)
        
        self.final = nn.Conv2d(channels, out_channels, kernel_size=1)
    
    def forward(self, x, x3, x2, x1):
        x = self.upconv1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.block1(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.block2(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.block3(x)
        
        x = self.upconv4(x)
        # Handle skip connection at first level
        x = self.block4(x)
        
        x = self.final(x)
        return x


class UNet(nn.Module):
    """Complete U-Net architecture"""
    def __init__(self, in_channels=3, out_channels=3, channels=32):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, channels)
        self.decoder = UNetDecoder(channels, out_channels)
    
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x = self.decoder(x4, x3, x2, x1)
        return x


class ResidualUNet(nn.Module):
    """U-Net with residual blocks"""
    def __init__(self, in_channels=3, out_channels=3, channels=32, num_residuals=4):
        super(ResidualUNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, channels)
        
        # Bottleneck with residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels * 8) for _ in range(num_residuals)]
        )
        
        self.decoder = UNetDecoder(channels, out_channels)
    
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x4 = self.residual_blocks(x4)
        x = self.decoder(x4, x3, x2, x1)
        return x


class SuperResolutionNet(nn.Module):
    """
    Specialized network for super-resolution with upsampling
    """
    def __init__(self, in_channels=3, out_channels=3, channels=32, scale_factor=4, num_residuals=4):
        super(SuperResolutionNet, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residuals)]
        )
        
        # Upsampling
        if scale_factor == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        elif scale_factor == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")
        
        # Final reconstruction
        self.conv_final = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv_first(x)
        residual = x
        x = self.residual_blocks(x)
        x = x + residual
        x = self.upsample(x)
        x = self.conv_final(x)
        return x


class DenoiseNet(nn.Module):
    """
    Specialized network for image denoising
    """
    def __init__(self, in_channels=3, out_channels=3, channels=32, num_residuals=8):
        super(DenoiseNet, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residuals)]
        )
        
        # Final reconstruction
        self.conv_final = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        x = self.conv_first(x)
        x = self.residual_blocks(x)
        x = self.conv_final(x)
        x = x + residual  # Residual connection from input
        return x


def create_model(task='sr', scale_factor=4, device='cpu'):
    """
    Factory function to create appropriate model
    Args:
        task: 'sr' for super-resolution, 'denoise' for denoising
        scale_factor: for SR task
        device: torch device
    Returns:
        model on specified device
    """
    if task == 'sr':
        model = SuperResolutionNet(scale_factor=scale_factor)
    elif task == 'denoise':
        model = DenoiseNet()
    elif task == 'inpainting':
        model = DenoiseNet()
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return model.to(device)

