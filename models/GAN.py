import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, latent_dim=256, device=device):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.device = device
        
        # Project latent vector to 4x4x512 (~10M params target)
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Decoder: 4x4x512 -> 32x32x3
        self.decoder = nn.Sequential(
            # 4x4x512 -> 8x8x512
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 8x8x512 -> 16x16x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 16x16x256 -> 32x32x3
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 512, 4, 4)
        img = self.decoder(h)
        return img


class Discriminator(nn.Module):
    def __init__(self, device=device):
        super(Discriminator, self).__init__()
        
        self.device = device
        
        # Encoder: 32x32x3 -> 4x4x512
        self.encoder = nn.Sequential(
            # 32x32x3 -> 16x16x256
            nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16x256 -> 8x8x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8x512 -> 4x4x512
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output: real or fake
        self.fc = nn.Linear(512 * 4 * 4, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return out
