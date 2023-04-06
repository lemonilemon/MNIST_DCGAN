from torch import nn
class Generator(nn.Module):
    def __init__(self, z_dim, h_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            # z_dim * 1 * 1
            self._block(z_dim, h_dim * 8, 4),
            # (h_dim * 8) * 4 * 4
            self._block(h_dim * 8, h_dim * 4, 4, 2, 1),
            self._block(h_dim * 4, h_dim * 2, 4, 2, 1),
            self._block(h_dim * 2, h_dim, 4, 2, 1),
            nn.ConvTranspose2d(h_dim, img_dim, 4, 2, 1),
            # img_dim * 64 * 64
            nn.Tanh()
        )
    def _block(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim, h_dim):
        super().__init__()
        self.disc = nn.Sequential(
            # img_dim * 64 * 64
            self._block(img_dim, h_dim, 4, 2, 1),
            # img_dim * 32 * 32
            self._block(h_dim, h_dim * 2, 4, 2, 1),
            self._block(h_dim * 2, h_dim * 4, 4, 2, 1),
            self._block(h_dim * 4, h_dim * 8, 4, 2, 1),
            # (h_dim) * 4 * 4 
            nn.Conv2d(h_dim * 8, 1, 4),
            # 1 * 1 * 1
            nn.Sigmoid()
        )
    def _block(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.disc(x)
