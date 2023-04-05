from torch import nn
from torch.cuda.amp.autocast_mode import autocast
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim), # 28x28x1
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )
    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01), # paper 上面就是直接使用 LeakyReLu
            nn.Linear(128, 1),
            nn.Sigmoid(), # 最後用 sigmoid 輸出為yes or no
        )
    def forward(self, x):
        return self.disc(x)
