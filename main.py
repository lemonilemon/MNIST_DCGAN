import models
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
from tensorboard import program
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Constants
EPOCHS = 5 
BATCH_SIZE = 32
NUM_WORKER = 8
Z_DIM = 126
H_DIM = 64
IMAGE_SIZE = 64 
IMAGE_DIM = 1
LEARNING_RATE = 3e-4
LOG_DIR = "./runs/DCGAN"

# Tensorboard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', LOG_DIR])
url = tb.launch()
print(f"Tensorflow listening on {url}")

# Transforms

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    transforms.Resize(IMAGE_SIZE)
])
# Data
dataset = datasets.MNIST(root = "./data", transform = train_transform, download = True)
loader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
writer_fake = SummaryWriter(LOG_DIR + "/fake")
writer_real = SummaryWriter(LOG_DIR + "/real")

# Model & optimizer ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = models.Generator(Z_DIM, H_DIM, IMAGE_DIM).to(device)
disc = models.Discriminator(IMAGE_DIM, H_DIM).to(device)
opt_G = torch.optim.Adam(gen.parameters(), lr = LEARNING_RATE)
opt_D = torch.optim.Adam(disc.parameters(), lr = LEARNING_RATE)
criterion = nn.BCELoss()

# Model Graph
writer_fake.add_graph(gen, torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device))
writer_real.add_graph(disc, torch.randn((BATCH_SIZE, IMAGE_DIM, IMAGE_SIZE, IMAGE_SIZE)).to(device))

# Initialize
fixed_noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
sample, _ = next(iter(loader))
img_grid_real = torchvision.utils.make_grid(sample, normalize=True)
writer_real.add_image("Mnist Real Images", img_grid_real, global_step = 0)
imglist = []

for epoch in range(1, EPOCHS + 1):
    # Loss Recorder
    avglossD = 0.0
    avglossG = 0.0

    loop = tqdm(loader)
    loop.set_description(f"Testing Epoch[{epoch}/{EPOCHS}]")
    for real, _ in loop:
        real = real.to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc.zero_grad()
        noise = torch.randn((batch_size, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward(retain_graph=True)
        opt_D.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        gen.zero_grad()
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        lossG.backward()
        opt_G.step()
        avglossG += lossG.item()
        avglossD += lossD.item()

    with torch.no_grad():
        avglossD /= len(dataset)
        avglossG /= len(dataset)
        fake = gen(fixed_noise)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize = True)
        print(f"Generator Loss = {avglossG}, Discriminator Loss = {avglossD}")
        writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step = epoch)  
        writer_fake.add_scalar("Loss", scalar_value = avglossG, global_step = epoch)
        writer_real.add_scalar("Loss", scalar_value = avglossD, global_step = epoch) 
        img = img_grid_fake.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        imglist.append(np.uint8(img * 255))
imglist = [Image.fromarray(img) for img in imglist] 
imglist[0].save("GIF/fake.gif", save_all = True, append_images = imglist[1:], duration = 600, loop = 0)
