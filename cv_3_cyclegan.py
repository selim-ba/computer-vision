## CycleGAN : Application to Style Transfer : Monet <-> Real Landscapes (+ Cycle Consistency)
## Code I have used on Google Colab (NVIDA A100 GPU), feel free to modify it is executed on your device

### Imports #########################################################
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import itertools
import glob
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from torchvision import transforms
from torchvision.utils import make_grid, save_image


### Google Colab Config ###############################################
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high") 


## Data Source : https://efrosgans.eecs.berkeley.edu/cyclegan/datasets
## Related Paper : https://arxiv.org/pdf/1703.10593
data_root = "/content/drive/MyDrive/cyclegan_data/datasets/monet2photo"
output_dir = "/content/drive/MyDrive/cyclegan_data/outputs_fast"
checkpoint_dir = "/content/drive/MyDrive/cyclegan_data/checkpoints_fast"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

### (Hyper)parameters ################################################
img_size = 256 
batch_size = 8  
num_workers = 4 
total_epochs = 100
init_lr = 2e-4
lambda_cyc = 10.0
decay_start_epoch = total_epochs // 2  #
save_interval = 5  #parameter used to save generated/translated images each 5 epochs


### Load and transform images ###################################@###
class Load_Transform_Unpaired_Img(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(glob.glob(os.path.join(root, "*.*")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize(int(img_size * 1.12),
                      interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop(img_size), #Resized and cropped images because had some issues with different image sizes

    transforms.RandomHorizontalFlip(), # Some modest data augmentation
    transforms.ToTensor(), #PIL -> tensor
    transforms.Normalize([0.5]*3, [0.5]*3), # Each RGB chanel is normalizd to [-1,1]
])

## Pers note : Monet (trainA, testA) ; Landscape (trainB, testB)
dataset_monet = Load_Transform_Unpaired_Img(os.path.join(data_root, "trainA"), transform=transform)
dataset_landscape = Load_Transform_Unpaired_Img(os.path.join(data_root, "trainB"), transform=transform)

loader_monet = torch.utils.data.DataLoader(
    dataset_monet,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=2)

loader_landscape = torch.utils.data.DataLoader(
    dataset_landscape,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=2)

loader_landscape_iter = itertools.cycle(loader_landscape)

### CycleGAN #############################
## Residual block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)
#print(ResBlock(64))

## Generator
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # downsample
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            layers += [ResBlock(ngf*4)]

        layers += [
            # upsample
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7, padding=0),
            nn.Tanh()
        ]
        self.Gen = nn.Sequential(*layers)

    def forward(self, x):
        return self.Gen(x)
#print(Generator())

## Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, stride=1, padding=1)
        ]
        self.Disc = nn.Sequential(*layers)

    def forward(self, x):
        return self.Disc(x)
#print(Discriminator())

### Pre-training instantiation ##########################
G = torch.compile(Generator().to(device))
F = torch.compile(Generator().to(device))

D_landscape = torch.compile(Discriminator().to(device))
D_monet = torch.compile(Discriminator().to(device))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

for net in [G, F, D_monet, D_landscape]:
    net.apply(init_weights)

## Loss functions
adv_criterion = nn.MSELoss()
cycle_criterion = nn.L1Loss()

## Optimizers
optimizer_G = optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=init_lr, betas=(0.5, 0.999))
optimizer_D_landscape = optim.Adam(D_landscape.parameters(), lr=init_lr, betas=(0.5, 0.999))
optimizer_D_monet = optim.Adam(D_monet.parameters(), lr=init_lr, betas=(0.5, 0.999))

## Schedulers (constant for 'decay_star_epoch' epochs, then linear decay)
def lambda_lr(epoch):
    if epoch <= decay_start_epoch:
        return 1.0
    return max(0.0, 1.0 - (epoch - decay_start_epoch) / (total_epochs - decay_start_epoch))

scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_lr)
scheduler_D_landscape = optim.lr_scheduler.LambdaLR(optimizer_D_landscape, lr_lambda=lambda_lr)
scheduler_D_monet = optim.lr_scheduler.LambdaLR(optimizer_D_monet, lr_lambda=lambda_lr)

## AMP Scaler (for NVIDA Ampere)
scaler_G = GradScaler()
scaler_D_landscape = GradScaler()
scaler_D_monet = GradScaler()

def real_label(x):
    return torch.ones_like(x, device=device)

def fake_label(x):
    return torch.zeros_like(x, device=device)

### Start epoch (or resume from epoch) #############@############
start_epoch = 1
latest_ckpt = os.path.join(checkpoint_dir, "latest.pt")

## Uncomment below if resuming from a certain epoch : 
#if os.path.isfile(latest_ckpt):
#    ckpt = torch.load(latest_ckpt, map_location=device)
#    G.load_state_dict(ckpt["G"])
#    F.load_state_dict(ckpt["F"])
#    D_monet.load_state_dict(ckpt["D_monet"])
#    D_landscape.load_state_dict(ckpt["D_landscape"])
#    optimizer_G.load_state_dict(ckpt["opt_G"])
#    optimizer_D_monet.load_state_dict(ckpt["opt_D_monet"])
#    optimizer_D_landscape.load_state_dict(ckpt["opt_D_landscape"])
#    scheduler_G.load_state_dict(ckpt.get("sched_G", {}))
#    scheduler_D_monet.load_state_dict(ckpt.get("sched_D_monet", {}))
#    scheduler_D_landscape.load_state_dict(ckpt.get("sched_D_landscape", {}))
#    start_epoch = ckpt.get("epoch", 0) + 1
#    print(f"Resuming from epoch {start_epoch}")

### Training #########################################################
# fixed samples for consistent visualization across epochs
fixed_monet = next(iter(loader_monet))[:4].to(device)
fixed_landscape = next(iter(loader_landscape))[:4].to(device)

G_losses, D_monet_losses, D_landscape_losses = [], [], []
t0 = time.time()

for epoch in range(start_epoch, total_epochs + 1):
    epoch_start = time.time()
    for real_monet in loader_monet:
        real_landscape = next(loader_landscape_iter)

        real_monet = real_monet.to(device, non_blocking=True)
        real_landscape = real_landscape.to(device, non_blocking=True)

        ## Generators : G and F
        with autocast():
            fake_landscape = G(real_monet)     # Monet to Landscape
            recov_monet = F(fake_landscape)    # Cycle Monet
            fake_monet = F(real_landscape)     # Landscape to Monet
            recov_landscape = G(fake_monet)    # Cycle Landscape

            pred_fake_landscape = D_landscape(fake_landscape)
            pred_fake_monet = D_monet(fake_monet)

            loss_GAN_G = adv_criterion(pred_fake_landscape, real_label(pred_fake_landscape))
            loss_GAN_F = adv_criterion(pred_fake_monet, real_label(pred_fake_monet))
            loss_cycle = cycle_criterion(recov_monet, real_monet) + cycle_criterion(recov_landscape, real_landscape)
            loss_G = loss_GAN_G + loss_GAN_F + lambda_cyc * loss_cycle

        optimizer_G.zero_grad()
        scaler_G.scale(loss_G).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()

        ## Disc. Landscape
        with autocast():
            pred_real_landscape = D_landscape(real_landscape)
            pred_fake_landscape_detached = D_landscape(fake_landscape.detach())

            loss_D_landscape_real = adv_criterion(pred_real_landscape, real_label(pred_real_landscape))
            loss_D_landscape_fake = adv_criterion(pred_fake_landscape_detached, fake_label(pred_fake_landscape_detached))
            loss_D_landscape = 0.5 * (loss_D_landscape_real + loss_D_landscape_fake)

        optimizer_D_landscape.zero_grad()
        scaler_D_landscape.scale(loss_D_landscape).backward()
        scaler_D_landscape.step(optimizer_D_landscape)
        scaler_D_landscape.update()

        ## Disc. Monet
        with autocast():
            pred_real_monet = D_monet(real_monet)
            pred_fake_monet_detached = D_monet(fake_monet.detach())

            loss_D_monet_real = adv_criterion(pred_real_monet, real_label(pred_real_monet))
            loss_D_monet_fake = adv_criterion(pred_fake_monet_detached, fake_label(pred_fake_monet_detached))
            loss_D_monet = 0.5 * (loss_D_monet_real + loss_D_monet_fake)

        optimizer_D_monet.zero_grad()
        scaler_D_monet.scale(loss_D_monet).backward()
        scaler_D_monet.step(optimizer_D_monet)
        scaler_D_monet.update()

    scheduler_G.step()
    scheduler_D_landscape.step()
    scheduler_D_monet.step()

    G_losses.append(loss_G.item())
    D_monet_losses.append(loss_D_monet.item())
    D_landscape_losses.append(loss_D_landscape.item())

    epoch_time = time.time() - epoch_start
    print(
        f"[Epoch {epoch}/{total_epochs}] G: {loss_G.item():.4f}  "
        f"D_monet: {loss_D_monet.item():.4f}  D_landscape: {loss_D_landscape.item():.4f}  "
        f"lr: {scheduler_G.get_last_lr()[0]:.2e}  time: {epoch_time:.1f}s"
    )

    if epoch % save_interval == 0 or epoch == total_epochs:
        with torch.no_grad():
            sample_monet = fixed_monet
            sample_landscape = fixed_landscape
            fake_landscape_vis = G(sample_monet)
            fake_monet_vis = F(sample_landscape)
            recov_monet_vis = F(fake_landscape_vis)
            recov_landscape_vis = G(fake_monet_vis)

        def unnorm(x):
            return (x * 0.5 + 0.5).clamp(0,1)

        date_str = datetime.now().strftime("%Y%m%d")

        ## Monet to Landscape
        grid_monet2landscape = make_grid(unnorm(fake_landscape_vis), nrow=4)
        save_image(grid_monet2landscape, os.path.join(output_dir, f"{date_str}_epoch_{epoch}_monet2landscape.png"))

        ## Cycle Monet: Monet to Landscape to Monet
        grid_cycle_monet = make_grid(unnorm(recov_monet_vis), nrow=4)
        save_image(grid_cycle_monet, os.path.join(output_dir, f"{date_str}_epoch_{epoch}_cycle_monet.png"))

        ## Real Monet
        grid_real_monet = make_grid(unnorm(sample_monet), nrow=4)
        save_image(grid_real_monet, os.path.join(output_dir, f"{date_str}_epoch_{epoch}_real_monet.png"))

        ## Landscape to Monet
        grid_landscape2monet = make_grid(unnorm(fake_monet_vis), nrow=4)
        save_image(grid_landscape2monet, os.path.join(output_dir, f"{date_str}_epoch_{epoch}_landscape2monet.png"))

        ## Cycle Landscape: Landscape to Monet to Landscape
        grid_cycle_landscape = make_grid(unnorm(recov_landscape_vis), nrow=4)
        save_image(grid_cycle_landscape, os.path.join(output_dir, f"{date_str}_epoch_{epoch}_cycle_landscape.png"))

        ## Real Landscape
        grid_real_landscape = make_grid(unnorm(sample_landscape), nrow=4)
        save_image(grid_real_landscape, os.path.join(output_dir, f"{date_str}_epoch_{epoch}_real_landscape.png"))

        ## Checkpoints
        ckpt = {
            "epoch": epoch,
            "G": G.state_dict(),
            "F": F.state_dict(),
            "D_monet": D_monet.state_dict(),
            "D_landscape": D_landscape.state_dict(),
            "opt_G": optimizer_G.state_dict(),
            "opt_D_monet": optimizer_D_monet.state_dict(),
            "opt_D_landscape": optimizer_D_landscape.state_dict(),
            "sched_G": scheduler_G.state_dict(),
            "sched_D_monet": scheduler_D_monet.state_dict(),
            "sched_D_landscape": scheduler_D_landscape.state_dict(),
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, f"cyclegan_epoch_{epoch}.pt"))
        torch.save(ckpt, latest_ckpt)


### Plot and Save Loss functions ######################################################
def plot_loss(G_losses, D_monet_losses, D_landscape_losses, output_dir):

    epochs = list(range(1, len(G_losses) + 1))

    plt.figure(figsize=(14, 10))
    plt.plot(epochs, G_losses, label="Generator loss (total)") #loss G + loss F + (labmda * loss cycle)
    plt.plot(epochs, D_monet_losses, label="Discriminator Monet loss")
    plt.plot(epochs, D_landscape_losses, label="Discriminator Landscape loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CycleGAN losses over epochs")
    plt.legend()
    plt.grid(True)

    date_str = datetime.now().strftime("%Y%m%d")
    plot_path = os.path.join(output_dir, f"{date_str}_losses.png")
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"Loss plot saved to {plot_path}")

plot_loss(G_losses, D_monet_losses, D_landscape_losses, output_dir)



