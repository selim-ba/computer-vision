# Deep Convolutional Generative Adversarial Network + MNIST dataset

### Imports ##################
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import os

### Checkpoint dir ###########
checkpoint_dir = '' #to save the model checkpoints (set to each 5 epochs in my code)
os.makedirs(checkpoint_dir, exist_ok=True) #if it doesnt already exists


### Data ##################
img_size = 28 #28 for mnist, # 32 when using celeba
batch_size = 128 

# transform block for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*1, [0.5]*1)
    # We normalize images to [-1,1] by substracting the mean by 0.5 and dividing the std by 0.5
    # Doing so, we match the output of Tanh() in the Generator (see below)

])

mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=128, shuffle=True)

### Generator ##################
class DCGAN_Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, feature_maps=64): # Input: z vector reshaped to (B, z_dim, 1, 1)
        super().__init__()
        self.G = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_maps * 4, 7, 1, 0, bias=False),   # Upsamples from 1x1 to 7x7. Output shape: (B, feature_maps*4, 7, 7)
            nn.BatchNorm2d(feature_maps * 4), # Normalizes activations to stabilize learning (mean=0, std=1)
            nn.ReLU(True), # non-linearity
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False), # Upsamples from 7x7 → 14x14. Output: (B, feature_maps*2, 14, 14)
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True), # non-linearity, keeps the feature maps sparse and expressive
            
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False), # Upsamples from 14x14 → 28x28. Output: (B, img_channels, 28, 28)
            nn.Tanh() # Output pixel values are mapped to [-1, 1] to match image normalization
        )

    def forward(self, z):
        return self.G(z.view(z.size(0), z.size(1), 1, 1))  # reshape to (B, z_dim, 1, 1)

# print(DCGAN_Generator())

### Discriminator ##################
class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_maps=64): # Input: Image tensor (B, img_channels, 28, 28)
        super().__init__()
        self.D = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False), # Downsamples from 28x28 → 14x14. Output: (B, feature_maps, 14, 14) ; # First layer to extract low-level features
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU avoids dying ReLU by allowing small gradients for negative inputs
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False), # Downsamples from 14x14 → 7x7. Output: (B, feature_maps*2, 7, 7)
            nn.BatchNorm2d(feature_maps * 2), # Batch normalization helps stabilize the discriminator's training
            nn.LeakyReLU(0.2, inplace=True), # Again use LeakyReLU for non-linearity with gradient flow on negative values

            nn.Flatten(), # Flatten the (B, feature_maps*2, 7, 7) to shape (B, feature_maps*2 * 7 * 7)
            nn.Linear((feature_maps * 2) * 7 * 7, 1), # Final linear layer to produce a single score per image
            nn.Sigmoid() # Outputs a probability [0,1] for whether image is real or fake
        )

    def forward(self, x):
        return self.D(x)
    
# print(DCGAN_Discriminator())

### Parameters ###################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
lr = 0.0002 #learning rate, i.e. how big the parameter updates are at each step
num_epochs = 100

G = DCGAN_Generator(z_dim).to(device)
D = DCGAN_Discriminator().to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G.apply(weights_init)
D.apply(weights_init)

criterion = nn.BCELoss() #Discriminator is a binary classifier, so we use the Binary Cross-Entropy loss

optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

### Training ###################
fixed_noise = torch.randn(64, z_dim, device=device)
# we use a fixed batch of random noise vectors to generate the same faces every few epochs (so we can track progress)

G_losses = []
D_losses = []
for epoch in range(num_epochs):
    for real_imgs, _ in dataloader:
        # We don't flatten anymore the real_imgs (comapred to GANs)
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0) # (B, 1, 28, 28)

        # Real and Fake labels : target (0/1) for the BCEloss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Discriminator training
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = G(z) # we sample a random noise vector and generate an image ; output shape : (B, 1, 28, 28)

        D_real = D(real_imgs) # how "real" D thinks the real images are, i.e. the discriminator predict proba for real images
        D_fake = D(fake_imgs.detach()) #how "real" D thinks the fake images are, i.e. the discriminator predicted proba for fake images
        # .detach() avoids backpropagation into the generator when training D

        d_loss_real = criterion(D_real, real_labels) #BCELoss for real images := 1
        d_loss_fake = criterion(D_fake, fake_labels) #BCELoss for fake images := 0
        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad() # we reset gradients
        d_loss.backward() #backpropagation in D
        optimizer_D.step() #update of D weights

        # Generator training
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = G(z) # we generate a fake image
        D_fake = D(fake_imgs) #we pass them through D

        g_loss = criterion(D_fake, real_labels)  # we label them to real (to fool D)
        # Loss is high when D(G(z)) is near 0, i.e. G can't fool D anymore
        # Loss is low when (D(G(Z))) is near 1, i.e. G is still fooling D

        optimizer_G.zero_grad() #reset gradients
        g_loss.backward() #backpropagate through G
        optimizer_G.step() #update G weights

    

    print(f"Epoch {epoch+1}/{num_epochs}  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")
    G_losses.append(g_loss.item())
    D_losses.append(d_loss.item())

    # Plot every few epochs
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            fake_imgs = G(fixed_noise).reshape(-1, 1, img_size, img_size) #using fixed_noise for consistency as explained before
            fake_imgs = fake_imgs * 0.5 + 0.5  # rescale from [-1, 1] to [0, 1]
            grid = make_grid(fake_imgs, nrow=8)
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            plt.title(f"Generated Numbers at Epoch {epoch+1}")
            plt.show()

        # Save model checkpoints every 5 epochs
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': G.state_dict(),
            'discriminator_state_dict': D.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
        }, f"{checkpoint_dir}/gan_checkpoint_epoch_{epoch+1}.pt")

### Plot the losses #####################
plt.figure(figsize=(10, 8))
plt.plot(G_losses, label="Generator Loss",marker='*',color='green')
plt.plot(D_losses, label="Discriminator Loss",marker='*',color='red')
plt.title("Generator and Discriminator Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


### Save model weights ############
#torch.save(G.state_dict(), os.path.join(checkpoint_dir, "generator_final.pth"))
#torch.save(D.state_dict(), os.path.join(checkpoint_dir, "discriminator_final.pth"))

# If needed to load + eval mode
#G.load_state_dict(torch.load(os.path.join(checkpoint_dir, "generator_final.pth")))
#G.eval()  
#D.load_state_dict(torch.load(os.path.join(checkpoint_dir, "discriminator_final.pth")))
#D.eval()

