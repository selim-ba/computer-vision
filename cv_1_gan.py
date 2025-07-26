# Generative Adversarial Network + MNIST dataset

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
batch_size = 128 #Might try 64 or 256 later, for the moment we keep 128 as default value for the batch_size

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
class Generator(nn.Module):
    # Generates a flattened image of img_dim (it will be the input for the Discriminator() )
    def __init__(self, z_dim=100, img_dim=1*img_size*img_size):
      # z_dim : dim of the latent noise vector z (100 is large enough to encode sufficient variability, we generally try values between 64-256)
      # img_dim : 1*28*28 = 784
        super().__init__()
        self.G = nn.Sequential(
            nn.Linear(z_dim, 128), #Fully connected linear layer : it computes y = Wx + b
            nn.LeakyReLU(0.2, inplace=True), #non-linear activation layer
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True), #non-linear activation layer
            nn.Linear(256, img_dim), #raw pixel value for the imput image
            nn.Tanh() #maps the raw pixel values to [-1,1] (see transforms.Normalize() in the transform block)
        )

    def forward(self, z):
        return self.G(z)
#print(Generator())

### Discriminator ##################
class Discriminator(nn.Module):
    # Takes as input a flattened image of dim img_dim from the Generator and outputs a proba of being fake or real
    def __init__(self, img_dim=1*img_size*img_size):
        super().__init__()
        self.D = nn.Sequential(
            nn.Linear(img_dim, 256), #MLP : 784 to 256
            nn.LeakyReLU(0.2, inplace=True),
            # As ReLU(x) = max(0,x), if the input is negative, then ReLU(input) will be 0, which leads to the "dying ReLU" problem
            # LeakyReLU(x) = x if x > 0, else = x * 0.2 ; 0.2 slope is large enough to avoid the dying ReLU problem, but still permits non-linearity
            nn.Linear(256, 128), #MLP : 256 to 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1), #MLP : 128 to 1
            nn.Sigmoid() #outputs a proba (scalar between 0 and 1)
        )

    def forward(self, x):
        return self.D(x)
#print(Discriminator())

### Parameters ###################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
lr = 0.0002 #learning rate, i.e. how big the parameter updates are at each step
num_epochs = 100

G = Generator(z_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
#Discriminator is a binary classifier, so we use the Binary Cross-Entropy loss

optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
# Adam combines momentum (moving average of gradients) and RMSprop (moving average of squared gradients)
# in GANs the Adam optimizer is prefered to SGD as it combines momentum and adaptative learning rates
# beta1 = 0.5 (default is 0.9, but 0.5 is recommanded in the DCGAN paper) and beta2=0.999 (default value)
# beta1 : controls the exponential decay of past gradients ; more responsive updates; less momentum helps GANs avoid instability
# beta2 : is for squared gradients ; keeps smooth adaptive step sizes without being too reactive

### Training ###################
fixed_noise = torch.randn(64, z_dim, device=device)
# we use a fixed batch of random noise vectors to generate the same faces every few epochs (so we can track progress)

G_losses = []
D_losses = []
for epoch in range(num_epochs):
    for real_imgs, _ in dataloader:
        # We flatten  images from. (batch_size, 1, H, W) to (batch_size, 3×H×W)
        real_imgs = real_imgs.view(real_imgs.size(0), -1).to(device)
        batch_size = real_imgs.size(0)

        # Real and Fake labels : target (0/1) for the BCEloss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Discriminator training
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = G(z) # we sample a random noise vector and generate an image

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
torch.save(G.state_dict(), os.path.join(checkpoint_dir, "generator_final.pth"))
torch.save(D.state_dict(), os.path.join(checkpoint_dir, "discriminator_final.pth"))

# If needed to load + eval mode
#G.load_state_dict(torch.load(os.path.join(checkpoint_dir, "generator_final.pth")))
#G.eval()  
#D.load_state_dict(torch.load(os.path.join(checkpoint_dir, "discriminator_final.pth")))
#D.eval()
