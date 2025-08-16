## U-Net: Application to Image Segmentation (Cityscapes)
## Code I have used on Google Colab (NVIDA A100 GPU), feel free to modify it, if executed on your local device

### Imports #####################################################
%env CUDA_LAUNCH_BLOCKING=1 # fixed CUDA issues on GoogleColab (A100 GPU)
from google.colab import drive
drive.mount('/content/drive')

import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


print(torch.__version__)
print(torch.version.cuda)

### Cityscapes dataset : loading and transformations #####################################################
class CityscapesDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform

        self.images = []
        for city in sorted(os.listdir(img_dir)):
            city_img_path = os.path.join(img_dir, city)
            city_mask_path = os.path.join(mask_dir, city)

            for img_name in sorted(os.listdir(city_img_path)):
                if img_name.endswith('_leftImg8bit.png'):
                    mask_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    self.images.append((
                        os.path.join(city_img_path, img_name),
                        os.path.join(city_mask_path, mask_name)
                    ))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, mask_path = self.images[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)   # [1, H, W]
            mask = mask.squeeze(0).long()        # [H, W]

        return image, mask

### Generated code with GPT-5, to help solve some issues related to CUDA drivers #####################################################
### Cityscapes color map
cityscapes_colors = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255, 0, 0), ( 0, 0,142), ( 0, 0, 70),
    ( 0,60,100), ( 0,80,100), (0, 0,230), (119, 11, 32)
]

# Mapping of Cityscapes original label IDs to training label IDs (19 classes)
# As defined in the Cityscapes dataset documentation
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
id_to_trainId = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
}

class MapClassIds:
    def __init__(self, id_map, ignore_index=255):
        self.id_map = id_map
        self.ignore_index = ignore_index

    def __call__(self, mask):
        # Apply the mapping to the mask tensor
        # Create a new tensor for the mapped IDs, initialized with the ignore_index
        mapped_mask = torch.full_like(mask, self.ignore_index, dtype=torch.long)
        # Iterate through the id_map and apply the mapping
        for original_id, train_id in self.id_map.items():
             # Only map if the target train_id is not the ignore_index
             if train_id != self.ignore_index:
                mapped_mask[mask == original_id] = train_id
             # Explicitly set original IDs that map to ignore_index to ignore_index
             elif original_id in self.id_map and self.id_map[original_id] == self.ignore_index:
                 mapped_mask[mask == original_id] = self.ignore_index

        return mapped_mask
    
def decode_segmap(mask):
    """
    mask: (H, W) with integer IDs in [0, 18]
    returns RGB image (H, W, 3)
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(cityscapes_colors):
        rgb[mask == cls_id] = color
    return rgb

### Transformations ################################################
image_transform = transforms.Compose([
    transforms.Resize((256, 512)),  # reduce the size of cityscape images
    transforms.ToTensor(),  # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.3257, 0.3690, 0.3223], std=[0.2112, 0.2148, 0.2115])
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 512), interpolation=Image.NEAREST),
    transforms.PILToTensor(),  # keeps integer IDs
    MapClassIds(id_to_trainId) # Map original IDs to training IDs
])

train_dataset = CityscapesDataset(
    img_dir='/content/drive/MyDrive/cityscapes/leftImg8bit/train',
    mask_dir='/content/drive/MyDrive/cityscapes/gtFine/train',
    transform=image_transform,
    target_transform=mask_transform
)

val_dataset = CityscapesDataset(
    img_dir='/content/drive/MyDrive/cityscapes/leftImg8bit/val',
    mask_dir='/content/drive/MyDrive/cityscapes/gtFine/val',
    transform=image_transform,
    target_transform=mask_transform
)

test_dataset = CityscapesDataset(
    img_dir='/content/drive/MyDrive/cityscapes/leftImg8bit/test',
    mask_dir='/content/drive/MyDrive/cityscapes/gtFine/test',
    transform=image_transform,
    target_transform=mask_transform
)


### Data Loaders ####################################################
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8)


### U-Net Class #####################################################
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), #First 3x3 Convolution (padding=1 for same spatial dimension)
            nn.BatchNorm2d(out_channels), # batch layer for training stability
            nn.ReLU(inplace=True), #non-linear activation layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), #Second 3x3 Convolution, followed by the same other layers
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Pytorch uses Kaiming's uniform initialization for the weights
        # Kaiming's normal weights initialization generally works better for segmentation (haven't tried kaiming's uniform for this study)
        for layer in self.doubleconv:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=19, init_features=64):
        super().__init__()
        features = init_features

        # Encoder
        self.enc_level_1 = DoubleConv(in_channels, features)
        self.pool_level_1 = nn.MaxPool2d(2)
        self.enc_level_2 = DoubleConv(features, features*2)
        self.pool_level_2 = nn.MaxPool2d(2)
        self.enc_level_3 = DoubleConv(features*2, features*4)
        self.pool_level_3 = nn.MaxPool2d(2)
        self.enc_level_4 = DoubleConv(features*4, features*8)
        self.pool_level_4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(features*8, features*16)

        # Decoder
        self.upsample_level_4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.dec_level_4 = DoubleConv(features*16, features*8)

        self.upsample_level_3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.dec_level_3 = DoubleConv(features*8, features*4)

        self.upsample_level_2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.dec_level_2 = DoubleConv(features*4, features*2)

        self.upsample_level_1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.dec_level_1 = DoubleConv(features*2, features)

        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)

        # Kaiming normal initialization
        nn.init.kaiming_normal_(self.conv_final.weight, mode='fan_out', nonlinearity='relu')
        if self.conv_final.bias is not None:
            nn.init.zeros_(self.conv_final.bias)

    def forward(self, x):
        # Encoder
        enc_level_1 = self.enc_level_1(x)
        enc_level_2 = self.enc_level_2(self.pool_level_1(enc_level_1))
        enc_level_3 = self.enc_level_3(self.pool_level_2(enc_level_2))
        enc_level_4 = self.enc_level_4(self.pool_level_3(enc_level_3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool_level_4(enc_level_4))

        # Decoder
        dec_level_4 = self.dec_level_4(torch.cat([self.upsample_level_4(bottleneck), enc_level_4], dim=1))
        dec_level_3 = self.dec_level_3(torch.cat([self.upsample_level_3(dec_level_4), enc_level_3], dim=1))
        dec_level_2 = self.dec_level_2(torch.cat([self.upsample_level_2(dec_level_3), enc_level_2], dim=1))
        dec_level_1 = self.dec_level_1(torch.cat([self.upsample_level_1(dec_level_2), enc_level_1], dim=1))

        return self.conv_final(dec_level_1)  # raw logits
    

### Compute mean IoU #####################################################
def compute_miou(preds, masks, num_classes=19, ignore_index=255):
    """
    preds: (B, H, W) predicted class indices
    masks: (B, H, W) ground truth class indices
    """
    miou = 0
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()
    
    for classes in range(num_classes):
        pred_class = (preds == classes)
        mask_class = (masks == classes)
        mask_ignore = (masks != ignore_index)
        
        intersection = np.logical_and(pred_class, mask_class & mask_ignore).sum()
        union = np.logical_or(pred_class & mask_ignore, mask_class & mask_ignore).sum()
        if union == 0:
            continue
        miou += intersection / union
        
    miou /= num_classes
    return miou

### Prepare for training #####################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=19, init_features=64).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_classes = 19
num_epochs = 50 #on Google Colab A100 GPU : 1min30 on the train set, 30sec on the validation set per epoch

### Training loop  #####################################################
train_losses = []
val_losses = []
val_mious = []

for epoch in range(num_epochs):
    ### Train
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    ### Validation
    model.eval()
    val_loss = 0
    miou_total = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            miou_total += compute_miou(preds, masks) * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_miou = miou_total / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_mious.append(val_miou)

    print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
    
    if (epoch + 1) % 5 == 0:
      model.eval()
      with torch.no_grad():
          ### Predictions/Visiualisations on the same first batch of images for consistency
          images, masks = next(iter(val_loader))
          images, masks = images.to(device), masks.to(device)
          outputs = model(images)
          preds = torch.argmax(outputs, dim=1)

          img = images[0].cpu().permute(1,2,0).numpy()
          img = (img * [0.2112, 0.2148, 0.2115] + [0.3257, 0.3690, 0.3223])  # unnormalize
          img = np.clip(img, 0, 1)

          mask = masks[0].cpu().numpy()
          pred_mask = preds[0].cpu().numpy()

          mask_rgb = decode_segmap(mask)
          pred_rgb = decode_segmap(pred_mask)

          plt.figure(figsize=(15,5))
          plt.subplot(1,3,1)
          plt.imshow(img)
          plt.title(f"Epoch {epoch+1} Input")
          plt.axis('off')

          plt.subplot(1,3,2)
          plt.imshow(mask_rgb)
          plt.title("Ground Truth")
          plt.axis('off')

          plt.subplot(1,3,3)
          plt.imshow(pred_rgb)
          plt.title("Prediction")
          plt.axis('off')

          plt.show()

torch.save(model.state_dict(), "unet_50_epochs_cityscapes_weights.pth")

### Loss and Mean IoU
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, num_epochs+1), val_mious, label='Val mIoU')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.title('Validation mIoU')
plt.legend()
plt.show()

### Some predictions on the test set  #########################
num_random_samples = 3 #random samples
random_indices = random.sample(range(len(test_dataset)), num_random_samples)

model.eval()
with torch.no_grad():
    for idx in random_indices:
        image, mask = test_dataset[idx]  

        img_tensor = image.unsqueeze(0).to(device) #(1, C, H, W)
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        img = image.permute(1, 2, 0).numpy()
        img = (img * [0.2112, 0.2148, 0.2115] + [0.3257, 0.3690, 0.3223]) #unnormalize
        img = np.clip(img, 0, 1)

        mask_rgb = decode_segmap(mask.numpy()) #to rgb
        pred_rgb = decode_segmap(pred)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_rgb)
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.show()





