# Computer Vision Projects

This repo showcases my journey through key areas of computer vision. Each project includes annotated code and, and a detailed report.

[1. Image Generation with Generative Adversarial Networks](#1-image-generation-with-generative-adversarial-networks)  
[2. Unpaired Image-To-Image Translation with CycleGAN](#2-unpaired-image-to-image-translation-with-cyclegan)  
[3. Real-Time Urban Scene Segmentation : Mask R-CNN vs U-NET](#3-real-time-urban-scene-segmentation--mask-r-cnn-vs-u-net)

## 1. Image Generation with Generative Adversarial Networks  
[📝 My Report – "Image Generation with GANs"](https://github.com/selim-ba/computer-vision/blob/main/cv_adversarial_networks_image_generation.pdf) | [👉 GAN Implementation](https://github.com/selim-ba/computer-vision/blob/main/cv_1_gan.py) | [👉 DCGAN Implementation](https://github.com/selim-ba/computer-vision/blob/main/cv_2_dcgan.py)

### 🖼️ Generated Digits (GAN)
![Generated Digits with my GAN model](https://github.com/selim-ba/computer-vision/blob/main/gif/gan_generation.gif)

### 🖼️ Generated Digits (DCGAN)
![Generated Digits with my DCGAN model](https://github.com/selim-ba/computer-vision/blob/main/gif/dcgan_generation.gif)

----------

## 2. Unpaired Image-To-Image Translation with CycleGAN
[📝 My Report – "Style Transfer with CycleGAN"](https://github.com/selim-ba/computer-vision/blob/main/cv_cyclegan_style_transfer.pdf) | [👉 CycleGAN Implementation](https://github.com/selim-ba/computer-vision/blob/main/cv_3_cyclegan.py)

### 🖼️ Style Transfer: Real Landscapes → Monet Style → Reconstructed Landscapes
#### 🌄 Original Landscape Photographs
![Real Landscape](https://github.com/selim-ba/computer-vision/blob/main/gif/20250804_epoch_100_real_landscape.png)
#### 🎨 Translated into Monet-style Paintings
![Landscape to Monet](https://github.com/selim-ba/computer-vision/blob/main/gif/cyclegan_landscape2monet.gif)
#### 🔁 Reconstructed Back to Landscapes (Cycle Consistency)
![Back to Landscape](https://github.com/selim-ba/computer-vision/blob/main/gif/cyclegan_cycle_landscape.gif)

### 🖼️  Style Transfer: Monet Paintings → Landscapes → Monet Reconstructions
#### 🎨 Original Monet Paintings
![Real Monet](https://github.com/selim-ba/computer-vision/blob/main/gif/20250804_epoch_100_real_monet.png)
#### 🌄 Translated into Realistic Landscape Photographs
![Monet -> Landscape](https://github.com/selim-ba/computer-vision/blob/main/gif/cyclegan_monet2landscape.gif)
#### 🔁 Reconstructed Back to Monet Paintings (Cycle Consistency)
![Back to Monet](https://github.com/selim-ba/computer-vision/blob/main/gif/cyclegan_cycle_monet.gif)

----------

## 3. Real-Time Urban Scene Segmentation : Mask R-CNN vs U-NET
Work In Progress
