# GAN Competition - Monet Style Transfer

A CycleGAN implementation for transforming photographs into Monet-style paintings, developed for the [Kaggle GAN Getting Started Competition](https://www.kaggle.com/competitions/gan-getting-started/overview).

## Overview

This project implements a CycleGAN (Cycle-Consistent Adversarial Network) to perform unpaired image-to-image translation between photographs and Monet-style paintings. The model learns to transform regular photos into artistic renditions that mimic Claude Monet's impressionist painting style.

## Features

- **CycleGAN Architecture**: Dual generator-discriminator pairs for bidirectional image translation
- **PatchGAN Discriminator**: Enhanced discrimination using patch-based adversarial loss
- **Residual Blocks**: Deep residual learning for improved feature transformation
- **PyTorch Lightning Integration**: Clean, scalable training pipeline
- **Weights & Biases Logging**: Comprehensive experiment tracking and visualization
- **Custom Data Loading**: Efficient data pipeline with augmentation

## Architecture

### Generators
- **Encoder**: Convolutional layers with downsampling (3→32→64→128 channels)
- **Transformer**: 3 residual blocks for feature transformation
- **Decoder**: Transposed convolutions with upsampling (128→64→32→3 channels)

### Discriminators
- **PatchGAN**: 4-layer convolutional discriminator with instance normalization
- Outputs patch-wise predictions for improved texture discrimination

### Loss Functions
- **Adversarial Loss**: MSE loss for generator-discriminator competition
- **Cycle Consistency Loss**: L1 loss ensuring round-trip reconstruction (coefficient: 5)
- **Identity Loss**: L1 loss preserving color composition (coefficient: 2)

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.5.0+ (with CUDA 11.8 support)
- PyTorch Lightning 2.4.0
- torchvision 0.20.0+
- Weights & Biases (wandb)
- OpenCV, Pillow, matplotlib

## Dataset Structure

Organize your data in the following structure:

```
data/
├── train_photos/     # Regular photographs for training
├── train_monet/      # Monet paintings for training
├── test_photos/      # Test photographs
└── test_monet/       # Test Monet paintings (optional)
```

## Usage

### Training

Run the main training script:

```bash
python main.py
```

Configuration options in `main.py`:
- `epochs`: Number of training epochs (default: 30)
- `batch_size`: Batch size for training (default: 16)
- `img_size`: Image resolution (default: 256x256)
- `g_lr`: Generator learning rate (default: 0.0002)
- `d_lr`: Discriminator learning rate (default: 0.0002)
- `rec_coef`: Cycle consistency loss coefficient (default: 5)
- `id_coef`: Identity loss coefficient (default: 2)

### Loading from Checkpoint

To resume training from a checkpoint:

```python
cycle_gan = CycleGAN.load_from_checkpoint(
    "./path/to/checkpoint.ckpt",
    gx=generator_x,
    gy=generator_y,
    dx=discriminator_x,
    dy=discriminator_y,
    g_lr=0.0002,
    d_lr=0.0002,
    rec_coef=5,
    id_coef=2,
)
```

### Inference

```python
import torch
from models import GeneratorGAN

# Load trained generator
generator = GeneratorGAN()
checkpoint = torch.load("path/to/checkpoint.ckpt")
generator.load_state_dict(checkpoint['state_dict']['gx'])
generator.eval()

# Transform image
with torch.no_grad():
    styled_image = generator(input_image)
```

## Project Structure

```
GAN-Competition/
├── main.py              # Main training script
├── models.py            # Generator, Discriminator, and CycleGAN model definitions
├── dataloader.py        # Dataset and DataModule classes
├── utils.py             # Weight initialization utilities
├── requirements.txt     # Project dependencies
├── README.md            # This file
└── LICENSE              # License information
```

## Model Components

### ResidualBlock (`models.py:8`)
Standard residual block with batch normalization and skip connections.

### GeneratorGAN (`models.py:41`)
U-Net style generator with residual transformation blocks.

### DiscriminatorPatchGAN (`models.py:80`)
PatchGAN discriminator with instance normalization layers.

### CycleGAN (`models.py:104`)
Complete CycleGAN training module with:
- Dual generators (X→Y and Y→X)
- Dual discriminators (for X and Y domains)
- Custom training loop with alternating generator/discriminator updates
- Automatic image logging every 50 steps

## Training Strategy

The model uses a custom training regime:
1. **Steps 0-40**: Train generators (adversarial + cycle consistency + identity loss)
2. **Steps 41-80**: Train discriminators
3. Repeat cycle throughout training

This alternating schedule helps stabilize GAN training and prevent mode collapse.

## Logging and Monitoring

Training metrics are logged to Weights & Biases:
- `loss_g`: Total generator loss
- `loss_d`: Total discriminator loss
- `validity`: Adversarial loss component
- `recon`: Cycle consistency loss
- `identity`: Identity preservation loss
- Sample images: Real and generated images logged every 50 steps

## Results

The model learns to:
- Transform photographs into Monet-style impressionist paintings
- Preserve content structure while applying artistic style
- Maintain cycle consistency (photo → painting → photo)
- Generate realistic textures and color palettes characteristic of Monet's work

## License

Copyright (c) 2025 EgoHackZero. All rights reserved.

See LICENSE file for details.

## Acknowledgments

- Based on the CycleGAN architecture by Zhu et al. ([Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593))
- Developed for the Kaggle GAN Getting Started Competition
- Built with PyTorch Lightning for scalable deep learning

## Author

**EgoHackZero**

For questions, issues, or contributions, please contact the author or open an issue in the repository.
