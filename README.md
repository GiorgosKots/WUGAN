# WUGAN: Antimicrobial Peptide Generation with GANs

WUGAN is a project aimed at generating antimicrobial peptides (AMPs) using a Generative Adversarial Network (GAN) architecture. This project leverages the Wasserstein loss function and a U-Net-based generator to create novel AMP sequences.

## Overview

- **Objective**: Generate antimicrobial peptides with potential therapeutic applications.
- **Technology**: Utilizes GANs with Wasserstein loss for stable training and a U-Net architecture for effective sequence generation.
- **Innovation**: Focuses on the unique application of GANs in the field of bioinformatics, specifically for generating AMPs.

## How It Works

1. **Data Preparation**: The model is trained on a dataset of known antimicrobial peptide sequences.
2. **Model Architecture**: The GAN consists of a generator and a discriminator. The generator produces peptide sequences, while the discriminator uses a U-Net architecture to evaluate their authenticity.
3. **Training**: The model is trained using the Wasserstein loss function, which helps in achieving more stable training dynamics and improved output quality.
4. **Generation**: Once trained, the generator can produce new peptide sequences that have potential antimicrobial properties.
