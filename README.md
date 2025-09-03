# WUGAN: Generating Antimicrobial Peptides with Wasserstein U-Net GANs

![WUGAN Logo](https://via.placeholder.com/150) <!-- Replace with your project logo -->

A cutting-edge **Generative Adversarial Network (GAN)** framework for designing novel **antimicrobial peptides (AMPs)** with therapeutic potential. WUGAN combines **Wasserstein GAN with Gradient Penalty (WGAN-GP)** and a **U-Net discriminator** to generate biologically plausible peptide sequences.

---

## 📌 Overview

WUGAN addresses the critical need for **novel antimicrobial agents** by leveraging **deep generative models** to design peptides with potential therapeutic applications. Unlike traditional GANs, WUGAN employs a **U-Net-based discriminator** to evaluate sequences at both **global and local scales**, ensuring high-quality and biologically relevant outputs.

### Key Features:
✅ **Wasserstein GAN with Gradient Penalty (WGAN-GP)** for stable training
✅ **U-Net discriminator** for multi-scale sequence evaluation
✅ **CutMix augmentation** for robust training
✅ **Computational validation** with JSD and AMP metrics

---

## 🔬 How It Works

### 1. Data Preparation
- **Input**: Curated dataset of known antimicrobial peptide sequences
- **Preprocessing**: Sequences are encoded and augmented using **CutMix** to create hybrid real/fake samples
- **Output**: Training-ready tensor representations of peptide sequences

### 2. Model Architecture

#### Generator
- Transforms random noise into AMP sequences
- Optimized to fool the U-Net discriminator

#### U-Net Discriminator (Key Innovation)
- **Global Branch**: Evaluates overall sequence authenticity (Wasserstein loss)
- **Local Branch**: Focuses on fine-grained patterns (e.g., amino acid motifs) via sequence-wise classification
- **Decoder Loss**: Ensures local consistency in generated sequences

### 3. Training Process
- **Adversarial Training**: Generator and discriminator compete in a minimax game
- **Loss Functions**:
  - Wasserstein Loss: Measures distribution distance
  - Gradient Penalty: Enforces Lipschitz continuity
  - Decoder Loss: Ensures local sequence consistency
- **Stable Training**: WGAN-GP avoids mode collapse and vanishing gradients

### 4. Generation & Validation
- **Output**: Novel AMP sequences with potential antimicrobial activity
- **Validation**:
  - Computational metrics (JSD, AMP scores)

---

## 💡 Innovations

1. **U-Net Discriminator**
   - Captures hierarchical features in peptide sequences
   - Evaluates both global structure and local motifs

2. **Stable Training with WGAN-GP**
   - Avoids common GAN training issues
   - Ensures reliable convergence and high-quality outputs

3. **Biologically Informed Design**
   - Generates sequences with realistic antimicrobial properties
   - Incorporates biological constraints (charge, hydrophobicity)

4. **CutMix Augmentation**
   - Creates hybrid sequences during training
   - Improves discriminator robustness

---

## 🧪 Applications

- **Drug Discovery**: Accelerate design of novel AMPs against antibiotic-resistant pathogens
- **Synthetic Biology**: Generate custom peptides for biomedical/industrial applications
- **Computational Biology**: Framework for AI-driven peptide design

---

## 📈 Why WUGAN?

|      Traditional Methods      |      WUGAN Approach        |
|-------------------------------|----------------------------|
| Costly experimental screening |  Computational generation  |
|   Limited sequence diversity  |   High-throughput design   |
|   Manual peptide engineering  |   AI-driven optimization   |
|    Time-consuming process     | Rapid candidate generation |

WUGAN **automates AMP discovery** by:
- Generating diverse candidate sequences computationally
- Focusing on biologically plausible designs
- Reducing reliance on brute-force experimentation

---

## 🛠 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/WUGAN.git
cd WUGAN

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (recommended)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
