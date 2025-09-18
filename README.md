# UNET-WGAN: A U-NET BASED ADVERSARIAL FRAMEWORK FOR GLOBAL AND LOCAL FEEDBACK IN ANTIMICROBIAL PEPTIDE GENERATION

A deep learning framework for **de novo antimicrobial peptide (AMP) design**. 
**UNet-WGAN** combines a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** and a **U-Net discriminator** with **dual loss functions** to generate **novel, diverse, and biologically plausible peptides**.  

---

## 📌 Overview

Antimicrobial resistance (AMR) is a major global health threat, driving the need for **novel therapeutic peptides**. UNet-WGAN addresses this by generating peptide-like DNA sequences that translate into realistic AMP candidates.  

Unlike prior GAN approaches, UNet-WGAN:  
- Uses a **U-Net discriminator** for both **global (sequence-level)** and **local (nucleotide-level)** supervision.  
- Couples **CutMix augmentation** with the decoder head, enabling **robust local feedback**.  
- Trains without external classifiers or conditional labels, relying solely on **architecture-driven supervision**.
- 
### ✨ Key Features  
- ✅ **WGAN-GP backbone** → stable adversarial training  
- ✅ **U-Net discriminator with dual losses** → global + local sequence fidelity  
- ✅ **CutMix augmentation** → hybrid inputs for stronger local supervision  
- ✅ **Biological evaluation metrics** → JSD, AMP scores, ORF validity, CAMPR4  

---

## 🔬 How It Works  

### 1. Data Preparation  
- **Input**: Experimentally validated AMP datasets (APD3, DRAMP, CAMP, etc.).  
- **Representation**: Sequences mapped to DNA, one-hot encoded over `{A, T, G, C, P}` (P = padding).  
- **Augmentation**: CutMix blends real and generated subsequences for decoder supervision.  

### 2. Model Architecture  

**Generator**  
- Maps random latent vectors to peptide-like DNA sequences.  
- Outputs categorical distributions via **Gumbel–Softmax**, ensuring differentiability.  

**U-Net Discriminator (Core Idea)**  
- **Global output**: Wasserstein score (real/fake at sequence level).  
- **Decoder head**: Per-nucleotide probabilities, trained jointly with CutMix masks.  
- **Dual-loss setup**: Combines adversarial loss + local residue-level loss.  

### 3. Training Objectives  
- **Generator loss** = adversarial + decoder (pixel-wise) loss.  
- **Discriminator loss** = adversarial + gradient penalty + decoder loss.  
- **Training strategy**: $d_{step}=5$, $g_{step}=2$, Adam optimizer with gradient clipping.  

### 4. Evaluation  
- **Jensen–Shannon Divergence (JSD)**: $k$-mer (3–6) distribution similarity.  
- **Physicochemical AMP score**: length 5–60, charge +1–12, hydrophobicity 30–60%, Lys/Arg 10–50%.  
- **ORF validity**: ensures generated DNA translates into peptides.  
- **CAMPR4 benchmark**: external RF/SVM/ANN classifiers for AMP potential.  

---

## 💡 Why It’s Different  

1. **Dual-Loss U-Net Discriminator**  
   - Captures **sequence-level realism** and **motif-level details** simultaneously.  

2. **CutMix for Sequences**  
   - First adaptation of **CutMix** to 1D DNA/peptide sequences.  
   - Applied only through the decoder head for robust local supervision.  

3. **Architecture-Driven Training**  
   - No external feedback classifiers.  
   - Faster, simpler, and still competitive with classifier-based models.  

---

## 🧪 Applications  

- **Drug discovery** → design of novel AMPs against resistant bacteria  
- **Synthetic biology** → peptide generation for industrial and biomedical use  
- **Bioinformatics** → framework for generative sequence modeling  

---

## 📊 Results (Highlights)  

- **CAMPR4 score (threshold 0.5):** UNet-WGAN 73.5% ± 6.1 → higher than AMPGAN, HydrAMP, FBGAN-ESM2, RLGen.  
- **Diversity:** 100% novel peptides (no >80% identity to training set).  
- **Similarity:** Intra-set similarity ~28.9%, comparable to top baselines (26–28%).  

---

## 🛠 Installation  

```bash
# Clone repository
git clone https://github.com/yourusername/UNet-WGAN.git
cd UNet-WGAN

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (recommended)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

