import numpy as np
from scipy.linalg import sqrtm
from transformers import AutoTokenizer, AutoModel
import torch
from Bio.Seq import Seq

class FREDCalculator:
    def __init__(self):
        # Load ProtBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        self.model = AutoModel.from_pretrained("Rostlab/prot_bert")

    def make_multiple_of_three(self, dna_seq):
        """Ensure DNA sequence length is a multiple of 3 by adding 'N' nucleotides"""
        remainder = len(dna_seq) % 3
        if remainder != 0:
            dna_seq += 'N' * (3 - remainder)
        return dna_seq

    def translate_dna_to_protein(self, dna_seq):
        """Translate DNA sequence to protein sequence"""
        try:
            # Ensure sequence is multiple of 3
            dna_seq = self.make_multiple_of_three(dna_seq)
            return str(Seq(dna_seq).translate())
        except:
            return ""  # Return empty string if translation fails

    def extract_features(self, protein_seq):
        """Extract features from protein sequence using ProtBERT"""
        if not protein_seq:
            return None

        # Tokenize the protein sequence
        inputs = self.tokenizer(protein_seq, return_tensors="pt", padding=True, truncation=True, max_length=1024)

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling to get a fixed-size feature vector
        features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return features

    def calculate_fred(self, real_features, generated_features):
        """Calculate Fréchet Reconstruction Distance (FReD)"""
        # Calculate mean and covariance for real features
        mu_r = np.mean(real_features, axis=0)
        sigma_r = np.cov(real_features, rowvar=False)

        # Calculate mean and covariance for generated features
        mu_g = np.mean(generated_features, axis=0)
        sigma_g = np.cov(generated_features, rowvar=False)

        # Calculate the difference between means
        diff = mu_r - mu_g

        # Calculate the square root of the product of covariance matrices
        covmean = sqrtm(sigma_r.dot(sigma_g))

        # Ensure covmean is real
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Calculate FReD
        fred = np.sum(diff**2) + np.trace(sigma_r + sigma_g - 2 * covmean)

        return fred

    def calculate_fred_from_sequences(self, real_seqs, generated_seqs):
        """Calculate FReD directly from DNA sequences"""
        # Ensure sequences are multiples of 3
        real_seqs = [self.make_multiple_of_three(seq) for seq in real_seqs]
        generated_seqs = [self.make_multiple_of_three(seq) for seq in generated_seqs]

        # Translate DNA to protein
        real_protein_seqs = [self.translate_dna_to_protein(seq) for seq in real_seqs]
        generated_protein_seqs = [self.translate_dna_to_protein(seq) for seq in generated_seqs]

        # Extract features
        real_features = [self.extract_features(seq) for seq in real_protein_seqs]
        generated_features = [self.extract_features(seq) for seq in generated_protein_seqs]

        # Remove None values
        real_features = [f for f in real_features if f is not None]
        generated_features = [f for f in generated_features if f is not None]

        # Calculate FReD
        return self.calculate_fred(real_features, generated_features)