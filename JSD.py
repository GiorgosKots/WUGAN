import torch
import numpy as np
from itertools import product
from collections import Counter
from scipy.spatial.distance import jensenshannon

# Set device based on availability of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Convert one-hot or Gumbel-softmax output to DNA string
def tensor_to_dna(seq_tensor):
    idx_to_base = ['P', 'A', 'T', 'G', 'C']
    indices = torch.argmax(seq_tensor, dim=-1)
    return ''.join([idx_to_base[i] for i in indices.tolist() if idx_to_base[i] != 'P'])

# 2. Get k-mer distribution
def get_kmer_distribution(seqs, k=6):
    all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    kmer_counts = Counter()
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_counts[kmer] += 1
    total = sum(kmer_counts.values())
    freq = np.array([kmer_counts[kmer] for kmer in all_kmers], dtype=np.float32)
    return freq / (total + 1e-8)

# 3. Evaluate per epoch - MODIFIED to handle multiple k values
def evaluate_kmer_jsd(real_batch, gen_batch, k_values=[3,4,5,6]):
    real_seqs = [tensor_to_dna(seq) for seq in real_batch]
    gen_seqs  = [tensor_to_dna(seq) for seq in gen_batch]

    jsds = []
    for k in k_values:
        real_dist = get_kmer_distribution(real_seqs, k=k)
        gen_dist  = get_kmer_distribution(gen_seqs, k=k)

        if np.all(real_dist == 0) and np.all(gen_dist == 0):
            jsd_k = 0.0
        else:
            jsd_k = jensenshannon(real_dist, gen_dist)

        jsds.append(jsd_k)
    
    # Return average JSD across all k values
    return float(np.mean(jsds))

def jsd(real_sequences, generated_sequences, k_values=[4,5,6]):
    """
    Calculate JSD between real and generated sequences.
    
    Args:
        real_sequences: Tensor of real sequences
        generated_sequences: Tensor of generated sequences
        k_values: List of k-mer sizes to use
    
    Returns:
        Average JSD score
    """
    return evaluate_kmer_jsd(real_sequences, generated_sequences, k_values=k_values)
# Example usage:
# from JSD import jsd
# avg_jsd = jsd(generator, dataloader)
