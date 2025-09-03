import numpy as np
import torch

def random_vector_boundingbox(length, lam):
    # Calculate box length based on lambda
    r = np.sqrt(1. - lam)
    box_length = int(length * r)
    
    # Randomly select start position
    start = np.random.randint(0, length - box_length + 1)
    
    # Calculate end position
    end = start + box_length
    
    return start, end

def create_cutmix_mask(real_sequences, lam=None):
    batch_size, seq_len = real_sequences.size(0), real_sequences.size(1)
    
    # Create mask for each sequence in the batch
    masks = []
    for i in range(batch_size):
        # Generate lambda if not provided
        if lam is None:
            lam = np.random.beta(1, 1)
        
        # Create mask for this sequence
        mask = torch.ones(seq_len)
        start, end = random_vector_boundingbox(seq_len, lam)
        mask[start:end] = 0
        
        # 50% chance to flip mask
        if torch.rand(1) > 0.5:
            mask = 1 - mask
        
        masks.append(mask)
    
    # Stack masks and add channel dimension
    return torch.stack(masks).unsqueeze(-1).to(real_sequences.device)