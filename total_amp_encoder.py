import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class AMPSequenceDataset(Dataset):
    def __init__(self, file_path, seq_length=156):
        self.seq_length = seq_length
        self.sequences = []
        self.sequence_ids = []
        self.char_to_idx = {'P': 0, 'A': 1, 'T': 2, 'G': 3, 'C': 4}
        
        # Read FASTA file
        with open(file_path, 'r') as f:
            current_seq = ""
            current_id = ""
            
            for line in f:
                line = line.strip()
                
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_seq:
                        # Process sequence
                        if len(current_seq) < seq_length:
                            # Padding with 'P'
                            current_seq = current_seq + 'P' * (seq_length - len(current_seq))
                        current_seq = current_seq[:seq_length]  # Truncate if too long
                        
                        self.sequences.append(current_seq)
                        self.sequence_ids.append(current_id)
                    
                    # Start new sequence
                    current_id = line[1:]  # Remove '>'
                    current_seq = ""
                else:
                    # Accumulate sequence
                    current_seq += line.upper()  # Ensure uppercase
            
            # Don't forget the last sequence
            if current_seq:
                if len(current_seq) < seq_length:
                    current_seq = current_seq + 'P' * (seq_length - len(current_seq))
                current_seq = current_seq[:seq_length]
                
                self.sequences.append(current_seq)
                self.sequence_ids.append(current_id)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Create one-hot encoding
        one_hot = torch.zeros(self.seq_length, len(self.char_to_idx))
        for i, char in enumerate(seq[:self.seq_length]):  # Ensure we don't exceed length
            if char in self.char_to_idx:  # Check if character is valid
                one_hot[i][self.char_to_idx[char]] = 1
            else:
                # Handle unknown characters (shouldn't happen with DNA, but just in case)
                one_hot[i][self.char_to_idx['P']] = 1  # Use padding for unknown chars
        
        return one_hot
    
    def get_sequence_id(self, idx):
        """Get the ID of a specific sequence"""
        return self.sequence_ids[idx]
    
    def decode_one_hot(self, one_hot_tensor):
        """Decode a one-hot tensor back to sequence string"""
        idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        indices = torch.argmax(one_hot_tensor, dim=-1)
        sequence = ''.join([idx_to_char[idx.item()] for idx in indices])
        return sequence


# # Test the dataset
# if __name__ == "__main__":
#     file_path = "all_amp.fasta"
#     dataset = AMPSequenceDataset(file_path)
    
#     # Create dataloader for training
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    
#     # Quick test
#     for batch in dataloader:
#         print(f"Batch shape: {batch.shape}")
#         break