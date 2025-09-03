import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SequenceDataset(Dataset):
    def __init__(self, file_path, seq_length=156):
        self.seq_length = seq_length
        self.sequences = []
        self.char_to_idx = {'P': 0, 'A': 1, 'T': 2, 'G': 3, 'C': 4}

        with open(file_path, 'r') as f:
            for line in f:
                seq = line.strip().split('\t')[0] # Remove the '1' and whitespace

                if len(seq) < seq_length: # Padding with 'P' 
                    seq = seq + 'P' * (seq_length - len(seq))
                seq = seq[:seq_length]  # Truncate if too long

                self.sequences.append(seq)

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
                one_hot[i][self.char_to_idx['P']] = 1  # Use padding for unknown chars

        return one_hot

# Test the dataset
if __name__ == "__main__":
    file_path = r"C:\Users\kotsgeo\Documents\GANs\Old\AMPdata.txt"
    dataset = SequenceDataset(file_path)

    # Print first sequence
    first_seq = dataset[0]
    print("First sequence shape:", first_seq.shape)
    print("\nMapping:", dataset.char_to_idx)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    print('\nNumber of sequences:', len(dataloader.dataset))
    print("\nNumber of batches in the DataLoader:", len(dataloader))
    # Check a batch
    for batch in dataloader:
        print("Batch shape:", batch.shape)
        print("\nSample from batch (showing where P padding is):")
        print(torch.argmax(batch[0], dim=1)[:10])  
        break