import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NumberSortingDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=6, min_number=0, max_number=99):
        """
        Create a dataset of number sequences and their sorted versions.
        Each sequence will have a start token (max_number + 1) and an end token (max_number + 2)
        
        Args:
            num_samples: Number of sequences to generate
            seq_length: Length of each sequence
            min_number: Minimum number in sequences
            max_number: Maximum number in sequences
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.min_number = min_number
        self.max_number = max_number
        self.vocab_size = max_number + 3  # +3 for start/end/pad tokens
        self.pad_token = 0
        self.start_token = max_number + 1
        self.end_token = max_number + 2
        
        # Generate data
        self.data = self._generate_data()
        
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Generate random sequence
            seq = np.random.randint(self.min_number + 1, self.max_number + 1, size=self.seq_length)
            
            # Create source sequence: [START] numbers [END]
            src = [self.start_token] + seq.tolist() + [self.end_token]
            
            # Create target sequence: [START] sorted_numbers [END]
            sorted_seq = np.sort(seq)
            tgt = [self.start_token] + sorted_seq.tolist() + [self.end_token]
            
            data.append((src, tgt))
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src), torch.tensor(tgt)

def create_dataloaders(batch_size=32, num_train=1000, num_val=200, num_test=100):
    """
    Create train, validation, and test dataloaders
    """
    # Create datasets
    train_dataset = NumberSortingDataset(num_samples=num_train)
    val_dataset = NumberSortingDataset(num_samples=num_val)
    test_dataset = NumberSortingDataset(num_samples=num_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, train_dataset.vocab_size

# Example usage and test
def test_dataset():
    print("Testing dataset generation...")
    
    # Create small test dataset
    dataset = NumberSortingDataset(num_samples=5, seq_length=4, max_number=9)
    
    print("\nDataset examples:")
    for i in range(5):
        src, tgt = dataset[i]
        print(f"\nExample {i+1}:")
        print(f"Input sequence:  {src.tolist()}")
        print(f"Target sequence: {tgt.tolist()}")
    
    # Test dataloader creation
    train_loader, val_loader, test_loader, vocab_size = create_dataloaders(
        batch_size=2,
        num_train=10,
        num_val=4,
        num_test=4
    )
    
    print("\nDataloader test:")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Test batch shapes
    src_batch, tgt_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Source batch shape: {src_batch.shape}")
    print(f"Target batch shape: {tgt_batch.shape}")

if __name__ == "__main__":
    test_dataset()
