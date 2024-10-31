import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from model import Transformer

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(42)

# Test parameters
batch_size = 4
src_seq_length = 10
tgt_seq_length = 10  # Making source and target same length for simplicity
src_vocab_size = 1000
tgt_vocab_size = 1000
d_model = 512
num_heads = 8

# Create random input data
src = torch.randint(0, src_vocab_size, (batch_size, src_seq_length))
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length))

# Create dataset and dataloader
dataset = TensorDataset(src, tgt)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Initialize model
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads
)

# Initialize trainer
trainer = L.Trainer(
    max_epochs=1,
    accelerator='auto',
    devices=1,
    enable_progress_bar=True,
    logger=False
)

def test_transformer():
    print("Testing forward pass...")
    try:
        # Get a batch
        src_batch, tgt_batch = next(iter(dataloader))
        
        # Prepare target input (remove last token) and output (remove first token)
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        
        # Forward pass
        output = model(src_batch, tgt_input, None, None)
        
        # Check output shape
        expected_shape = (batch_size, tgt_seq_length - 1, tgt_vocab_size)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        # Verify loss calculation
        loss = torch.nn.functional.cross_entropy(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        
        print("\nShapes:")
        print(f"Source: {src_batch.shape}")
        print(f"Target input: {tgt_input.shape}")
        print(f"Target output: {tgt_output.shape}")
        print(f"Model output: {output.shape}")
        print(f"Initial loss: {loss.item()}")
        
        # Test training
        print("\nTesting training...")
        trainer.fit(model, dataloader)
        print("Training successful!")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_transformer()

