import torch
import lightning as L
from torch.utils.data import DataLoader
from model import Transformer
from dataset import create_dataloaders

def train_transformer():
    # Create dataloaders
    train_loader, val_loader, test_loader, vocab_size = create_dataloaders(
        batch_size=32,
        num_train=1000,
        num_val=200,
        num_test=100
    )
    
    # Initialize model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,  # Smaller model for faster training
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        dropout=0.1,
        learning_rate=0.0001
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        logger=True,
    )
    
    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test model
    trainer.test(model, test_loader)
    
    # Example inference
    print("\nTesting inference on sample sequences:")
    model.eval()
    with torch.no_grad():
        for i, (src, tgt) in enumerate(test_loader):
            if i >= 3:  # Only show first 3 examples
                break
                
            output = model(src, src)  # Using same sequence for encoder and decoder
            # Get the highest probability tokens
            pred = output.argmax(dim=-1)
            
            print(f"\nExample {i+1}:")
            print(f"Input sequence:  {src[0].tolist()}")
            print(f"Target sequence: {tgt[0].tolist()}")
            print(f"Model output:    {pred[0].tolist()}")

if __name__ == "__main__":
    train_transformer()
