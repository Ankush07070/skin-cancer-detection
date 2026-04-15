import torch
import os

def save_checkpoint(model, optimizer, epoch, path="artifacts/model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)

    print(f"💾 Model saved at epoch {epoch+1}")


def load_checkpoint(model, optimizer=None, path="artifacts/model.pth", device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"📂 Loaded model from epoch {checkpoint['epoch']+1}")
    return checkpoint["epoch"]