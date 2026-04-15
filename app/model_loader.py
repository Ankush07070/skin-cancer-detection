import torch
from src.components.model import SkinCancerModel

def load_model(path, device):
    model = SkinCancerModel(num_classes=7)
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model