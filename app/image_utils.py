print("🔥 image_utils loaded")

from PIL import Image
from torchvision import transforms

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)