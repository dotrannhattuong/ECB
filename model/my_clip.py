import clip
import torch

class CustomCLIPModel(torch.nn.Module):
    def __init__(self, name="ViT-B/32", device='cpu'):
        super(CustomCLIPModel, self).__init__()
        self.model, _ = clip.load(name, device=device)

    def forward(self, image):
        image_features = self.model.encode_image(image).to(dtype=torch.float32)
        return image_features