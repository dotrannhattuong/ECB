import clip
import torch

class CustomCLIPModel(torch.nn.Module):
    def __init__(self, name="ViT-B/32", device='cpu'):
        super(CustomCLIPModel, self).__init__()
        self.model, transform = clip.load(name, device=device)

        del self.model.transformer

    def forward(self, image):
        with torch.no_grad():
            image_features = self.model.encode_image(image).to(dtype=torch.float32)
            return image_features / image_features.norm(dim=-1, keepdim=True)