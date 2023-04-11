import torch
from PIL import Image

class ZoeDepth:
    def __init__(self):
        repo = "isl-org/ZoeDepth"
        self.model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.zoe = self.model_zoe_n.to(self.DEVICE)
        
    def predict(self, image):
        depth_tensor = self.zoe.infer_pil(image, output_type="tensor")
        return depth_tensor

    def predict_from_local_file(self, filepath):
        image = Image.open(filepath).convert("RGB")
        depth_tensor = self.zoe.infer_pil(image, output_type="tensor")
        return depth_tensor

    def predict_from_url(self, url):
        image = Image.open(url).convert("RGB")
        depth_tensor = self.zoe.infer_pil(image, output_type="tensor")
        return depth_tensor

    def save_raw_depth(self, depth, filepath):
        depth.save(filepath, format='PNG', mode='I;16')

    def colorize_depth(self, depth):
        colored = self.zoe.colorize(depth)
        return colored

    def save_colored_depth(self, depth, filepath):
        colored = self.zoe.colorize(depth)
        Image.fromarray(colored).save(filepath)