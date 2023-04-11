import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image

class RAFTModel:
    def __init__(self):
        plt.rcParams["savefig.bbox"] = "tight"
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        self.model = self.model.eval()

    @staticmethod
    def plot(imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()

    def preprocess(self, img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[512, 512], antialias=False)
        img2_batch = F.resize(img2_batch, size=[512, 512], antialias=False)
        return self.transforms(img1_batch, img2_batch)

    def predict_flow(self, img1_batch, img2_batch):
        img1_batch, img2_batch = self.preprocess(img1_batch, img2_batch)
        list_of_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))
        return list_of_flows[-1]

    def visualize(self, img1_batch, predicted_flows):
        flow_imgs = flow_to_image(predicted_flows)
        img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
        grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
        self.plot(grid)