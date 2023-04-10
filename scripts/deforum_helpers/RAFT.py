import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from pathlib import Path
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image


class RAFTModel:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._initialize()
        return instance

    def _initialize(self):
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        self.model = self.model.eval()

    @staticmethod
    def plot(imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
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

    @staticmethod
    def preprocess(img1_batch, img2_batch, transforms):
        img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
        img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
        return transforms(img1_batch, img2_batch)

    def predict_flows(self, img1_batch, img2_batch):
        img1_batch, img2_batch = self.preprocess(img1_batch, img2_batch, self.transforms)
        list_of_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))
        predicted_flows = list_of_flows[-1]
        return predicted_flows

    @staticmethod
    def flow_to_image(predicted_flows):
        flow_imgs = flow_to_image(predicted_flows)
        return flow_imgs
