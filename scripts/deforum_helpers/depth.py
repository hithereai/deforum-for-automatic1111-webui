import math, os, subprocess
import cv2
import hashlib
import numpy as np
import torch
import gc
import torchvision.transforms as T
from einops import rearrange, repeat
from PIL import Image
from basicsr.utils.download_util import load_file_from_url
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms.functional as TF
from .general_utils import checksum
from modules import lowvram, devices
from modules.shared import opts
from .depth_zoe import ZoeDepth
from .depth_adabins import AdaBinsModel

class DepthModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        keep_in_vram = kwargs.get('keep_in_vram', False)
        use_zoe_depth = kwargs.get('use_zoe_depth', False)
        Width = kwargs.get('Width', 512)
        Height = kwargs.get('Height', 512)
        model_switched = cls._instance and cls._instance.use_zoe_depth != use_zoe_depth
        resolution_changed = cls._instance and (cls._instance.Width != Width or cls._instance.Height != Height)


        if cls._instance is None or (not keep_in_vram and not hasattr(cls._instance, 'midas_model')) or model_switched or resolution_changed:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(models_path=args[0], device=args[1], half_precision=True, keep_in_vram=keep_in_vram, use_zoe_depth=use_zoe_depth, Width=Width, Height=Height)
        elif cls._instance.should_delete and keep_in_vram:
            cls._instance._initialize(models_path=args[0], device=args[1], half_precision=True, keep_in_vram=keep_in_vram, use_zoe_depth=use_zoe_depth, Width=Width, Height=Height)
        cls._instance.should_delete = not keep_in_vram
        return cls._instance
        
    def _initialize(self, models_path, device, half_precision=True, keep_in_vram=False, use_zoe_depth=False, Width=512, Height=512):
        self.keep_in_vram = keep_in_vram
        self.Width = Width
        self.Height = Height
        self.adabins_helper = None
        self.depth_min = 1000 # for saving func
        self.depth_max = -1000 # for saving func
        self.device = device
        self.use_zoe_depth = use_zoe_depth
        # midas params, might be moved laters:
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.midas_transform = T.Compose([
            Resize(384, 384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32,
                   resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
            self.normalization,
            PrepareForNet()
        ])
        
        if self.use_zoe_depth:
            self.zoe_depth = ZoeDepth(self.Width, self.Height)
        if not self.use_zoe_depth:
            midas_model_filename = 'dpt_large-midas-2f21e586.pt'
            self.check_and_download_midas_model(models_path, midas_model_filename)

            if not self.keep_in_vram or not hasattr(self, 'midas_model'):
                self.load_midas_model(models_path, midas_model_filename)
            if half_precision:
                self.midas_model = self.midas_model.half()
                
    def load_midas_model(self, models_path, midas_model_filename):
        model_file = os.path.join(models_path, midas_model_filename)
        print("Loading MiDaS model...")
        self.midas_model = DPTDepthModel(
            path=model_file,
            backbone="vitl16_384",
            non_negative=True,
        )
        self.midas_model.eval().to(self.device, memory_format=torch.channels_last if self.device == torch.device("cuda") else None)
         
    def check_and_download_midas_model(self, models_path, midas_model_filename):
        model_file = os.path.join(models_path, midas_model_filename)
        if not os.path.exists(model_file):
            load_file_from_url(r"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", models_path)
            if checksum(model_file) != "fcc4829e65d00eeed0a38e9001770676535d2e95c8a16965223aba094936e1316d569563552a852d471f310f83f597e8a238987a26a950d667815e08adaebc06":
                raise Exception(f"Error while downloading dpt_large-midas-2f21e586.pt. Please download from here: https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt and place in: {models_path}")
        
    def predict(self, prev_img_cv2, midas_weight, half_precision) -> torch.Tensor:
        use_adabins = midas_weight < 1.0 and self.adabins_helper is not None

        img_pil = self.pil_image_from_cv2_image(prev_img_cv2)

        midas_depth = None
        if self.use_zoe_depth:
            depth_tensor = self.predict_depth_with_zoe(img_pil)
        else:
            depth_tensor = self.predict_depth_with_midas(prev_img_cv2, half_precision)

        self.debug_print("Shape of depth_tensor:", depth_tensor.shape)
        self.debug_print("Tensor data:", depth_tensor)
        
        if use_adabins: # need to use AdaBins. first, try to get adabins depth estimation from our image
            use_adabins, adabins_depth = AdaBinsModel._instance.predict(img_pil, prev_img_cv2, use_adabins)
            if use_adabins: # if there was no error in getting the depth, align other depths (midas/zoe/leres) with adabins' depth
                depth_tensor = self.blend_and_align_with_adabins(depth_tensor, adabins_depth, midas_weight, self.use_zoe_depth)

        return depth_tensor

    def pil_image_from_cv2_image(self, img_cv2):
        return Image.fromarray(cv2.cvtColor(img_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR))

    def predict_depth_with_zoe(self, img_pil):
        return self.zoe_depth.predict(img_pil).to(self.device)

    def predict_depth_with_midas(self, prev_img_cv2, half_precision):
        img_midas = prev_img_cv2.astype(np.float32) / 255.0
        img_midas_input = self.midas_transform({"image": img_midas})["image"]
        sample = torch.from_numpy(img_midas_input).float().to(self.device).unsqueeze(0)

        if self.device.type == "cuda" or self.device.type == "mps":
            sample = sample.to(memory_format=torch.channels_last)
            if half_precision:
                sample = sample.half()

        with torch.no_grad():
            midas_depth = self.midas_model.forward(sample)
        midas_depth = torch.nn.functional.interpolate(
            midas_depth.unsqueeze(1),
            size=img_midas.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        torch.cuda.empty_cache()
        depth_tensor = torch.from_numpy(np.expand_dims(midas_depth, axis=0)).squeeze().to(self.device)
        
        return depth_tensor
        
    def blend_and_align_with_adabins(self, depth_tensor, adabins_depth, midas_weight, use_zoe_depth):
        depth_tensor = torch.subtract(50.0, depth_tensor) / 19.0
        blended_depth_map = (depth_tensor.cpu().numpy() * midas_weight + adabins_depth * (1.0 - midas_weight))
        depth_tensor = torch.from_numpy(np.expand_dims(blended_depth_map, axis=0)).squeeze().to(self.device)
        return depth_tensor

    def to_image(self, depth: torch.Tensor):
        depth = depth.cpu().numpy()
        depth = np.expand_dims(depth, axis=0) if len(depth.shape) == 2 else depth
        self.depth_min = min(self.depth_min, depth.min())
        self.depth_max = max(self.depth_max, depth.max())
        denom = max(1e-8, self.depth_max - self.depth_min)
        temp = rearrange((depth - self.depth_min) / denom * 255, 'c h w -> h w c')
        temp = repeat(temp, 'h w 1 -> h w c', c=3)
        return Image.fromarray(temp.astype(np.uint8))

    def save(self, filename: str, depth: torch.Tensor):
        self.to_image(depth).save(filename)

    def to(self, device):
        self.device = device
        if self.use_zoe_depth:
            self.zoe_depth.zoe.to(device)
        else:
            self.midas_model.to(device)
        if self.adabins_helper is not None:
            self.adabins_helper.to(device)
        gc.collect()
        torch.cuda.empty_cache()

    def delete_model(self):
        if self.use_zoe_depth:
            self.zoe_depth.delete()
            del self.zoe_depth
        else:
            del self.midas_model
        gc.collect()
        torch.cuda.empty_cache()
        devices.torch_gc()
    
    def debug_print(self, message, tensor=None):
        DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)
        if DEBUG_MODE:
            print(message)
            if tensor is not None:
                print(tensor)