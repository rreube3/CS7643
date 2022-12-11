import argparse
import torch
import os
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import List, Tuple

from img_transform.transforms import EyeMaskCustomTransform, EyeDatasetCustomTransform

IMG_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    torch.nn.ConstantPad2d((0, 75, 0, 56), 0),
    EyeDatasetCustomTransform(mask_threshold=0.25),
])


LBL_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    torch.nn.ConstantPad2d((0, 75, 0, 56), 0),
    EyeMaskCustomTransform(mask_threshold=0.25),
])


class RetinaSegmentationDataset(Dataset):
    def __init__(self, rootdir: str,
                 basenames: List,
                 img_transforms: torch.nn.Module = IMG_TRANSFORMS,
                 lbl_transforms: torch.nn.Module = LBL_TRANSFORMS,
                 has_labels: bool = True):
        self._rootdir = rootdir
        self._basenames = basenames
        self._img_transforms = img_transforms
        self._lbl_transforms = lbl_transforms
        self._has_labels = has_labels
        
    def __len__(self) -> int:
        return len(self._basenames)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self._rootdir, "images", self._basenames[index])
        lbl_path = os.path.join(self._rootdir, "labels", self._basenames[index])

        with open(img_path, "rb") as f:
            img = pickle.load(f)
            # Apply the transforms for the image
            img = self._img_transforms(img)
            
        #'''
        if self._has_labels:
            with open(lbl_path, "rb") as f:
                lbl = pickle.load(f)
                # Apply the transforms for the labels
                lbl = self._lbl_transforms(lbl)
        else:
            lbl = torch.zeros_like(img)
            
        return img, lbl
        
