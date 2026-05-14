from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
    
    

@register_dataset(name='brain_mri')
class BrainMRIDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        
        # Load .npy — now (50, 248, 248) float32
        self.images = np.load(root)
        assert self.images.ndim == 3, f"Expected (N,H,W), got {self.images.shape}"
        print(f"Loaded {len(self.images)} images, shape: {self.images.shape}, "
              f"range: [{self.images.min():.4f}, {self.images.max():.4f}]")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        img = self.images[index]                         # (248, 248) float32 [0,1]
        img = torch.from_numpy(img.astype(np.float32))  # (248, 248)
        img = img.unsqueeze(0).unsqueeze(0)              # (1, 1, 248, 248)
        img = torch.nn.functional.interpolate(
                img, size=(256, 256),
                mode='bilinear',
                align_corners=False)                     # (1, 1, 256, 256)
        img = img.squeeze(0).repeat(3, 1, 1)             # (3, 256, 256)

        # Normalize [0,1] → [-1,1] to match fine-tuning convention
        img = img * 2.0 - 1.0

        return img