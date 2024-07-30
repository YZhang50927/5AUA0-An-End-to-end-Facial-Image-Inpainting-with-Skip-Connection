import glob
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from config import *

cfg = get_cfg()

class ImageDataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = cfg.img_size
        self.mask_size = cfg.mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.jpg" % cfg.dataset_path))
        self.files = self.files[:-cfg.val_num] if mode == "train" else self.files[-cfg.val_num:]
    
    def apply_random_mask(self, img):
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part
    
    def apply_center_mask(self, img):
        loc = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, loc : loc + self.mask_size, loc : loc + self.mask_size] = 1

        return masked_img, loc

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        img = self.transform(img)
        
        if self.mode == "train":
            masked_img, aux = self.apply_random_mask(img)
        else:
            masked_img, aux = self.apply_center_mask(img)
        return img, masked_img, aux

    def __len__(self):
        return len(self.files)