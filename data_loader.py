
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from config import *
from datasets import *

cfg = get_cfg()

transforms_ = [
    transforms.Resize((cfg.img_size, cfg.img_size), InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_loader = DataLoader(
    ImageDataset(transforms_=transforms_),
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.n_cpu,
)

test_loader = DataLoader(
    ImageDataset(transforms_=transforms_, mode="val"),
    batch_size=cfg.test_bs,
    shuffle=True,
    num_workers=cfg.n_cpu,
)