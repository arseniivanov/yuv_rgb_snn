from torch.utils.data import Dataset
from torchvision.io import read_image
import glob
import torch
import os

def rgb_to_yuv(img):
    if isinstance(img, torch.Tensor):
        if len(img.shape) == 4:
            img = img.permute(0, 2, 3, 1)  # Change (B,C,H,W) to (B,H,W,C) for pytorch tensor
        else:
            img = img.permute(1, 2, 0)  # Change (C,H,W) to (H,W,C) for pytorch tensor
    img = img.to(torch.float32)
    in_img = img.contiguous().view(-1, 3)
    out_img = torch.mm(in_img, torch.tensor([[0.299, -0.14713, 0.615],
                                             [0.587, -0.28886, -0.51499],
                                             [0.114, 0.436, -0.10001]]).t()).view(*img.shape)
    return out_img

class DogsVsCatsDataset(Dataset):
    def __init__(self, img_dir, color_space='rgb', transform=None):
        self.img_dir = img_dir
        self.color_space = color_space
        self.transform = transform
        self.img_labels = self.get_image_labels()

    def get_image_labels(self):
        image_files = glob.glob(os.path.join(self.img_dir, '*.jpg'))
        labels = [os.path.split(filename)[-1].split('.')[0] for filename in image_files]
        labels = [0 if label == "cat" else 1 for label in labels]
        return list(zip(image_files, labels))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        img = read_image(img_path).float()
        img /= 255.0  # Normalize to [0,1]
        if self.color_space == 'yuv':
            img = rgb_to_yuv(img)
        if self.transform:
            img = self.transform(img)
        return img, label
