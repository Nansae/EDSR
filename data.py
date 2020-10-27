import os
import random
import numpy as np
import torch

from PIL import Image
from natsort import natsorted
from torchvision import transforms as T

class Patches(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        self.phase = phase

        imgs_100x = []
        imgs_50x = []

        if phase == 'train' or phase == 'test':
            for path, subdirs, files in os.walk(root):
                if "50x" in path:
                    for fn in os.listdir(path):
                        p = os.path.join(path, fn)
                        if fn.endswith(".bmp") == True:
                            imgs_50x.append(p)
                if "100x" in path:
                    for fn in os.listdir(path):
                        p = os.path.join(path, fn)
                        if fn.endswith(".bmp") == True:
                            imgs_100x.append(p)

            self.imgs_100x = imgs_100x
            self.imgs_50x = imgs_50x

        if self.phase == 'train' or self.phase == 'test':
            self.transforms = T.Compose([
                T.ToTensor()
                # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __getitem__(self, index):
        if self.phase == 'train' or self.phase == 'test':
            path_100x = self.imgs_100x[index]
            data_100 = Image.open(path_100x).convert('RGB')
            data_100 = self.transforms(data_100)

            path_50x = self.imgs_50x[index]
            data_50 = Image.open(path_50x).convert('RGB')
            data_50 = self.transforms(data_50)

        return data_50, data_100

    def __len__(self):
        return len(self.imgs_50x)

def random_crop(image, crop_height, crop_width):
    width, height = image.size
    if (crop_width <= width) and (crop_height <= height):
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)

    return x, y

if __name__ == "__main__":
    root = 'General-100'
    out1 = 'data/train/100x'
    out2 = 'data/train/50x'
    value = 100
    files = natsorted(os.listdir(root))
    for file in files:
        img = Image.open(os.path.join(root, file))

        for i in range(10):
            x, y = random_crop(img, value, value)
            crop = img.crop((x, y, x+value, y+value))

            save_name = "{}_".format(i)+file
            crop.save('{}'.format(os.path.join(out1, save_name)))

            # crop.save('{}'.format(os.path.join(out1, file)))
            crop.resize((50, 50)).save('{}'.format(os.path.join(out2, save_name)))