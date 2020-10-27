import os
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
from model import EDSR
from tqdm import tqdm
from data import Patches

def test(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = EDSR(num_layers=args.layers, feature_size=args.featuresize).to(device)
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).to(device)
    model.load_state_dict(torch.load(args.savedir))
    model.eval()

    test_dataset = Patches(root=args.path, phase='train')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize)

    for data in test_loader:
        data50, data100 = data
        data50 = data50.to(device)

        _, out_imgs = model(data50)

        res = []
        for idx in range(len(out_imgs)):
            images = []
            ## input
            data50_cpu = torch.squeeze(data50[idx]).cpu()
            images.append(transforms.ToPILImage()(data50_cpu).convert("RGB"))
            ## output
            output = torch.squeeze(out_imgs[idx]).cpu()
            images.append(transforms.ToPILImage()(output).convert("RGB"))
            # origin
            data100_cpu = torch.squeeze(data100[idx])
            images.append(transforms.ToPILImage()(data100_cpu).convert("RGB"))
            print(idx, len(images))
            res.append(images)

        fig = plt.figure(figsize=(7, 8))
        rows = args.batchsize
        cols = 3

        titles = ['input', 'output', 'origin']
        axes = []
        for r in range(rows):
            for c in range(cols):
                axes.append(fig.add_subplot(rows, cols, (r*cols+c) + 1))
                subplot_title = titles[c]
                axes[-1].set_title(subplot_title)
                plt.imshow(res[r][c])
                plt.savefig('res.png', dpi=300)

        # for i in range(cols*rows):
        #     axes.append(fig.add_subplot(rows, cols, i+1))
        #     subplot_title = titles[i]
        #     axes[-1].set_title(subplot_title)
        #     plt.imshow(images[i])
        #     plt.savefig('res.png', dpi=300)

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='data/test')
    parser.add_argument("--layers", default=32, type=int)
    parser.add_argument("--featuresize", default=256, type=int)
    parser.add_argument("--batchsize", default=3, type=int)
    parser.add_argument("--savedir", default='saved_models/edsr_step_50.pth')
    args = parser.parse_args()

    test(args)