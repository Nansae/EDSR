import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms as T
from PIL import Image
from model import EDSR
from tqdm import tqdm
from data import Patches

def train(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    mean_value = 0.5

    # root = 'data/train'
    train_dataset = Patches(root=args.path, phase='train')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize)

    model = EDSR(num_layers=args.layers, feature_size=args.featuresize)
    model.to(device).train()

    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, args.epochs+1):
        pbar = tqdm(trainloader)
        for data in pbar:
            data50, data100 = data
            data50, data100 = data50.to(device), data100.to(device)

            optimizer.zero_grad()
            output, _ = model(data50)
            loss = F.l1_loss(output, data100-mean_value)
            loss.backward()
            optimizer.step()
            pbar.set_description("epoch: %d train_loss: %.4f" % (epoch, loss.item()))

        if epoch%50 == 0:
            torch.save(model.state_dict(), args.savedir + '/edsr_step_{}.pth'.format(epoch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='data/train')
    parser.add_argument("--scale", default=2, type=int)
    parser.add_argument("--layers", default=32, type=int)
    parser.add_argument("--featuresize", default=256, type=int)
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--savedir", default='saved_models')
    parser.add_argument("--epochs", default=50, type=int)
    args = parser.parse_args()

    train(args)
