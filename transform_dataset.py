#!/usr/bin/env python3

import argparse
import pathlib
import PIL.Image

import torch
import torchvision

import models.networks


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", "-w", required=True)
    parser.add_argument("--src_dir", "-s", required=True)
    parser.add_argument("--tgt_dir", "-t", required=True)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    if args.gpu == -1:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)
    net = models.networks.UnetGenerator(3, 3, 8)
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.to(device)

    forward_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])
    backward_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((120, 160))
    ])

    src_dir = pathlib.Path(args.src_dir)
    tgt_dir = pathlib.Path(args.tgt_dir)

    for img_path in src_dir.iterdir():
        if img_path.suffix not in [".jpg", ".png"]:
            continue
        img = PIL.Image.open(img_path.as_posix())
        img = forward_transform(img)
        img = net(img)
        img = backward_transform(img)

        tgt_path = tgt_dir / img_path.name
        img.save(tgt_path.as_posix())



