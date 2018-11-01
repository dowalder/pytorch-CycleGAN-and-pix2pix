#!/usr/bin/env python3

import argparse
import pathlib
import PIL.Image
import time

import torch
import torchvision
import cv2
import numpy as np

import models.networks


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def backward_transform(image):
    image = tensor2im(image)
    image = cv2.resize(image, (160, 120))
    return PIL.Image.fromarray(image)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", required=True)
    parser.add_argument("--src_dir", "-s", required=True)
    parser.add_argument("--tgt_dir", "-t", required=True)
    parser.add_argument("--recursive", "-r", action="store_true", help="recursively iterate through directories")

    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    if args.gpu == -1:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)
    net = models.networks.UnetGenerator(3, 3, 8)
    net.to(device)

    state_dict = torch.load(args.weights, map_location=device)
    for key in list(state_dict.keys()):
        if "num_batches_tracked" in key:
            del state_dict[key]
    net.load_state_dict(state_dict)

    forward_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    src_root = pathlib.Path(args.src_dir)
    tgt_root = pathlib.Path(args.tgt_dir)

    def transform_folder(src_dir: pathlib.Path, tgt_dir: pathlib.Path):
        for img_path in src_dir.iterdir():
            if img_path.is_dir() and args.recursive:
                tgt = tgt_dir / img_path.name
                tgt.mkdir(exist_ok=True)
                transform_folder(img_path, tgt)
            if img_path.suffix not in [".jpg", ".png"]:
                continue
            img = PIL.Image.open(img_path.as_posix()).convert("RGB")
            img = forward_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                img = net(img).cpu().squeeze()
            img = backward_transform(img)

            tgt_path = tgt_dir / img_path.name
            img.save(tgt_path.as_posix())
        print("Finished {}".format(src_dir))

    transform_folder(src_root, tgt_root)


if __name__ == "__main__":
    main()

