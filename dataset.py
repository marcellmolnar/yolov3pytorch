import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import warnings

from augmentations import AUGMENTATION_TRANSFORMS, DEFAULT_TRANSFORMS


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ListDataset(Dataset):
    def __init__(self, listPath, imgSize, transform, validation=False):
        with open(listPath, "r") as file:
            self.imgFiles = file.readlines()

        self.labelFiles = []
        for path in self.imgFiles:
            labelDir = "labels".join(os.path.dirname(path).rsplit("images", 1))
            labelFile = os.path.splitext(os.path.join(labelDir, os.path.basename(path)))[0] + '.txt'
            self.labelFiles.append(labelFile)

        self.imgSize = imgSize
        self.transform = transform

        self.ds = len(self.imgFiles)
        if not validation:
            self.ds = int(self.ds/1)

    def __len__(self):
        return self.ds

    def __getitem__(self, index):
        index = index % self.ds
        # image
        try:
            imgPath = self.imgFiles[index % len(self.imgFiles)].rstrip()
            img = np.array(Image.open(imgPath).convert('RGB'), dtype=np.uint8)
        except Exception:
            #print(f"Could not read image '{imgPath}'.")
            return

        # label
        try:
            labelPath = self.labelFiles[index % len(self.labelFiles)].rstrip()
            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(labelPath).reshape(-1, 5)
        except Exception:
            #print(f"Could not read label '{labelPath}'.")
            return

        # transform
        try:
            img, bbTargets = self.transform((img, boxes))
        except Exception:
            print("Could not apply transformation")
            return

        return img, bbTargets

    def collateFn(self, batch):
        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs, bbTargets = list(zip(*batch))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.imgSize) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bbTargets):
            boxes[:, 0] = i
        bbTargets = torch.cat(bbTargets, 0)

        return imgs, bbTargets


def createDataLoader(imgPath, batchSize, imgSize, validation=False):
    """Creates a DataLoader for training.

    :param imgPath: Path to file containing all paths to training images.
    :type imgPath: str
    :param batchSize: Size of each image batch
    :type batchSize: int
    :param imgSize: Size of each image dimension for yolo
    :type imgSize: int
    :param validation: Whether to use image augmentation or not.
    :type validation: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        imgPath,
        imgSize,
        AUGMENTATION_TRANSFORMS if not validation else DEFAULT_TRANSFORMS,
        validation)
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collateFn)
    return dataloader
