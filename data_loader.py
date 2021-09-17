#!/usr/bin/env python3
from PIL import Image, ImageOps
import torch
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
# RESIZE THE IMAGE TO 128x128


# LOAD THE DATASET FROM THE DESIRED PATH


def loadDataset(trainPath, testPath):

    # Load all the images
    transformation = transforms.Compose([
        # Randomly augment the image data
        # Random horizontal flip
        transforms.RandomHorizontalFlip(0.5),
        # Random vertical flip
        transforms.RandomVerticalFlip(0.3),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images, transforming them
    trainDataset = torchvision.datasets.ImageFolder(
        root=trainPath,
        transform=transformation
    )

    testDataset = torchvision.datasets.ImageFolder(
        root=testPath,
        transform=transformation
    )

    # define a loader for the training data we can iterate through in 50-image batches
    trainLoader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    # define a loader for the testing data we can iterate through in 50-image batches
    testLoader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    return trainLoader, testLoader
