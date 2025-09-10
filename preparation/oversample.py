import os
import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from accelerate.utils import set_seed


parser = argparse.ArgumentParser(description='Butterfly dataset minority oversampling',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', type=str, default=os.environ.get("SCRATCH"),
                    help='Directory with butterfly dataset and indices')
args = parser.parse_args()


# Keep reproducible (Accelerate sets all seeds)
set_seed(42)


# Transformations of training data to avoid overfitting
train_transform = v2.Compose([
    v2.RandomResizedCrop(size = 224, scale = (0.5, 1), ratio = (0.8, 1.2)), #randomly crop image between to up to half the image, randomly change the aspect ratio, and crop to 224 x 224 px
    v2.RandomHorizontalFlip(p = 0.3),
    v2.RandomVerticalFlip(p = 0.3),
    v2.RandomPerspective(distortion_scale = 0.2, p = 0.4), #Randomly distort the perspective of the image
    v2.RandomRotation(degrees = 50, expand = False), #Random Rotation from -50 to +50 dgrees
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #normalize based on ImageNet
])

# Transformations of validation data (only resizes and crops images and normalizes)
val_transform = v2.Compose([
    v2.Resize(size = 224),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Connect dataset
data_dir = os.path.join(args.data_dir, "Schmetterlinge_sym")
dataset = ImageFolder(root = data_dir)
# Load both training and validation but not test subset (indices) of the data
idx_train_val = np.load(os.path.join(args.data_dir, "preparation",
                                     "train_idx.npy"))


# Class names
classes = dataset.classes
# Number of classes
n_classes = len(classes)  # -> Used for final classification layer
# Number of examples per class -> used for train-test-split
targets = [dataset.targets[i] for i in idx_train_val]


# Split indices into training and validation sets
train_idx, val_idx = train_test_split(
    idx_train_val,
    test_size = 0.2,
    shuffle = True,
    stratify = targets  # Ensure similar number of examples per class
)


# Load training data
train_data = ImageFolder(root = data_dir, transform = train_transform)
train_data = Subset(train_data, train_idx)


# Oversampling of minority classes

sample_startTime = time.time()

# Count the number of images per class (replace with targets?)
labels = [label for image, label in train_data]
counts = Counter(labels)

# Get the weight for each class (inverse value of the number of images in each class)
class_weights_dict = dict(zip(counts.keys(),
            [1/weights for weights in list(counts.values())]))

# Assign the weights to each sample in the unbalanced dataset
sample_weights = [class_weights_dict.get(i) for i in labels]

# Oversample minority classes with a weighted random sampler
oversampler = torch.utils.data.WeightedRandomSampler(
                    weights = sample_weights,
                    num_samples = len(train_idx),
                    replacement = True)

sample_endTime = time.time()

print("Time used to sample training data: {:.2f}s".format(sample_endTime-sample_startTime))


# Save indices in preparation folder
indices = list(iter(oversampler))
torch.save(indices, os.path.join(args.data_dir, "preparation",
                                 "oversampled_idx.pth"))
