import os
import argparse
import torchvision
import numpy as np
from collections import Counter


parser = argparse.ArgumentParser(description='Butterfly dataset symlink creation',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', type=str, default=os.environ.get("SCRATCH"),
                    help='Directory with butterfly dataset and indices')
args = parser.parse_args()


# divide dataset in test and training/validation data and save idx of that
dataset = torchvision.datasets.ImageFolder(root = os.path.join(args.data_dir,
                                "Schmetterlinge"))
targets = dataset.targets

# get indices of all classes with >= n images
n = 50
class_counts = Counter(dataset.targets)
larger_200 = {i for i in class_counts if class_counts[i] >= n}
idx = [i  for i in range(len(dataset)) if dataset.imgs[i][1] in larger_200]

#create a new root folder with symbolic links to the classes that are kept

#get classes in the subset of the data
targets = [dataset.targets[i] for i in idx]

#classes
classes = list(np.unique([dataset.classes[i] for i in targets]))

for dirs in classes:
    src = os.path.join(args.data_dir, "Schmetterlinge", dirs)
    dst = os.path.join(args.data_dir, "Schmetterlinge_sym", dirs)
    os.symlink(src, dst)
