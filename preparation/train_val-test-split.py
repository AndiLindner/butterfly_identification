import os
import argparse
import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
from accelerate.utils import set_seed


parser = argparse.ArgumentParser(description='Train-val-test split of Butterfly dataset',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', type=str, default=os.environ.get("SCRATCH"),
                    help='Directory with butterfly dataset and indices')
args = parser.parse_args()

# Keep reproducible (Accelerate sets all seeds)
set_seed(42)


dataset = torchvision.datasets.ImageFolder(root =
        os.path.join(args.data_dir, "Schmetterlinge_sym"))
targets = dataset.targets

# Split into test and train/val by class
train_val_idx, test_idx = train_test_split(
    np.arange(len(targets)),
    test_size = 0.1,
    shuffle = True,
    stratify = targets
)

#save these indices
preparation_dir = os.path.join(args.data_dir, "preparation")
os.makedirs(preparation_dir, exist_ok=True)
np.save(os.path.join(preparation_dir, "train_val_idx.npy"),
        train_val_idx)
np.save(os.path.join(preparation_dir, "test_idx.npy"),
        test_idx)
