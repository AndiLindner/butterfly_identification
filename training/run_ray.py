import os
import argparse
from functools import partial
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import OrderedDict
import torch
import torchvision
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
import timm
import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from accelerate.utils import set_seed


# Parse command line arguments
parser = argparse.ArgumentParser(description='Butterflies with DDP and Ray',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default="regnet_x_32gf",
                    help='Pretrained model')
parser.add_argument('--model-path', type=str, default="", required=False,
                    help='Finetuned model to be super-finetuned')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Input batch size for training')
parser.add_argument('--num-samples', type=int, default=16,
                    help='Number of samples to draw from the search config space')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--oversampling', type=int, default=0,
                    help='Whether using oversampling or class weights')
parser.add_argument('--data-dir', type=str, default=os.environ.get("SCRATCH"),
                    help='Directory with butterfly dataset and indices')
parser.add_argument('--results-dir', type=str, default=os.environ.get("SCRATCH"),
                    help='Directory to store trial results')
args = parser.parse_args()


def data_prep(args):

    # Transformations of training data to avoid overfitting
    train_transform = v2.Compose([
        v2.RandomResizedCrop(size = 224, scale = (0.5, 1), ratio = (0.8, 1.2)),
        v2.RandomHorizontalFlip(p = 0.3),
        v2.RandomVerticalFlip(p = 0.3),
        v2.RandomPerspective(distortion_scale = 0.2, p = 0.4),
        v2.RandomRotation(degrees = 50, expand = False),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformations of validation data (only resizes and crops images and normalizes)
    val_transform = v2.Compose([
        v2.Resize(size = 224),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    data_dir = os.path.join(args.data_dir, "Schmetterlinge_sym")
    dataset = ImageFolder(root = data_dir)

    # Load both training and validation but not test subset (indices) of the data
    train_val_idx = np.load(os.path.join(args.data_dir,
        "preparation", "train_val_idx.npy"))

    # Cannot be serialized by Ray -> use as constant
    #n_classes = len(dataset.classes)

    # Get or load targets -> required for class weights
    #targets = [dataset.targets[i] for i in train_val_idx]
    targets = np.load(os.path.join(args.data_dir, "preparation", "targets.npy"))

    # Split indices into training and validation sets or load indices
    #train_idx, val_idx = train_test_split(
    #    train_val_idx,
    #    test_size = 0.2,
    #    shuffle = True,
    #    stratify = targets  # Ensure similar number of examples per class
    #)

    train_idx = np.load(os.path.join(args.data_dir, "preparation", "train_idx.npy"))
    val_idx = np.load(os.path.join(args.data_dir, "preparation", "val_idx.npy"))

    # Cores per data loader
    num_workers = (int(os.environ['OMP_NUM_THREADS'])//
                     int(os.environ['SLURM_GPUS_PER_TASK']))

    # Load training data
    train_data = ImageFolder(root = data_dir, transform = train_transform)
    train_data = Subset(train_data, train_idx)

    if args.oversampling:
        # Load indices with oversampled minority classes
        indices = torch.load(os.path.join(args.data_dir,
            "preparation", "oversampled_idx.pth"))
        # Sampler to load indices of minority-oversampled dataset
        train_sampler = SubsetRandomSampler(indices)
        train_dl = DataLoader(train_data,
                batch_size = args.batch_size,
                sampler = train_sampler,
                pin_memory=True,
                num_workers = num_workers)

    else:
        # Data loader without weighted sampling
        train_dl = DataLoader(train_data,
                batch_size = args.batch_size,
                pin_memory=True,
                num_workers = num_workers)

    # Load valiation data with fewer trafos (and without reweighting)
    val_data = ImageFolder(root = data_dir,
            transform = val_transform)
    val_data = Subset(val_data, val_idx)
    val_dl = DataLoader(val_data,
            batch_size = args.batch_size,
            num_workers = num_workers,
            pin_memory = True)

    return train_dl, val_dl, targets


def worker(config):

    # Keep reproducible; for train-val-split as in previous trainings
    set_seed(42)

    # Use cmd line args via Ray config
    args = config["args"]

    train_dl, val_dl, targets = data_prep(args)  # targets for class weights
    n_classes = 163  # Use as constant due to Ray serialization issue

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the (pre-trained) model
    if (args.model.startswith("timm")):
        model = eval(f"timm.create_model('{args.model}', \
                pretrained=True, num_classes={n_classes})")
    else:
        model = eval(f"torchvision.models.{args.model}(weights='DEFAULT')")

    # Adapt the last layer to classes of the dataset for finetuning
    if ("resne" in args.model or "regne" in args.model):
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, n_classes))
    elif (args.model.startswith("densenet")):
        if (args.model.endswith("201")):
            model.classifier=nn.Linear(1920, n_classes)
        elif (args.model.endswith("169")):
            model.classifier=nn.Linear(1664, n_classes)
        elif (args.model.endswith("161")):
            model.classifier=nn.Linear(2208, n_classes)
        elif (args.model.endswith("121")):
            model.classifier=nn.Linear(1024, n_classes)
    elif (args.model.startswith("vgg")):
        num_ftrs = 512*7*7
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, n_classes),
            nn.LogSoftmax(dim=1))
    elif (args.model.startswith("efficientnet")):
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.1, inplace=True),
            nn.Linear(1280, n_classes),)
    elif (args.model.startswith("vit")):
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if (args.model.startswith("vit_b")):
            hidden_dim=768
        elif (args.model.startswith("vit_l")):
            hidden_dim=1024
        elif (args.model.startswith("vit_h")):
            # Need to load other than default weights for images of size 224
            model = torchvision.models.vit_h_14(weights='IMAGENET1K_SWAG_LINEAR_V1')
            hidden_dim=1280
        heads_layers["head"] = nn.Linear(hidden_dim, n_classes)
        model.heads = nn.Sequential(heads_layers)
    elif (args.model.startswith("swin")):
        if (args.model.endswith("_t")):
            embed_dim=96
        elif (args.model.endswith("_s")):
            embed_dim=96
        elif (args.model.endswith("_b")):
            embed_dim=128
        num_features = embed_dim * 2 ** 3
        model.head = nn.Linear(num_features, n_classes)
    elif (args.model.startswith("maxvit")):
        if ("tiny" in args.model):
            block_channels=[64, 128, 256, 512]
        elif ("small" in args.model or "base" in args.model):
            block_channels=[96, 192, 384, 768]
        elif ("large" in args.model):
            block_channels=[128, 256, 512, 1024]
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels[-1]),
            nn.Linear(block_channels[-1], block_channels[-1]),
            nn.Tanh(),
            nn.Linear(block_channels[-1], n_classes, bias=False)
        )
    elif (args.model.startswith("convnext")):
        if (args.model.endswith("tiny")):
            lastblock_input_channels = 768
        elif (args.model.endswith("small")):
            lastblock_input_channels = 768
        elif (args.model.endswith("base")):
            lastblock_input_channels = 1024
        elif (args.model.endswith("large")):
            lastblock_input_channels = 1536
        class LayerNorm2d(nn.LayerNorm):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                x = x.permute(0, 3, 1, 2)
                return x
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        model.classifier = nn.Sequential(
            norm_layer(lastblock_input_channels),
            nn.Flatten(1),
            nn.Linear(lastblock_input_channels, n_classes))
    elif (args.model.startswith("mobilenet")):
        if (args.model.endswith("small")):
            lastconv_output_channels=576
            last_channel=1024
        elif (args.model.endswith("large")):
            lastconv_output_channels=960
            last_channel=1280
        model.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, n_classes),
        )


    # Tuning optimizer stuff
    base_lr = config["learning_rate"]

    if (config["optimizer"] == "Adam"):
        opt = torch.optim.Adam(model.parameters(), lr = base_lr,
                               weight_decay=config["weight_decay"])
    elif (config["optimizer"] == "AdamW"):
        opt = torch.optim.AdamW(model.parameters(), lr = base_lr,
                               weight_decay=config["weight_decay"])
    elif (config["optimizer"] == "RMSprop"):
        opt = torch.optim.RMSprop(model.parameters(), lr = base_lr,
                               weight_decay=config["weight_decay"])
    elif (config["optimizer"] == "Adadelta"):
        opt = torch.optim.Adadelta(model.parameters(), lr = base_lr,
                               weight_decay=config["weight_decay"])
    elif (config["optimizer"] == "ASGD"):
        opt = torch.optim.ASGD(model.parameters(), lr = base_lr,
                               weight_decay=config["weight_decay"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr = base_lr,
                              momentum=0.9)

    if (config["scheduler"] == "StepLR"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer = opt,
                step_size = 2, # every 2 epochs
                gamma = 0.75 # factor by which the learning rate is scaled
                )
    elif (config["scheduler"] == "ReduceLROnPlateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = opt, # reduces learning when loss stops decreasing
                mode = "min",
                factor = 0.75, # factor by which the learning rate is scaled
                patience = 2) # number of epochs without improvement until learning rate decreases
    elif (config["scheduler"] == "CosineAnnealingLR"):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer = opt,
                T_max = args.epochs, # number of epochs for one cycle
                eta_min = 1e-8)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer = opt,
                factor=1,
                total_iters=50000)

    # Cross-Entropy Loss
    if not args.oversampling:
        # Estimation of class weights using original train+val dataset
        class_weights = compute_class_weight("balanced",
                classes=np.arange(n_classes),
                y=np.array(targets))
        # Put class weiths to device manually; loss moved at backward method
        class_weights = torch.from_numpy(class_weights).float().to(device)
        lossFN = nn.CrossEntropyLoss(weight=class_weights)
    else:
        lossFN = nn.CrossEntropyLoss()

    # Load model weights, a checkpoint or start fresh
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path), strict=False)
        #print(f"Using model saved at {args.model_path}.\n")

    model.to(device)

    # Calculate number of training and validation steps
    trainSteps = math.ceil(len(train_dl.dataset) / args.batch_size)
    valSteps = math.ceil(len(val_dl.dataset) / args.batch_size)


    # Training loop
    for _ in range(args.epochs):
        model.train()  # Put model in training mode
        totalTrainLoss = 0  # Set the losses to 0 for the current epoch
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0

        # Loop over the training set
        for(x,y) in train_dl:

            # Move data to GPU
            x = x.to(device)
            y = y.to(device)

            # Forward pass and training loss
            pred = model(x)   # Make a prediction for x
            loss = lossFN(pred, y)  # Calculate the loss

            # Backpropagation and update weights
            opt.zero_grad()
            loss.backward()  # Backpropagation
            opt.step()

            # Update total training loss and number of correct predictions
            trainCorrect += pred.argmax(1).eq(y).sum().item()
            totalTrainLoss += loss.item()

        # Evaluation
        with torch.no_grad():
            model.eval()  # Put model to evaluation mode

            # Loop over validation set
            for (x,y) in val_dl:

                # Move data to GPU
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                valCorrect += pred.argmax(1).eq(y).sum().item()
                totalValLoss += lossFN(pred, y).item()

        # Calculate average costs per batch
        avgTrainCost = totalTrainLoss / trainSteps
        avgValCost = totalValLoss / valSteps
        # Calculate training and validation accuracy
        trainCorrect = trainCorrect/len(train_dl.dataset)
        valCorrect = valCorrect/len(val_dl.dataset)

        # Update learning rate according to schedule
        scheduler.step(avgValCost)  # Metric ignored if not monitored by scheduler

        # Report some metrics to Ray Tune -> end of iteration
        #tune.report({"val_accuracy":valCorrect})
        tune.report(metrics={
            "train_cost":avgTrainCost, "train_accuracy":trainCorrect,
            "val_cost":avgValCost, "val_accuracy":valCorrect})


if __name__ =="__main__":

    # Connect to Ray cluster
    ray.init(
        #num_cpus=int(os.environ["SLURM_CPUS_PER_TASK"]),
        #num_gpus=int(os.environ["SLURM_GPUS_PER_TASK"]),
        log_to_driver=False,  # Gather output of workers
        ignore_reinit_error=True,
        include_dashboard=False
    )

    resources = ray.cluster_resources()

    print(f"\nCluster has {resources['CPU']} CPUs, "
    f"{resources['GPU'] if 'GPU' in resources else 0} GPUs, "
    f"{resources['memory'] * 1e-9} GBs memory, "
    f"{resources['object_store_memory'] * 1e-9} GBs object storage memory.\n")

    #MAX_CONCURRENT = (int(os.environ['SLURM_JOB_NUM_NODES']) *
    #                  int(os.environ['SLURM_GPUS_PER_TASK']))
    MAX_CONCURRENT = int(resources['GPU'])

    # Scheduler
    scheduler = ASHAScheduler(
    metric="val_accuracy",  # Metric to optimize
    mode="max",
    max_t=args.epochs,   # Maximum iterations
    grace_period=1, # Minimum iterations before stopping
    reduction_factor=2  # Factor for reducing trials each round
    )

    # Search space
    config = {
    #"batch_size": tune.grid_search([64, 128]),  # Batch size will be fine
    "learning_rate": tune.loguniform(1e-6, 1e-5),  # Low for super-finetuning
    "optimizer": tune.choice(["RMSprop", "Adam", "AdamW",
                              "ASGD"]),  # Adadelta worse
    #"momentum": tune.grid_search([0.6, 0.7, 0.8, 0.9]),  # Only SGD, RMSprop
    "weight_decay": tune.loguniform(1e-7, 1e-3),  # Low for super-finetuning
    "scheduler": tune.choice(["StepLR", "ReduceLROnPlateau",
                              "CosineAnnealingLR"]),
    "args": args,
    }

    # Search algorithm
    search_algorithm = HyperOptSearch(
        metric="val_accuracy",
        mode="max",
        n_initial_points=(args.num_samples//2)  # rand. samples before opt.
        )
    # One trial per GPU (overrides max_concurrent_trials)
    search_algorithm = ConcurrencyLimiter(search_algorithm,
            max_concurrent=MAX_CONCURRENT)

    # Resources; one GPU per searcher
    worker_with_resources = tune.with_resources(worker,
            #{"gpu":1})
            {"cpu": (int(os.environ['SLURM_CPUS_PER_TASK'])/int(os.environ['SLURM_GPUS_PER_TASK'])),
            "gpu": 1})

    # Tuner
    tuner = tune.Tuner(
    #partial(worker_with_resources, args), # Trainable function
    worker_with_resources, # Trainable
    param_space=config,  # Search space
    tune_config=tune.TuneConfig(
        #metric="val_accuracy",  # Either here or in scheduler
        #mode="max",
        scheduler=scheduler,  # Scheduler to use
        search_alg=search_algorithm, # Include search algorithm
        num_samples=args.num_samples,  # Number of samples to draw from the search space
        max_concurrent_trials=MAX_CONCURRENT,  # Maximum number of parallel trials
        ),
    run_config=tune.RunConfig(
        name=f"{args.model}_{os.environ['SLURM_JOB_ID']}",
        storage_path=args.results_dir,
        verbose=1,
        log_to_file=True,  # Capture also parallel worker output
        stop={"training_iteration": args.epochs},
        checkpoint_config=tune.CheckpointConfig(
            num_to_keep=1,
            checkpoint_frequency=0,
            checkpoint_at_end=False)
        )
    )

    # Perform hyperparameter scans and print results
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_accuracy", mode="max")
    print("\n")
    print("Configuration of best run:" , best_result.config)
    print("\n")
    print("Training cost of best run:", best_result.metrics["train_cost"])
    print("Training accuracy of best run:", best_result.metrics["train_accuracy"])
    print("Validation cost of best run:", best_result.metrics["val_cost"])
    print("Validation accuracy of best run:", best_result.metrics["val_accuracy"])
    print("\n")

    ray.shutdown()
