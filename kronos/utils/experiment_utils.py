import logging
import random
import time

import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from kronos import factories
from kronos.utils.file_utils import mkdir

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.utils import load_state_dict_from_url
from kronos.models.layers import conv2d
from ipdb import set_trace


def seed_rng(seed):
    """Seeds python, numpy, and torch RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_config(
    config_dict, config_path=None, override_list=None, serialize_path=None,
):
    """Updates a config dict.

    This function first updates the base dict values
    with values from a config file, then optionally
    overrides these values with a provided list of
    args, and finally optionally serializes the
    updated contents back to a yaml file.

    Args:
        config (dict): The base config dict.
        config_file (str): Path to an experimental run yaml
            file containing config options we'd like to override.
        override_list (list): Additional command line args we'd
            like to override in the config file.
        serialize_path (str): Path to the directory where to serialize
            the config file. Set to `None` to prevent serialization.
    """
    # override values given a yaml file
    if config_path is not None:
        logging.info(f"Updating default config with {config_path}.")
        config_dict.merge_from_file(config_path)

    # override values given a list
    if override_list is not None:
        logging.info("Also overriding with command line args.")
        config_dict.merge_from_list(override_list)

    # serialize the config file
    if serialize_path is not None:
        with open(serialize_path, "w") as f:
            config_dict.dump(stream=f, default_flow_style=False)

    # make config file immutable
    config_dict.freeze()


def init_experiment(
    logdir, config, config_file=None, override_list=None, transient=False,
):
    """Initializes a training experiment.

    Instantiates the compute device (CPU, GPU), serializes the config
    file to the log directory, optionally updates the config variables
    with values from a provided yaml file and seeds the RNGs.

    Args:
        logdir (str): Path to the log directory.
        config (dict): The module-wide config dict.
        config_file (str): Path to an experimental run yaml
            file containing config options we'd like to override.
        override_list (list): Additional command line args we'd
            like to override in the config file.
        transient (bool): Set to `True` to make a transient session,
            i.e. a session where the logging and config params are
            not saved to disk. This is useful for debugging sessions.
    """
    # init compute device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(
            "Using GPU {}.".format(torch.cuda.get_device_name(device))
        )
    else:
        logging.info("No GPU found. Falling back to CPU.")
        device = torch.device("cpu")

    # create logdir and update config dict
    if not transient:
        # if a yaml file already exists in the
        # log directory, it means we're resuming
        # from a previous run.
        # so we update the values of our config
        # file with the values in the yaml file.
        if osp.exists(osp.join(logdir, "config.yml")):
            logging.info(
                "Config yaml already exists in log dir. Resuming training."
            )
            update_config(
                config, osp.join(logdir, "config.yml"), override_list
            )
        # if no yaml file exists in the log directory,
        # it means we're starting a new experiment.
        # so we want to update our config file
        # but also serialize it to the log dir.
        else:
            # create the log directory if
            # it doesn't already exist
            mkdir(logdir)

            update_config(
                config,
                config_file,
                override_list,
                osp.join(logdir, "config.yml"),
            )
    else:
        logging.info("Transient model turned ON.")
        update_config(config, config_file, override_list)

    # seed rngs
    if config.SEED is not None:
        logging.info(f"Experiment seed: {config.SEED}.")
        seed_rng(config.SEED)
    else:
        logging.info("No RNG seed has been set for this experiment.")

    return config, device


def get_dataloaders(config, debug):
    """Wraps datasets in dataloaders.

    Args:
        config (dict): The module-wide config dict.
        debug (dict): A dictionary containing debugging-related
            params.
    """
    dloaders = {}
    pinned_memory = torch.cuda.is_available()
    num_workers = 4 if torch.cuda.is_available() else 0
    sample_sequential = False
    augment = True

    # We only need labeled data loaders if we are using an evaluator that
    # requires it. Currently, only phase_alignment requires labels.
    # TODO(kevin): Deal with this later.
    labeled = None

    # parse debug params
    if debug is not None:
        debug_default_params = {
            "sample_sequential": True,
            "augment": False,
            "num_workers": 0,
        }
        debug_default_params.update(debug)
        sample_sequential = debug_default_params["sample_sequential"]
        num_workers = debug_default_params["num_workers"]
        augment = debug_default_params["augment"]
        logging.info(f"Using {num_workers} workers.")

    batch_labeled = True if labeled == "both" else False

    # pretraining train dataset
    train_dataset = factories.PreTrainingDatasetFactory.from_config(
        config, "train", not augment, labeled)
    batch_sampler = factories.BatchSamplerFactory.from_config(
        config, train_dataset, False, sample_sequential, batch_labeled)
    dloaders["pretrain_train"] = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pinned_memory)

    # pretraining valid dataset
    valid_dataset = factories.PreTrainingDatasetFactory.from_config(
        config, "valid", True, labeled)
    batch_sampler = factories.BatchSamplerFactory.from_config(
        config, valid_dataset, False, sample_sequential, batch_labeled)
    dloaders["pretrain_valid"] = torch.utils.data.DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collate_fn,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pinned_memory)

    # downstream train dataset
    dloaders["downstream_train"] = {}
    train_downstream_datasets = factories.DownstreamDatasetFactory.from_config(
        config, "train", True, labeled=labeled)
    for (
        action_class,
        train_downstream_dataset,
    ) in train_downstream_datasets.items():
        batch_sampler = factories.BatchSamplerFactory.from_config(
            config,
            train_downstream_dataset,
            True,  # Turns off data augmentation.
            sample_sequential,
            batch_labeled)
        dloaders["downstream_train"][
            action_class
        ] = torch.utils.data.DataLoader(
            train_downstream_dataset,
            collate_fn=train_downstream_dataset.collate_fn,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pinned_memory)

    # downstream valid dataset
    dloaders["downstream_valid"] = {}
    valid_downstream_datasets = factories.DownstreamDatasetFactory.from_config(
        config, "valid", True, labeled=labeled)
    for (
        action_class,
        valid_downstream_dataset,
    ) in valid_downstream_datasets.items():
        batch_sampler = factories.BatchSamplerFactory.from_config(
            config,
            valid_downstream_dataset,
            True,  # Turns off data augmentation.
            sample_sequential,
            batch_labeled)
        dloaders["downstream_valid"][
            action_class
        ] = torch.utils.data.DataLoader(
            valid_downstream_dataset,
            collate_fn=valid_downstream_dataset.collate_fn,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pinned_memory)

    return dloaders


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 64)

    def forward(self, x):
        batch_size, t, c, h, w = x.shape
        x = x.view((batch_size * t, c, h, w))
        feats = self.model(x)
        return {
            "embs": feats.view((batch_size, t, -1)),
        }


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SiameseAENet(ResNet):
    """Siamese net with a reconstruction loss."""

    def __init__(
        self, block=BasicBlock, layers=[2, 2, 2, 2], **kwargs):
        super().__init__(block, layers, **kwargs)

        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            progress=False)
        self.load_state_dict(state_dict)

        self.head = nn.Linear(512, 64)

        # Upsampling.
        factor = 2
        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Compute embeddings.
        batch_size, t, c, h, w = x.shape
        x = x.view((batch_size * t, c, h, w))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Compute embeddings.
        feats = self.avgpool(x4)  # 60, 512, 1, 1
        flat_feats = torch.flatten(feats, 1)
        embs = self.head(flat_feats)
        embs = embs.view((batch_size, t, -1))

        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        recon = self.out_conv(x)

        return {
            "embs": embs,
            "reconstruction": recon,
        }


class SupervisedNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 128)
        self.head = nn.Linear(128, out_dim)

    def forward(self, x):
        return self.head(F.relu(self.model(x)))


def get_factories(config, device, debug=None):
    """Feed config to factories and return objects.

    Args:
        config (dict): The module-wide config dict.
        device (torch.device): The computing device.
        debug (dict): A dictionary containing debugging-related
            params.
    """
    dloaders = get_dataloaders(config, debug)
    # model = factories.ModelFactory.from_config(config)
    # optimizer = factories.OptimizerFactory.from_config(config, model)
    model = SiameseAENet()
    # model = Net()
    model.num_ctx_frames = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    trainer = factories.TrainerFactory.from_config(
        config, model, optimizer, device
    )
    evaluators = factories.EvaluatorFactory.from_config(config)
    return model, optimizer, dloaders, trainer, evaluators


class Stopwatch:
    """A simple timer for measuring elapsed time."""

    def __init__(self):
        self.reset()

    def elapsed(self):
        """Return the elapsed time since the stopwatch was reset.
        """
        return time.time() - self.time

    def done(self, target_interval):
        return self.elapsed() >= target_interval

    def reset(self):
        """Reset the stopwatch, i.e. start the timer.
        """
        self.time = time.time()
