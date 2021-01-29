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

import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.utils import load_state_dict_from_url
from kronos.models.layers import conv2d


def seed_rng(seed):
    """Seeds python, numpy, and torch RNGs."""
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


def get_factories(config, device, debug=None):
    """Feed config to factories and return objects.

    Args:
        config (dict): The module-wide config dict.
        device (torch.device): The computing device.
        debug (dict): A dictionary containing debugging-related
            params.
    """
    dloaders = get_dataloaders(config, debug)
    model = factories.ModelFactory.from_config(config)
    # optimizer = factories.OptimizerFactory.from_config(config, model)
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
