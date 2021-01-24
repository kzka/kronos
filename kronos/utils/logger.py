import numpy as np
import os.path as osp
import torch

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """A Tensorboard-based logger.
    """

    def __init__(self, log_dir, force_write=False):
        """Constructor.

        Args:
            log_dir (str): The directory in which to store
            force_write (bool): Whether to force write to an
                already existing log dir. Set to `True` if
                resuming training.
        """
        self._log_dir = log_dir

        # setup summary writer
        writer_dir = osp.join(self._log_dir, "train_logs")
        if osp.exists(writer_dir) and not force_write:
            raise ValueError(
                "You might be overwriting a directory that already "
                "has train_logs. Please provide a new experiment name "
                "or set --resume to True when launching train script."
            )
        self._writer = SummaryWriter(writer_dir)

    def close(self):
        self._writer.close()

    def log_scalar(self, scalar, global_step, prefix, name=""):
        """Log a scalar value.
        """
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.item()
        assert np.isscalar(scalar), "Not a scalar."
        msg = "/".join([prefix, name]) if name else prefix
        self._writer.add_scalar(msg, scalar, global_step)

    def log_dict_scalars(self, dict_scalars, global_step, prefix):
        """Log a dictionary of scalars.
        """
        assert isinstance(dict_scalars, dict)
        for name, scalar in dict_scalars.items():
            self.log_scalar(scalar, global_step, prefix, name)

    def log_learning_rate(self, optimizer, global_step):
        """Log the learning rate.

        Args:
            optimizer (torch.optim.Optimizer): An optimizer.
            global_step (int): The training iteration number.
        """
        assert isinstance(optimizer, torch.optim.Optimizer)
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        self.log_scalar(lr, global_step, "learning_rate")

    def log_loss(self, losses, global_step):
        """Log a loss.

        Args:
            losses (dict or float): Either a scalar or a dict
                of scalars.
            global_step (int): The training iteration number.
        """
        if not isinstance(losses, dict):
            losses = {"": losses}
        self.log_dict_scalars(losses, global_step, "loss")

    def log_metric(self, metrics, global_step, metric_name):
        """Log a metric.

        Args:
            metrics (dict): A dict containing the metric values.
                Possible keys can be 'scalar' or 'image'. Values
                can be scalars, lists or dicts.
            global_step (int): The training iteration number.
            metric_name (str): The name of the metric.
        """
        assert isinstance(metrics, dict)
        for split, metric in metrics.items():
            if "scalar" in metric:
                if isinstance(metric["scalar"], dict):
                    for k, v in metric["scalar"].items():
                        self.log_scalar(
                            v,
                            global_step,
                            split,
                            "{}_{}".format(metric_name, k),
                        )
                else:
                    self.log_scalar(
                        metric["scalar"], global_step, split, metric_name
                    )
            if "image" in metric:
                img = metric["image"][0]
                self._writer.add_image(
                    "{}/image/{}".format(split, metric_name),
                    img_tensor=img,
                    global_step=global_step,
                    dataformats="HWC",
                )
