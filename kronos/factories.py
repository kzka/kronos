"""Module-wide factories.
"""

import abc

from typing import Dict, Callable

import os.path as osp
import torch

from kronos import datasets, models, trainers, evaluators
from kronos.datasets import batch_samplers, frame_samplers
from kronos.config import CONFIG
from kronos.utils import file_utils


class Factory(abc.ABC):
    """Base factory abstraction.

    All subclasses must implement `from_config` method.
    """

    PRODUCTS: Dict[str, Callable] = {}

    @classmethod
    def create(cls, name, *args, **kwargs):
        """Create a factory object by its name and args.
        """
        if name not in cls.PRODUCTS:
            raise KeyError("{} is not supported.".format(name))

        return cls.PRODUCTS[name](*args, **kwargs)

    @abc.abstractclassmethod
    def from_config(cls, config):
        """Create a factory object from a config.
        """
        pass


class FrameSamplerFactory(Factory):
    """Factory to create frame samplers.
    """

    PRODUCTS = {
        "all": frame_samplers.AllSampler,
        "stride": frame_samplers.StridedSampler,
        "offset_uniform": frame_samplers.OffsetUniformSampler,
        "window": frame_samplers.WindowSampler,
        "phase": frame_samplers.PhaseSampler,
    }

    @classmethod
    def from_config(cls, config, downstream):
        kwargs = {
            "num_frames": config.SAMPLING.NUM_FRAMES_PER_SEQUENCE,
            "num_ctx_frames": config.SAMPLING.NUM_CONTEXT_FRAMES,
            "ctx_stride": config.SAMPLING.CONTEXT_STRIDE,
            "pattern": config.SAMPLING.IMAGE_EXT,
            "seed": config.SEED,
        }
        if downstream:
            kwargs["stride"] = config.SAMPLING.STRIDE_ALL_SAMPLER
            kwargs.pop("num_frames")
            return cls.create("all", **kwargs)
        if config.SAMPLING.STRATEGY == "stride":
            kwargs["stride"] = config.SAMPLING.STRIDE_STRIDED_SAMPLER
            kwargs["offset"] = config.SAMPLING.USE_OFFSET_STRIDED_SAMPLER
        elif config.SAMPLING.STRATEGY == "offset_uniform":
            kwargs[
                "offset"
            ] = config.SAMPLING.RANDOM_OFFSET_OFFSET_UNIFORM_SAMPLER

        return cls.create(config.SAMPLING.STRATEGY, **kwargs)


class DatasetFactory(Factory):
    """Base dataset factory.
    """

    PATHS = {
        "penn_action": CONFIG.DIRS.PENN_ACTION_DIR,
        "mime": CONFIG.DIRS.MIME_DIR,
        "chamfer": CONFIG.DIRS.CHAMFER_DIR,
        "embodied_flip_book": CONFIG.DIRS.EMBODIED_FLIP_BOOK_DIR,
        "embodied_glasses": CONFIG.DIRS.EMBODIED_GLASSES_DIR,
        "embodied_cup_in_plate": CONFIG.DIRS.EMBODIED_CUP_IN_PLATE_DIR,
        "embodied_pens_in_cup": CONFIG.DIRS.EMBODIED_PENS_IN_CUP_DIR,
        "demo": CONFIG.DIRS.DEMO_DIR,
        "magical": CONFIG.DIRS.MAGICAL_DIR,
    }

    PRODUCTS = {
        "tcc": datasets.VideoDataset,
        "stcc": datasets.VideoDataset,
        "sal": datasets.VideoDataset,
        "svtcn": datasets.VideoDataset,
    }

    @classmethod
    def from_config(
        cls, config, downstream, split, debug=False, labeled=None,
    ):
        """Create a dataset directly from a config.
        """
        assert split in [
            "train",
            "valid",
        ], "{} is not a supported split.".format(split)

        print(f"Dataset: {cls.PATHS[config.DATASET]}")
        dataset_path = osp.join(cls.PATHS[config.DATASET], split)

        if split == "train":
            augment = config.AUGMENTATION.TRAIN
        else:
            augment = config.AUGMENTATION.EVAL

        # restrict action classes if they have been
        # provided.
        # else, load all from the data directory.
        if config.ACTION_CLASS:
            action_classes = config.ACTION_CLASS
        else:
            action_classes = file_utils.get_subdirs(
                dataset_path, basename=True, nonempty=True
            )

        # figure out image size
        image_size = config.IMAGE_SIZE
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        image_size = tuple(image_size)

        if debug:
            # The minimum obligatory data augmentation we want to keep is resize
            # and mean-var normalization.
            # augment = ["global_resize", "normalize"]
            augment = ["global_resize"]

        # print('Action classes: \n')
        # for action_class in action_classes:
        #     print(f'\t{action_class}')
        # print('\n')

        if downstream:
            dataset = {}
            for action_class in action_classes:
                frame_sampler = FrameSamplerFactory.from_config(
                    config, downstream=True
                )
                single_class_dataset = cls.create(
                    config.TRAINING_ALGO,
                    dataset_path,
                    frame_sampler,
                    labeled=labeled,
                    seed=config.SEED,
                    augment_params=augment,
                    image_size=image_size,
                )
                single_class_dataset.restrict_subdirs(action_class)
                dataset[action_class] = single_class_dataset
        else:
            frame_sampler = FrameSamplerFactory.from_config(
                config, downstream=False
            )
            dataset = cls.create(
                config.TRAINING_ALGO,
                dataset_path,
                frame_sampler,
                labeled=labeled,
                seed=config.SEED,
                augment_params=augment,
                image_size=image_size,
            )
            dataset.restrict_subdirs(action_classes)

        return dataset


class PreTrainingDatasetFactory(DatasetFactory):
    """Factory to create pre-training datasets."""

    PRODUCTS = DatasetFactory.PRODUCTS

    @classmethod
    def from_config(cls, config, split, debug, labeled):
        return super().from_config(config, False, split, debug, labeled,)


class DownstreamDatasetFactory(DatasetFactory):
    """Factory to create downstream task datasets.
    """

    PRODUCTS = DatasetFactory.PRODUCTS

    @classmethod
    def from_config(cls, config, split, debug, labeled):
        return super().from_config(config, True, split, debug, labeled,)


class BatchSamplerFactory(Factory):
    """Factory to create different batch samplers.
    """

    PRODUCTS = {
        "same_action": batch_samplers.SameClassBatchSampler,
        "same_action_downstream": batch_samplers.SameClassBatchSamplerDownstream,
        "random": batch_samplers.RandomBatchSampler,
    }

    @classmethod
    def from_config(cls, config, dataset, downstream, debug, labeled):
        if downstream:
            return cls.create(
                config.SAMPLING.DOWNSTREAM_BATCH_SAMPLER,
                dataset.dir_tree,
                sequential=True if debug else False,
                labeled=labeled,
            )
        return cls.create(
            config.SAMPLING.PRETRAIN_BATCH_SAMPLER,
            dataset.dir_tree,
            config.BATCH_SIZE,
            sequential=True if debug else False,
            labeled=labeled,
        )


class TrainerFactory(Factory):
    """Factory to create trainer classes.
    """

    PRODUCTS = {
        "tcc": trainers.TCCTrainer,
        "stcc": trainers.SubTCCTrainer,
        "sal": trainers.SALTrainer,
        "svtcn": trainers.SVTCNTrainer,
    }

    @classmethod
    def from_config(cls, config, model, optimizer, device):
        return cls.create(
            config.TRAINING_ALGO,
            model,
            optimizer,
            device,
            config.FP16_OPT,
            config,
        )


class EvaluatorFactory(Factory):
    """Factory to create downstream task evaluators.
    """

    PRODUCTS = {
        "kendalls_tau": evaluators.KendallsTau,
        "phase_alignment_topk": evaluators.PhaseAlignmentTopK,
        "cycle_consistency": evaluators.CycleConsistency,
        "nn_visualizer": evaluators.NearestNeighbourVisualizer,
        "reward_visualizer": evaluators.RewardVisualizer,
        # "linear_probe": evaluators.LinearProbe,
    }

    @classmethod
    def from_config(cls, config):
        evaluators = {}
        for eval_name in config.EVAL.DOWNSTREAM_TASK_EVALUATORS:
            kwargs = {"distance": config.EVAL.DISTANCE}

            if eval_name == "kendalls_tau":
                kwargs["stride"] = config.EVAL.KENDALLS_TAU.STRIDE
            elif eval_name == "cycle_consistency":
                kwargs["mode"] = config.EVAL.CYCLE_CONSISTENCY.MODE
                kwargs["stride"] = config.EVAL.CYCLE_CONSISTENCY.STRIDE
            elif eval_name == "phase_alignment":
                kwargs["topk"] = config.EVAL.PHASE_ALIGNMENT_TOPK.TOPK
                kwargs["num_ctx_frames"] = config.SAMPLING.NUM_CONTEXT_FRAMES
            elif eval_name == "nn_visualizer":
                kwargs["num_ctx_frames"] = config.SAMPLING.NUM_CONTEXT_FRAMES
                kwargs[
                    "num_frames"
                ] = config.EVAL.NEAREST_NEIGHBOUR_VISUALIZER.NUM_FRAMES
            elif eval_name == "linear_probe":
                kwargs.pop("distance")
                kwargs[
                    "num_frames_per_seq"
                ] = config.EVAL.LINEAR_PROBE.NUM_FRAMES_PER_SEQ
            elif eval_name == 'reward_visualizer':
                kwargs['l2_normalize'] = (
                    config.EVAL.REWARD_VISUALIZER.L2_NORMALIZE)

            evaluators[eval_name] = cls.create(eval_name, **kwargs)

        return evaluators


class ModelFactory(Factory):
    """Factory to create models.
    """

    PRODUCTS = {
        "tcc": models.TCCNet,
        "stcc": models.TCCNet,
        "sal": models.SALNet,
        "svtcn": models.TCNNet,
        "mvtcn": models.TCNNet,
    }

    @classmethod
    def from_config(cls, config):
        kwargs = {}
        kwargs["num_ctx_frames"] = config.SAMPLING.NUM_CONTEXT_FRAMES
        kwargs["model_config"] = config.MODEL
        return cls.create(config.TRAINING_ALGO, **kwargs)


class OptimizerFactory(Factory):
    """Factory to create optimizers.
    """

    PRODUCTS = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }

    @classmethod
    def from_config(cls, config, model):
        param_groups = []
        for model_name, param_dict in model.param_groups().items():
            params = []
            if "reg" in param_dict:
                params.append(
                    {
                        "params": param_dict["reg"],
                        "weight_decay": getattr(
                            config.OPTIMIZER, f"{model_name.upper()}_L2_REG"
                        ),
                        "lr": getattr(
                            config.OPTIMIZER, f"{model_name.upper()}_LR"
                        ),
                    }
                )
            if "free" in param_dict:
                params.append(
                    {
                        "params": param_dict["free"],
                        "lr": getattr(
                            config.OPTIMIZER, f"{model_name.upper()}_LR"
                        ),
                    }
                )
            param_groups.extend(params)
        if config.OPTIMIZER.TYPE == "sgd":
            kwargs = {"momentum": config.OPTIMIZER.MOMENTUM}
        else:
            kwargs = {}

        return cls.create(config.OPTIMIZER.TYPE, param_groups, **kwargs,)
