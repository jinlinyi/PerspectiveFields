#!/usr/bin/env python3
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.logger import setup_logger

# required so that .register() calls are executed in module scope
import perspective2d.modeling  # noqa
from perspective2d.config import get_perspective2d_cfg_defaults
from perspective2d.data import PerspectiveMapper
from perspective2d.evaluation import PerspectiveEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Args:
            cfg (CfgNode)
            dataset_name (str): name of dataset to test on

        Returns:
            PerspectiveEvaluator: class used to evaluate perspective field predictions
        """
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["gravity", "latitude", "perspective"]:
            return PerspectiveEvaluator(
                dataset_name, cfg, True, output_dir=cfg.OUTPUT_DIR
            )
        else:
            raise ValueError("The evaluator type is wrong")

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Args:
            cfg (CfgNode)
            dataset_name (str): name of dataset to test on

        Returns:
            torch.utils.data.DataLoader: a torch DataLoader, that loads the given detection dataset,
                                         with test-time transformation and batching.
        """
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=PerspectiveMapper(cfg, False, dataset_names=(dataset_name,)),
        )

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Args:
            cfg (CfgNode)

        Returns:
            torch.utils.data.DataLoader: a dataloader. Each output from it is a list[mapped_element]
                                         of length total_batch_size / num_workers,
                                         where mapped_element is produced by the mapper
        """
        dataset_names = cfg.DATASETS.TRAIN
        return build_detection_train_loader(
            cfg, mapper=PerspectiveMapper(cfg, True, dataset_names=dataset_names)
        )

    @classmethod
    def test(cls, cfg, model):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
        Returns:
            dict: a dict of result metrics
        """
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = cls.build_test_loader(cfg, dataset_name)
            evaluator = cls.build_evaluator(cfg, dataset_name)
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
        return results


# TODO: set args type
def setup(args):
    """setup model configurations

    Args:
        args (_type_): command-line arguments

    Returns:
        CfgNode: model configurations
    """
    cfg = get_cfg()
    get_perspective2d_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "meshrcnn" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="perspective"
    )
    return cfg


def main(args):
    cfg = setup(args)
    print(cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        print(res)
        return res

    trainer = Trainer(cfg)

    print("# of layers require gradient:")
    for c in trainer.checkpointer.model.named_children():
        grad = np.array(
            [
                param.requires_grad
                for param in getattr(trainer.checkpointer.model, c[0]).parameters()
            ]
        )
        print(c[0], grad.sum())
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


parser = default_argument_parser()

if __name__ == "__main__":
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
