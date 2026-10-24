#!/usr/bin/env python3
import argparse
import os
import random



def _new_run_config(args, fresh_args):
    """Build a clean config per run; legacy launchers mutated the global cfg."""
    from utils.config import _C as base_cfg

    run_cfg = base_cfg.clone()
    data_file = os.path.join("./configs/data", args.data + ".yaml")
    model_file = os.path.join("./configs/model", args.model + ".yaml")
    run_cfg.defrost()
    run_cfg.merge_from_file(data_file)
    run_cfg.merge_from_file(model_file)
    run_cfg.merge_from_list(args.opts)

    overrides = dict(fresh_args)
    if args.test_checkpoint:
        overrides["model_dir"] = args.test_checkpoint
    if overrides:
        override_list = []
        for key, value in overrides.items():
            override_list.extend([key, value])
        run_cfg.merge_from_list(override_list)

    return run_cfg


def main(args, fresh_args=None):
    import numpy as np
    import torch
    from utils.logger import setup_logger_tiaoyou
    from trainer_Sage import Trainer

    fresh_args = fresh_args or {}
    run_cfg = _new_run_config(args, fresh_args)

    if run_cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join("_" + item for item in args.opts)
        run_cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        run_cfg.output_dir = os.path.join("./output", run_cfg.output_dir)

    add_str = "".join("_{}_{}".format(key, value) for key, value in fresh_args.items())
    print("Output directory: {}".format(run_cfg.output_dir))
    setup_logger_tiaoyou(run_cfg.output_dir, add_str)

    if run_cfg.seed is not None:
        seed = run_cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if run_cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print("** Current Config **")
    print(run_cfg)
    print("************")

    trainer = Trainer(run_cfg)
    if run_cfg.model_dir is not None:
        trainer.load_model(run_cfg.model_dir)

    if run_cfg.zero_shot:
        return trainer.test()

    if run_cfg.test_train:
        if run_cfg.model_dir is None:
            marker = "_test_train_True"
            if marker not in run_cfg.output_dir:
                raise ValueError("test_train=True needs MODEL_DIR or an output directory containing {}".format(marker))
            run_cfg.model_dir = run_cfg.output_dir.split(marker, 1)[0]
            print("Model directory: {}".format(run_cfg.model_dir))
        trainer.load_model(run_cfg.model_dir)
        return trainer.test("train")

    if run_cfg.test_only:
        if run_cfg.model_dir is None:
            raise ValueError("test_only=True requires --test-checkpoint or MODEL_DIR in config/overrides")
        return trainer.test()

    return trainer.train()



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", "-d", default="", help="data config name under configs/data")
    parser.add_argument("--model", "-m", default="", help="model config name under configs/model")
    parser.add_argument("--test-checkpoint", default=None,
                        help="checkpoint directory for test_only mode; replaces legacy hard-coded paths")
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="config overrides, e.g. TEST_ONLY True SEED 0")
    return parser.parse_args()


def cli():
    args = parse_args()
    if not args.data or not args.model:
        raise SystemExit("Both -d/--data and -m/--model are required. See --help for examples.")
    main(args, {})


if __name__ == "__main__":
    cli()
