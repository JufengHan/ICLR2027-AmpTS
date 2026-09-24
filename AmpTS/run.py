import argparse
import os
import random

import numpy as np
import torch


def set_random_seed(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_setting(args, iteration):
    return (
        f"{args.task_name}_{args.data}_AmpTS"
        f"_sl{args.seq_len}_pl{args.pred_len}"
        f"_T{args.restoration_steps}"
        f"_hist{args.history_mode}"
        f"_ck{args.conv_kernel}"
        f"_seed{args.random_seed}"
        f"_{args.des}_{iteration}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AmpTS")

    parser.add_argument("--task_name", type=str, default="long_term_forecast", choices=["long_term_forecast"])
    parser.add_argument("--is_training", type=int, required=True, choices=[0, 1])
    parser.add_argument("--model_id", type=str, default="AmpTS")
    parser.add_argument("--model", type=str, default="AmpTS", choices=["AmpTS"])

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./dataset/")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--inverse", action="store_true", default=False)

    parser.add_argument("--restoration_steps", type=int, default=8)
    parser.add_argument(
        "--history_mode",
        type=str,
        default="conv_linear_decomp",
        choices=["conv_linear", "conv_decomp", "conv_linear_decomp"],
    )
    parser.add_argument("--conv_kernel", type=int, default=3)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--des", type=str, default="run")
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--use_amp", action="store_true", default=False)

    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    args = parser.parse_args()
    set_random_seed(args.random_seed)

    if args.restoration_steps < 2:
        raise ValueError("restoration_steps must be at least 2")
    if args.conv_kernel <= 0 or args.conv_kernel % 2 == 0:
        raise ValueError("conv_kernel must be a positive odd integer")

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device("cuda:{}".format(args.gpu))
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device_id) for device_id in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    for iteration in range(args.itr if args.is_training else 1):
        setting = build_setting(args, iteration)
        print(
            "{} | data={} | seq={} | pred={} | T={} | history={} | seed={}".format(
                "Train" if args.is_training else "Test",
                args.data,
                args.seq_len,
                args.pred_len,
                args.restoration_steps,
                args.history_mode,
                args.random_seed,
            )
        )
        exp = Exp_Long_Term_Forecast(args)
        if args.is_training:
            exp.train(setting)
            exp.test(setting)
        else:
            exp.test(setting, test=1)

        if args.use_gpu:
            if args.gpu_type == "mps" and hasattr(torch.backends, "mps"):
                torch.backends.mps.empty_cache()
            elif args.gpu_type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
