import json
import os
import math
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from model import MNISTDiffusion, NUM_DIGIT_CLASSES
from utils import ExponentialMovingAverage


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )  # [0,1] to [-1,1]

    train_dataset = MNIST(
        root="./mnist_data", train=True, download=True, transform=preprocess
    )
    test_dataset = MNIST(
        root="./mnist_data", train=False, download=True, transform=preprocess
    )

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--ckpt", type=str, help="define checkpoint path", default="")
    parser.add_argument(
        "--n_samples",
        type=int,
        help="define sampling amounts after every epoch trained",
        default=36,
    )
    parser.add_argument(
        "--model_base_dim", type=int, help="base dim of Unet", default=64
    )
    parser.add_argument(
        "--timesteps", type=int, help="sampling steps of DDPM", default=1000
    )
    parser.add_argument(
        "--model_ema_steps", type=int, help="ema model evaluation interval", default=10
    )
    parser.add_argument(
        "--model_ema_decay", type=float, help="ema model decay", default=0.995
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        help="training log message printing frequence",
        default=10,
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="set to normal sampling method without clip x_0 which could yield unstable samples",
    )
    parser.add_argument("--cpu", action="store_true", help="cpu training")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--fixed_step_noises",
        action="store_true",
        help="use fixed noise sequence for each sampling step (fully deterministic generation)",
    )
    parser.add_argument(
        "--step_noises",
        type=str,
        default="",
        help="path to load step noise sequence from file (shape: timesteps, n_samples, C, H, W)",
    )
    parser.add_argument(
        "--gif_every_steps",
        type=int,
        default=50,
        help="when saving denoising GIF, sample a frame every this many steps (default 50 -> ~20 frames for 1000 steps)",
    )
    # 条件生成 (Classifier-Free Guidance)
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="train/sample with digit class labels for conditional generation and CFG",
    )
    parser.add_argument(
        "--class_dropout_prob",
        type=float,
        default=0.1,
        help="probability of dropping class label during training (for CFG); only used when --conditional",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="comma-separated digit labels for sampling (e.g. '3' or '0,1,2,3,4,5'); length must match n_samples when not single digit",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=0.0,
        help="Classifier-Free Guidance scale at sampling; 0=no CFG, try 2.0~3.0 when --conditional",
    )

    args = parser.parse_args()
    return args


def setup_logging(exp_dir):
    log_path = os.path.join(exp_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_denoising_gif(frames, path, nrow, duration=100, loop=0):
    """Save a list of (N,C,H,W) tensors in [0,1] as a GIF. Each frame is a grid of images."""
    pil_frames = []
    for grid_t in frames:
        grid = make_grid(grid_t, nrow=nrow, padding=2, pad_value=1.0)
        arr = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        if arr.shape[2] == 1:
            arr = arr.squeeze(-1)
            pil_frames.append(Image.fromarray(arr, mode="L"))
        else:
            pil_frames.append(Image.fromarray(arr))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
    )


def save_train_config(exp_dir, args, model):
    """Save training args, model config, and noise options to exp_dir/config.json."""
    config = {
        "exp_dir": os.path.abspath(exp_dir),
        "hyperparams": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "model_ema_steps": args.model_ema_steps,
            "model_ema_decay": args.model_ema_decay,
            "log_freq": args.log_freq,
            "no_clip": args.no_clip,
            "cpu": args.cpu,
        },
        "model": {
            "image_size": 28,
            "in_channels": 1,
            "base_dim": args.model_base_dim,
            "dim_mults": [2, 4],
            "timesteps": args.timesteps,
            "num_params": sum(p.numel() for p in model.parameters()),
            "conditional": getattr(args, "conditional", False),
            "num_classes": NUM_DIGIT_CLASSES if getattr(args, "conditional", False) else None,
            "class_dropout_prob": getattr(args, "class_dropout_prob", 0.1),
        },
        "sampling": {
            "n_samples": args.n_samples,
            "fixed_step_noises": args.fixed_step_noises,
            "step_noises_path": args.step_noises if args.step_noises else None,
            "cfg_scale": getattr(args, "cfg_scale", 0.0),
        },
        "resume": {
            "ckpt": args.ckpt if args.ckpt else None,
        },
    }
    path = os.path.join(exp_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def main(args):
    set_seed(args.seed)

    exp_dir = os.path.join(
        "results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logging(exp_dir)
    writer = SummaryWriter(log_dir=exp_dir)
    logger.info("Experiment directory: %s", os.path.abspath(exp_dir))
    logger.info("TensorBoard log_dir: %s", os.path.abspath(exp_dir))
    logger.info("Arguments: %s", vars(args))

    device = "cpu" if args.cpu else "cuda"
    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=28
    )
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
        num_classes=NUM_DIGIT_CLASSES if args.conditional else None,
        class_dropout_prob=args.class_dropout_prob if args.conditional else 0.1,
    ).to(device)
    save_train_config(exp_dir, args, model)
    logger.info("Saved train config to %s", os.path.join(exp_dir, "config.json"))

    # torchvision ema setting
    # https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        args.lr,
        total_steps=args.epochs * len(train_dataloader),
        pct_start=0.25,
        anneal_strategy="cos",
    )
    loss_fn = nn.MSELoss(reduction="mean")

    # load checkpoint (须与训练时一致使用 --conditional，否则结构不匹配)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded checkpoint from %s", args.ckpt)

    # Initial noise for sampling: create on first sampling (reused within run)
    initial_sampling_noise = None

    # Step noises for deterministic sampling: load from file if given, else create on first sampling
    step_noises = None
    if args.step_noises:
        step_noises = torch.load(args.step_noises, map_location=device)
        logger.info("Loaded step noise sequence from %s", args.step_noises)
        if step_noises.shape[1] != args.n_samples:
            raise ValueError(
                f"step_noises n_samples ({step_noises.shape[1]}) does not match "
                f"--n_samples ({args.n_samples})"
            )

    global_steps = 0
    try:
        for i in range(args.epochs):
            model.train()
            for j, (image, target) in enumerate(train_dataloader):
                noise = torch.randn_like(image).to(device)
                image = image.to(device)
                target = target.to(device) if args.conditional else None
                pred = model(image, noise, y=target)
                loss = loss_fn(pred, noise)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if global_steps % args.model_ema_steps == 0:
                    model_ema.update_parameters(model)
                global_steps += 1
                writer.add_scalar("train/loss", loss.detach().cpu().item(), global_steps)
                writer.add_scalar(
                    "train/lr", scheduler.get_last_lr()[0], global_steps
                )
                if j % args.log_freq == 0:
                    logger.info(
                        "Epoch[%d/%d],Step[%d/%d],loss:%.5f,lr:%.5f",
                        i + 1,
                        args.epochs,
                        j,
                        len(train_dataloader),
                        loss.detach().cpu().item(),
                        scheduler.get_last_lr()[0],
                    )
            ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}
            torch.save(
                ckpt, os.path.join(exp_dir, "steps_{:0>8}.pt".format(global_steps))
            )

            model_ema.eval()
            if initial_sampling_noise is None:
                initial_sampling_noise = torch.randn(
                    args.n_samples, 1, 28, 28, device=device
                )
                noise_path = os.path.join(exp_dir, "sampling_initial_noise.pt")
                torch.save(initial_sampling_noise.cpu(), noise_path)
                logger.info("Saved initial sampling noise to %s", noise_path)
            
            # Generate step noises if fixed_step_noises is enabled and not loaded from file
            if args.fixed_step_noises and step_noises is None:
                step_noises = torch.randn(
                    args.timesteps,
                    args.n_samples,
                    1,
                    28,
                    28,
                    device=device,
                )
                step_noises_path = os.path.join(exp_dir, "sampling_step_noises.pt")
                torch.save(step_noises.cpu(), step_noises_path)
                logger.info("Saved step noise sequence to %s", step_noises_path)
            
            # 解析采样时的类别标签与 CFG
            sampling_labels = None
            if args.conditional and args.labels.strip():
                parts = [p.strip() for p in args.labels.split(",") if p.strip()]
                if len(parts) == 1:
                    sampling_labels = int(parts[0])
                else:
                    if len(parts) != args.n_samples:
                        raise ValueError(
                            f"--labels 有 {len(parts)} 个值，需等于 --n_samples ({args.n_samples})"
                        )
                    sampling_labels = [int(p) for p in parts]
            cfg_scale = args.cfg_scale if args.conditional else 0.0

            nrow = int(math.sqrt(args.n_samples))
            samples, frames = model_ema.module.sampling(
                args.n_samples,
                clipped_reverse_diffusion=not args.no_clip,
                device=device,
                initial_noise=initial_sampling_noise,
                step_noises=step_noises if args.fixed_step_noises else None,
                return_frames=True,
                save_every_steps=args.gif_every_steps,
                labels=sampling_labels,
                cfg_scale=cfg_scale,
            )
            save_image(
                samples,
                os.path.join(exp_dir, "steps_{:0>8}.png".format(global_steps)),
                nrow=nrow,
            )
            save_denoising_gif(
                frames,
                os.path.join(exp_dir, "steps_{:0>8}.gif".format(global_steps)),
                nrow=nrow,
            )
            writer.add_images("samples", samples, global_steps)
    finally:
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
