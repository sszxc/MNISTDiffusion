"""
可视化 MNIST 前向扩散：从 t=0 到 t=T 的加噪过程。
确保 t=T 时图像完全变成白噪声，并保存为 GIF 动图。
"""
import os
import sys

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 保证能从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MNISTDiffusion


# 6*6 一起看
GRID_SIZE = 6
N_IMAGES = GRID_SIZE * GRID_SIZE  # 36
UPSCALE = 5  # 最近邻放大倍数


def load_mnist_batch(n, image_size=28):
    """加载 n 张 MNIST 图片，与 train_mnist 相同的预处理：[0,1] -> [-1,1]。返回 (n, 1, H, W)。"""
    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
    xs = [dataset[i][0].unsqueeze(0) for i in range(n)]
    return torch.cat(xs, dim=0)


def forward_diffusion_at_t(diffusion: MNISTDiffusion, x_0, t, noise, device):
    """对 x_0 在给定 t 和固定 noise 下做前向扩散，得到 x_t。支持 batch，(N,1,H,W)。"""
    t_tensor = torch.full((x_0.shape[0],), t, dtype=torch.long, device=device)
    return diffusion._forward_diffusion(x_0, t_tensor, noise)


def tensor_to_pil(x):
    """(1,1,H,W) 或 (1,H,W)，值域 [0,1]，转为 PIL Image（灰度）。"""
    x = x.squeeze().cpu().float().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255).astype(np.uint8)
    return Image.fromarray(x, mode="L")


def make_grid_frame(images_pil, upscale=UPSCALE, grid_size=GRID_SIZE):
    """将 grid_size*grid_size 张 PIL 图先最近邻放大 upscale 倍，再拼成一张大图。"""
    w = images_pil[0].width * upscale
    h = images_pil[0].height * upscale
    total_w = grid_size * w
    total_h = grid_size * h
    out = Image.new("L", (total_w, total_h), 0)
    for i, img in enumerate(images_pil):
        img_big = img.resize((w, h), Image.NEAREST)
        row, col = i // grid_size, i % grid_size
        out.paste(img_big, (col * w, row * h))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 28
    timesteps = 1000
    # 采样多少帧（从 t=0 到 t=T）
    num_frames = 25
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 模型只用来取 schedule 的 buffer，不加载权重
    diffusion = MNISTDiffusion(
        image_size=image_size,
        in_channels=1,
        timesteps=timesteps,
    ).to(device)
    diffusion.eval()

    # 取 6*6=36 张 MNIST
    x_0 = load_mnist_batch(N_IMAGES, image_size).to(device)
    # 每张图固定一条噪声轨迹
    noise = torch.randn_like(x_0, device=device)

    # 在 t=0, t≈T/(num_frames-1), ..., t=T-1 处采样
    t_indices = np.linspace(0, timesteps - 1, num_frames, dtype=int)

    frames = []
    for t in t_indices:
        with torch.no_grad():
            x_t = forward_diffusion_at_t(diffusion, x_0, t, noise, device)
        # 从 [-1,1] 转到 [0,1]，每张转成 PIL 并最近邻放大 5 倍
        x_vis = (x_t + 1.0) / 2.0
        images_pil = [tensor_to_pil(x_vis[i : i + 1]) for i in range(x_vis.shape[0])]
        grid_img = make_grid_frame(images_pil, upscale=UPSCALE, grid_size=GRID_SIZE)
        # 在放大后的网格图上加 t 标签
        try:
            draw = ImageDraw.Draw(grid_img)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        draw.text((8, 8), f"t={t}", fill=255, font=font)
        frames.append(grid_img)

    # 保存 GIF
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, "add_noise_vis.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,  # 每帧约 120ms
        loop=0,
    )
    print(f"动图已保存: {gif_path}")

    # 验证 t=T-1 时接近纯噪声
    with torch.no_grad():
        x_T = forward_diffusion_at_t(diffusion, x_0, timesteps - 1, noise, device)
    alpha_T = diffusion.alphas_cumprod[timesteps - 1].item()
    sqrt_alpha_T = np.sqrt(alpha_T)
    sqrt_one_minus_alpha_T = np.sqrt(1.0 - alpha_T)
    # x_T = sqrt_alpha_T * x_0 + sqrt_one_minus_alpha_T * noise
    # 若 alpha_T ≈ 0，则 x_T ≈ noise
    print(f"t=T-1 时 alpha_cumprod[T-1] = {alpha_T:.6f}")
    print(f"  sqrt(alpha_cumprod) = {sqrt_alpha_T:.6f}, sqrt(1-alpha_cumprod) = {sqrt_one_minus_alpha_T:.6f}")
    if alpha_T < 0.01:
        print("  -> t=T 时图像已基本为白噪声，符合预期。")
    else:
        print("  -> 若 alpha 偏大，可检查 variance schedule。")


if __name__ == "__main__":
    main()
