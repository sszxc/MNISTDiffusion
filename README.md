# MNIST Diffusion
![60 epochs training from scratch](assets/demo.gif "60 epochs training from scratch")

Only simple depthwise convolutions, shorcuts and naive timestep embedding, there you have it! A fully functional denosing diffusion probabilistic model while keeps ultra light weight **4.55MB** (the checkpoint has 9.1MB but with ema model double the size).

## Training
Install packages
```bash
pip install -r requirements.txt
```
Start default setting training 
```bash
python train_mnist.py
```
Feel free to tuning training parameters, type `python train_mnist.py -h` to get help message of arguments.

## Reference
A neat blog explains how diffusion model works(must read!): https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

The Denoising Diffusion Probabilistic Models paper: https://arxiv.org/pdf/2006.11239.pdf 

A pytorch version of DDPM: https://github.com/lucidrains/denoising-diffusion-pytorch

## Playing
- [x] [理解 Stable Diffusion UNet 网络 « bang's blog](https://blog.cnbang.net/tech/3823/)
- [x] test 的时候使用使用同样的初始噪声
- [x] 可视化加噪过程
- [x] 保存训练相关的参数到 result 文件夹
- [ ] 比较不同的网络尺寸
- [ ] 条件生成 (Extra): 尝试加入类别标签（Digit Label），实现 Classifier-Free Guidance (CFG)，让模型能根据你的指令生成特定的数字