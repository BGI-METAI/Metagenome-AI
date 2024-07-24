
import os
import paddle
import paddle.nn as nn
from matplotlib import pyplot as plt
%matplotlib inline
from tqdm import tqdm
from paddle import optimizer
# from utils import *
from modules import UNet    # 模型
import logging
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = paddle.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return paddle.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = paddle.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = paddle.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = paddle.randn(shape=x.shape)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return paddle.randint(low=1, high=self.noise_steps, shape=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with paddle.no_grad():
            x = paddle.randn((n, 3, self.img_size, self.img_size))
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):


                t = paddle.to_tensor([i] * x.shape[0]).astype("int64")
                # print(x.shape, t.shape)

                # print(f"完成第{i}步")
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = paddle.randn(shape=x.shape)
                else:
                    noise = paddle.zeros_like(x)
                x = 1 / paddle.sqrt(alpha) * (x - ((1 - alpha) / (paddle.sqrt(1 - alpha_hat))) * predicted_noise) + paddle.sqrt(beta) * noise
        model.train()
        x = (x.clip(-1, 1) + 1) / 2
        x = (x * 255)
        return x

def train(args):
    # setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)

    image = next(iter(dataloader))[0]

    model = UNet()
    opt = optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            # print(images)
            t = diffusion.sample_timesteps(images[0].shape[0])
            x_t, noise = diffusion.noise_images(images[0], t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)  # 损失函数

            opt.clear_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(MSE=loss.item())

            # print(("MSE", loss.item(), "global_step", epoch * l + i))
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        
        if epoch % 20 == 0:
            paddle.save(model.state_dict(), f"car_models/ddpm_uncond{epoch}.pdparams")
            sampled_images = diffusion.sample(model, n=8)

            for i in range(8):
                img = sampled_images[i].transpose([1, 2, 0])
                img = np.array(img).astype("uint8")
                plt.subplot(2,4,i+1)
                plt.imshow(img)
            plt.show()

def launch():
    import argparse

    # 参数设置
    class ARGS:
        def __init__(self):
            self.run_name = "DDPM_Uncondtional"
            self.epochs = 150
            self.batch_size = 24
            self.image_size = 64
            self.dataset_path = r"/home/aistudio/work/cars"
            self.device = "cuda"
            self.lr = 1.5e-4

    args = ARGS()
    train(args)


if __name__ == '__main__':
    launch()
    pass
