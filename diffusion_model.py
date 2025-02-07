import os
import math
import random

import torch
import torchvision.transforms as transforms
from torch import autocast
import torch.distributed as dist
import numpy as np
from PIL import Image
from contextlib import nullcontext
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from omegaconf import OmegaConf
from einops import rearrange
from rich import print
import cv2

def set_diffusion(ckpt="zero123-xl.ckpt",
                 config_path="configs/sd-shapenet-finetune-c_concat-256.yaml"):

    assert os.path.exists(ckpt)
    device = torch.cuda.current_device()

    # Instantiate all models beforehand for efficiency.
    config = OmegaConf.load(config_path)
    print('Instantiating LatentDiffusion...')

    diffusion = instantiate_from_config(config.model)
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    diffusion.load_state_dict(sd, strict=False)

    model = Diffusion(diffusion)

    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[device], output_device=device, find_unused_parameters=False)
    return model


class Diffusion(torch.nn.Module):
    def __init__(self, model):
        super(Diffusion, self).__init__()

        self.model = model

    def get_device(self):
        return self.model.device

    def mix_gb(self, image_tensor):
        # 加载图像和背景
        background_path = '/home/ubuntu/zhouhang/sun2012pascalformat/sun2012pascalformat/JPEGImages'
        background_image = os.listdir(background_path)
        background_image = random.choice(background_image)
        background_image = os.path.join(background_path, background_image)
        background = Image.open(background_image)

        # 转换为 Tensor，并标准化
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        bg_tensor = preprocess(background)
        mask_tensor = torch.zeros_like(image_tensor)

        # 将像素值为 [1, 1, 1] 的像素点改为 0，不为 [1, 1, 1] 的像素点改为 1
        mask = torch.all(image_tensor >= torch.tensor([0.95, 0.95, 0.95], dtype=torch.float32).view(3, 1, 1), dim=0)
        mask_tensor[:, mask] = 0
        mask_tensor[:, ~mask] = 1

        mix_tensor = image_tensor * mask_tensor + bg_tensor * (1 - mask_tensor)

        return mix_tensor

    def render(self, args, sampler, image, pose, scale, h, w, i_view, taxonomy_names, sample_names, if_save=False):
        input_im = image.unsqueeze(dim=0)
        c = self.model.get_learned_conditioning(input_im).tile(1, 1, 1)
        T = torch.tensor([pose[0], math.sin(pose[1]), math.cos(pose[1]), 0])
        T = T[None, None, :].repeat(1, 1, 1).to(c.device)
        c = torch.cat([c, T], dim=-1).float()
        # print(c.size()) #[1, 1, 772]
        c = self.model.cc_projection(c)
        # print(c.size()) #[1, 1, 768]
        cond = {}
        cond['c_crossattn'] = [c]
        cond['c_concat'] = [
            self.model.encode_first_stage((input_im.to(c.device))).mode().detach()
            .repeat(1, 1, 1, 1)]
        if scale != 1.0:
            uc = {}
            uc['c_concat'] = [torch.zeros(1, 4, h // 8, w // 8).to(c.device)]
            uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
        else:
            uc = None

        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(S=50,
                                         conditioning=cond,
                                         batch_size=1,
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         eta=1.0,
                                         x_T=None)
        # print(samples_ddim.size()) #[1, 4, 32, 32]
        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
        # print(x_samples_ddim.size()) #[1, 3, 256, 256]
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        # print(x_samples_ddim.size()) #[1, 3, 256, 256]

        if args.pix3d:
            out = 255.0 * self.mix_gb(x_samples_ddim[0].cpu()).numpy()
            out = rearrange(out, 'c h w -> h w c')
        else:
            out = 255.0 * rearrange(x_samples_ddim[0].cpu().numpy(), 'c h w -> h w c')

        torch.cuda.empty_cache()

        return out

    def forward(self, args, images, poses, train_transforms, taxonomy_names, sample_names, if_save=False):
        # print(images.size())  # [batchsize, 3, 3, 224, 224]
        # print(poses.size())  # [batchsize, 3, 2]

        batch_size = images.size()[0]
        n_views = images.size()[1]

        precision = 'fp32'
        scale = 3.0
        h, w = 256, 256

        sampler = DDIMSampler(self.model)

        precision_scope = autocast if precision == 'autocast' else nullcontext
        with precision_scope('cuda'):
            with self.model.ema_scope():
                out_images = []
                for bs in range(batch_size):
                    outs = []
                    for i_view in range(n_views):
                        out = self.render(args, sampler, images[bs][i_view], poses[bs][i_view], scale, h, w, i_view, taxonomy_names[bs], sample_names[bs], if_save)
                        outs.append(out)
                    outs = train_transforms(np.asarray(outs))
                    out_images.append(outs)
                out_images = torch.stack(out_images)

        torch.cuda.empty_cache()

        return out_images.cpu()

    @torch.no_grad()
    def sample_model(self, input_im, elevation, azimuth, precision='autocast', ddim_eta=1.0, ddim_steps=100, scale=3.0, h=512, w=512):

        sampler = DDIMSampler(self.model)

        precision_scope = autocast if precision == 'autocast' else nullcontext
        with precision_scope("cuda"):
            with self.model.ema_scope():
                c = self.model.get_learned_conditioning(input_im).tile(1, 1, 1)

                T = torch.tensor([np.radians(elevation), np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth)), 0])
                T = T[None, None, :].float().to(c.device)
                c_T = torch.cat([c, T], dim=-1)
                c_T = self.model.cc_projection(c_T)
                cond = {}
                cond['c_crossattn'] = [c_T]
                cond['c_concat'] = [self.model.encode_first_stage(input_im).mode().detach()
                                    .repeat(1, 1, 1, 1)]
                if scale != 1.0:
                    uc = {}
                    uc['c_concat'] = [torch.zeros(1, 4, h // 8, w // 8).to(c.device)]
                    uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
                else:
                    uc = None

                shape = [4, h // 8, w // 8]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=cond,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta,
                                                 x_T=None)
                # print(samples_ddim.shape)
                # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
                x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                ret_img = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                del cond, c, x_samples_ddim, samples_ddim, uc, c_T, input_im, sampler
                torch.cuda.empty_cache()
                return ret_img

    @torch.no_grad()
    def predict_gradio(self, input_im, elevation, azimuth, ddim_steps=100, scale=3.0, save_path="instant_ngp"):
        # sampler.to(device)
        x_samples_ddims_8 = self.sample_model(input_im, elevation, azimuth,
                                                   ddim_steps=ddim_steps, scale=scale)
        out_image = rearrange(x_samples_ddims_8[0].numpy(), 'c h w -> h w c')
        save_image = Image.fromarray((out_image * 255.0).astype(np.uint8))
        if save_path:
            save_image.save(os.path.join(save_path, 'generate_image.png'))

        del x_samples_ddims_8
        return out_image
