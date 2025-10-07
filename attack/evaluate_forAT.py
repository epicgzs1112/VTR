import numpy as np
import torch
import torch.distributed as dist
import sys
sys.path.append("..")
#import matplotlib.pyplot as plt
from zero123 import dataset as data_transforms
from pix2vox.config import cfg
from lrgt.losses.losses import DiceLoss
'''
    Define the adaptability of each NES solutions
'''

class batch_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, images, poses, labels, category_name, sample_name):
        self.images = images
        self.poses = poses
        self.labels = labels
        self.category_name = category_name
        self.sample_name = sample_name

    def __len__(self):
        return self.images.size()[0]

    def __getitem__(self, item):
        return self.images[item], self.poses[item], self.labels[item], self.category_name[item], self.sample_name[item]



def test_reconstruction(args, model, images_tensor, groundtrue_tensor):
    if args.attack_model == 'pix2vox':
        loss_function = torch.nn.BCELoss()
    elif args.attack_model == 'lrgt':
        loss_function = DiceLoss()
    with torch.no_grad():

        # Switch models to evaluation mode
        model['encoder'].eval()
        model['decoder'].eval()
        model['merger'].eval()
        if args.attack_model == 'pix2vox':
            model['refiner'].eval()

        # Test the encoder, decoder, refiner and merger
        if args.attack_model == 'pix2vox':
            image_features = model['encoder'](images_tensor)
            raw_features, generated_volumes = model['decoder'](image_features)
            generated_volumes = model['merger'](raw_features, generated_volumes)
            generated_volumes = model['refiner'](generated_volumes)
        elif args.attack_model == 'lrgt':
            image_features = model['encoder'](images_tensor)
            generated_volumes = model['merger'](image_features)
            generated_volumes = model['decoder'](generated_volumes).squeeze(dim=1)
            generated_volumes = generated_volumes.clamp_max(1)

        # Loss
        losses = []
        for i in range(groundtrue_tensor.size()[0]):
            loss = loss_function(generated_volumes[i], groundtrue_tensor[i])
            losses += [loss.cpu().detach()]

    torch.cuda.empty_cache()
    return losses

def comput_fitness(args, model, image, label, category_name, sample_name, solutions, diffusion):
    '''

    Args:
        solutions: The value of the parameter currently sampled
    Returns:
        reward: Fitness value
    '''
    popsize = solutions.shape[0]

    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

    transform = data_transforms.Compose([
        data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        data_transforms.ToTensor()
    ])

    poses = []

    # Render an image using the resulting parameters
    for i in range(popsize):
        pose = []
        pose.append([np.deg2rad(solutions[i][0]), np.deg2rad(solutions[i][1])])
        # theta_2, theta_3 = find_equilateral_triangle_vertices(1.2 * np.sin(solutions[0]), solutions[1])
        theta_2, theta_3 = solutions[i][1] + 120, solutions[i][1] + 240
        if theta_2 > 180:
            theta_2 = theta_2 - 360
        if theta_3 > 180:
            theta_3 = theta_3 - 360
        pose.append([np.deg2rad(solutions[i][0]), np.deg2rad(theta_2)])
        pose.append([np.deg2rad(solutions[i][0]), np.deg2rad(theta_3)])

        poses.append(torch.tensor(pose))

    poses = torch.stack(poses)

    dataset = batch_dataset(image, poses, label, category_name, sample_name)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
    loader.sampler.set_epoch(0)
    device = torch.cuda.current_device()

    losses = []
    for image, pose, label, category_name, sample_name in loader:
        with torch.no_grad():
            gen_image = diffusion(args, image.to(device), pose.to(device), transform, category_name, sample_name)
            loss = test_reconstruction(args, model, gen_image.to(device), label.to(device))
            losses.append(loss)
        del gen_image, image, pose, label
        torch.cuda.empty_cache()

    loss_tensor = torch.tensor(losses).to(device)
    loss_tensor_list = [torch.zeros_like(loss_tensor) for _ in range(args.world_size)]
    dist.all_gather(loss_tensor_list, loss_tensor)
    loss_list = []
    for i in range(len(loss_tensor_list)):
        for l in loss_tensor_list[i]:
            loss_list.append(l.cpu().detach().numpy())

    torch.cuda.synchronize(torch.device(torch.cuda.current_device()))

    return loss_list