import numpy as np
import os

import torch
import torch.nn as nn
from PIL import Image
import torch.distributed as dist
from torchvision import transforms
from zero123 import dataset as data_transforms
from pix2vox.config import cfg
from zero123.diffusion_model import set_diffusion
from lrgt.losses.losses import DiceLoss

def unit_spherical_grid(nb_points, return_radian=False, return_vertices=False):
    """
    a function that samples a grid of sinze `nb_points` around a sphere of radius `r` . it returns azimth and elevation angels arouns the sphere. if `return_vertices` is true .. it returns the 3d points as well
    """
    r = 1.0
    vertices = []
    azim = []
    elev = []
    alpha = 4.0*np.pi*r*r/nb_points
    d = np.sqrt(alpha)
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alpha/d_nu
    count = 0
    for m in range(0, m_nu):
        nu = np.pi*(m+0.5)/m_nu
        m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
        for n in range(0, m_phi):
            phi = 2*np.pi*n/m_phi
            xp = r*np.sin(nu)*np.cos(phi)
            yp = r*np.sin(nu)*np.sin(phi)
            zp = r*np.cos(nu)
            vertices.append([xp, yp, zp])
            azim.append(phi)
            elev.append(nu-np.pi*0.5)
            count = count + 1
    if not return_radian:
        azim = np.rad2deg(azim)
        elev = np.rad2deg(elev)
    if return_vertices:
        return azim[:nb_points], elev[:nb_points], np.array(vertices[:nb_points])
    else:
        return azim[:nb_points], elev[:nb_points]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 4, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(4),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(4, 8, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 16, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )

    def forward(self, voxel):
        voxel = voxel.unsqueeze(dim=1) # torch.Size([batch_size, 1, 32, 32, 32])
        voxel = self.layer1(voxel) # torch.Size([batch_size, 4, 16, 16, 16])
        voxel = self.layer2(voxel) # torch.Size([batch_size, 8, 8, 8, 8])
        voxel = self.layer3(voxel) # torch.Size([batch_size, 16, 4, 4, 4])

        return voxel

class ViewSelector(nn.Module):
    def __init__(self, nb_views=24, shape_features_size=1024):
        super().__init__()
        self.nb_views = nb_views
        self.shape_features_size = shape_features_size
        views_elev = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        views_azim = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        self.view_transformer = nn.Sequential(
            nn.Linear(shape_features_size + 2*self.nb_views, shape_features_size),
            nn.BatchNorm1d(shape_features_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(shape_features_size, shape_features_size),
            nn.BatchNorm1d(shape_features_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(shape_features_size, 5 * self.nb_views),
            nn.BatchNorm1d(5 * self.nb_views),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(5 * self.nb_views, 2 * self.nb_views),
            nn.BatchNorm1d(2 * self.nb_views),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2 * self.nb_views, 2 * self.nb_views),
            nn.Tanh(),
        )
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_azim', views_azim)

    def forward(self, voxel_feature):
        voxel_feature = voxel_feature.view(-1, self.shape_features_size)
        batch_size = voxel_feature.size(0)
        c_views_elev = self.views_elev.expand(batch_size, self.nb_views)
        c_views_azim = self.views_azim.expand(batch_size, self.nb_views)
        c_views_azim = c_views_azim + \
                       torch.rand((batch_size, self.nb_views),
                                  device=c_views_azim.device) * 360.0 - 180.0
        c_views_elev = c_views_elev + \
                       torch.rand((batch_size, self.nb_views),
                                  device=c_views_elev.device) * 180.0 - 90.0
        adjutment_vector = self.view_transformer(
            torch.cat([voxel_feature, c_views_elev, c_views_azim], dim=1))
        adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
        return  adjutment_vector[0] * 90.0, adjutment_vector[1] * 180.0/self.nb_views

class MVTN(nn.Module):
    def __init__(self, nb_views, shape_features_size):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.view_selector = ViewSelector(nb_views, shape_features_size)

    def forward(self,voxel):
        voxel_feature = self.feature_extractor(voxel)
        return self.view_selector(voxel_feature)

class batch_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, images, poses, category, sample):
        self.images = images
        self.poses = poses
        self.category = category
        self.sample = sample

    def __len__(self):
        return self.images.size()[0]

    def __getitem__(self, item):
        return self.images[item], self.poses[item], self.category[item], self.sample[item]

def reduce_value(value):
    world_size = int(os.environ['WORLD_SIZE'])

    if world_size < 2:
        return value

    with torch.no_grad():
        torch.distributed.all_reduce(value)
        value /= world_size

    return value

def mvtn_search(args, model, mvtn_model, dataloader, epoch):
    device = torch.cuda.current_device()

    diffusion = set_diffusion()
    diffusion.eval()

    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

    transform = data_transforms.Compose([
        data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        data_transforms.ToTensor()
    ])

    optimizer = torch.optim.Adam(mvtn_model.parameters(), lr=0.0001)
    losses = AverageMeter()

    if args.attack_model == 'pix2vox':
        loss_function = torch.nn.BCELoss()
    elif args.attack_model == 'lrgt':
        loss_function = DiceLoss()

    for it in range(args.iteration):
        for batch_idx, (taxonomy_names, sample_names, data,
                        target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            with torch.no_grad():

                model['encoder'].eval()
                model['decoder'].eval()
                model['merger'].eval()
                if args.attack_model == 'pix2vox':
                    model['refiner'].eval()

                if args.attack_model == 'pix2vox':
                    image_features = model['encoder'](data)
                    raw_features, generated_volumes = model['decoder'](image_features)
                    generated_volumes = model['merger'](raw_features, generated_volumes)
                    generated_volumes = model['refiner'](generated_volumes)
                elif args.attack_model == 'lrgt':
                    image_features = model['encoder'](data)
                    generated_volumes = model['merger'](image_features)
                    generated_volumes = model['decoder'](generated_volumes).squeeze(dim=1)
                    generated_volumes = generated_volumes.clamp_max(1)
                generated_volumes = torch.ge(generated_volumes, 0.4).float()
                res = torch.pow(torch.sub(generated_volumes, target), 2)

            elev, azim = mvtn_model(res)

            image_bacth = []
            pose_batch = []
            for i in range(batch_size):
                if args.abo:
                    image = args.abo_path % (taxonomy_names[i], sample_names[i], 6)
                else:
                    image = args.trainset_path % (taxonomy_names[i], sample_names[i], 6)
                image = Image.open(image)
                image.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
                image = image.resize([256, 256], Image.Resampling.LANCZOS)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = transforms.ToTensor()(image).unsqueeze(0)
                image = image * 2 - 1
                image = transforms.functional.resize(image, [256, 256])
                image = image.expand(1, -1, -1, -1)
                image_bacth.append(image)

                pose = []
                pose_id = np.random.choice(np.arange(0, 24), size=1, replace=False)
                pose.append([torch.deg2rad(elev[i][pose_id[0]]), torch.deg2rad(azim[i][pose_id[0]])])
                pose_batch.append(torch.tensor(pose))

            image_bacth = torch.stack(image_bacth)
            pose_batch = torch.stack(pose_batch)

            with torch.no_grad():
                ft_image_batch = diffusion(args, image_bacth.to(device), pose_batch.to(device), transform, taxonomy_names, sample_names)
                ft_image_batch = torch.cat([ft_image_batch.to(device), data[:, :2]], dim=1)

            if args.attack_model == 'pix2vox':
                image_features = model['encoder'](ft_image_batch)
                raw_features, generated_volumes = model['decoder'](image_features)
                generated_volumes = model['merger'](raw_features, generated_volumes)
                generated_volumes = model['refiner'](generated_volumes)
            elif args.attack_model == 'lrgt':
                image_features = model['encoder'](ft_image_batch)
                generated_volumes = model['merger'](image_features)
                generated_volumes = model['decoder'](generated_volumes).squeeze(dim=1)
                generated_volumes = generated_volumes.clamp_max(1)
            loss = -loss_function(generated_volumes, target)

            mvtn_model.zero_grad()
            loss.backward()
            optimizer.step()

            loss = reduce_value(loss)

            if torch.distributed.get_rank() == 0:
                losses.update(loss.item())

            del data, target
            torch.cuda.empty_cache()

        if torch.distributed.get_rank() == 0:
            print(
                '[Iteration %d/%d] Loss = %.4f'
                % (it, args.iteration, -losses.avg))

    optimizer.zero_grad()

    if torch.distributed.get_rank() == 0:
        file_name = f'checkpoint_{epoch}.pth'

        output_path = os.path.join(args.mvtn_path, file_name + "_%s" % args.attack_model)
        if not os.path.exists(args.mvtn_path):
            os.makedirs(args.mvtn_path)

        checkpoint = {
            'epoch_idx': epoch,
            'mvtn_state_dict': mvtn_model.state_dict(),
        }

        torch.save(checkpoint, output_path)
        print('Saved checkpoint to %s ...' % output_path)

    torch.distributed.barrier()

    return mvtn_model

