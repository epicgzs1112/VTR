from __future__ import print_function

from time import time
import datetime
import os
import json
import numpy as np
import random
import torch
import torch.distributed as dist
from PIL import Image
from rich import print
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset.opts import get_opts
from dataset.shapenet import ShapeNetDataLoader, DatasetType, Pix3dDataLoader, ABODataLoader
import dataset.data_transforms
from attack.NES_GMM_forAT import NES_GMM_search
from attack.mvtn import mvtn_search, MVTN
from attack.ours import ours_search
from average_meter import AverageMeter
from pix2vox.config import cfg
from lrgt.losses.losses import DiceLoss
from diffusion_model import set_diffusion
from reconstruction_model import pix2vox, opt_pix2vox, lrgt, opt_lrgt
from test_on_voxel import test_net

import sys
sys.path.insert(0, sys.path[0]+"/../")

args = get_opts()

args.rank = int(os.environ["RANK"])
args.world_size = int(os.environ['WORLD_SIZE'])
args.gpu = int(os.environ['LOCAL_RANK'])

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=5400))
#dist.init_process_group(backend='gloo', world_size=args.world_size, rank=args.rank)
torch.cuda.set_device(args.gpu)
torch.distributed.barrier()

@torch.no_grad()
def render_image(all_args, imgs_list, taxonomy_names, sample_names, diffusion, train_transforms):
    images = []
    poses = []
    categorys = []
    samples = []
    batch_size = all_args.shape[0]
    for i in range(batch_size):
        pose = []
        if args.AT_type == 'ours' or args.AT_type == 'random' or args.AT_type == 'nature' or args.AT_type == 'MVTN':
            for j in range(args.n_views):
                pose.append([np.deg2rad(all_args[i][j][0]), np.deg2rad(all_args[i][j][1])])
        else:
            pose.append([np.deg2rad(all_args[i][0]), np.deg2rad(all_args[i][1])])
            theta_2, theta_3 = all_args[i][1] + 120, all_args[i][1] + 240
            if theta_2 > 180:
                theta_2 = theta_2 - 360
            if theta_3 > 180:
                theta_3 = theta_3 - 360
            pose.append([np.deg2rad(all_args[i][0]), np.deg2rad(theta_2)])
            pose.append([np.deg2rad(all_args[i][0]), np.deg2rad(theta_3)])

        #process_image
        img = Image.open(imgs_list[i])
        img.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
        img = img.resize([256, 256], Image.Resampling.LANCZOS)
        img = np.asarray(img, dtype=np.float32) / 255.0
        if(img.shape[2] == 4):
            alpha = img[:, :, 3:4]
            white_im = np.ones_like(img)
            img = alpha * img + (1.0 - alpha) * white_im
            img = img[:, :, 0:3]
        img = transforms.ToTensor()(img).unsqueeze(0)
        img = img * 2 - 1
        img = transforms.functional.resize(img, [256, 256])

        if args.AT_type == 'MVTN':
            image = img.expand(1, -1, -1, -1)
        else:
            image = img.expand(args.n_views, -1, -1, -1)
        category = imgs_list[i].split('/')[-3]
        sample = imgs_list[i].split('/')[-2]

        images.append(image)
        poses.append(torch.tensor(pose))
        categorys.append(category)
        samples.append(sample)

    images = torch.stack(images)
    poses = torch.stack(poses)
    device = torch.cuda.current_device()
    diffusion.eval()
    with torch.no_grad():
        out = diffusion(args, images.to(device), poses.to(device), train_transforms, taxonomy_names, sample_names, if_save=True)

    del images, poses, image, img
    torch.cuda.empty_cache()

    return out

def GMSampler(dist_pool_mu, dist_pool_sigma, img, taxonomy_names, sample_names, diffusion, transforms):
    M = dist_pool_mu.shape[0]
    # N = dist_pool_mu.shape[1]
    k = dist_pool_mu.shape[2]
    num_para = args.search_num
    batch_size = img.size()[0]

    if args.abo:
        label_list = os.listdir(f'{args.abo_path}'.split('%s')[0])
    elif args.pix3d:
        label_list = os.listdir(f'{args.trainset_pix3d_path}'.split('%s')[0])
    else:
        label_list = os.listdir(f'{args.trainset_path}'.split('%s')[0])
    label_list.sort()

    a = [90, 180]
    b = [0, 0]

    sample_all = np.zeros([batch_size, num_para])
    imgs_list = []
    for i in range(batch_size):
        sample = np.zeros(num_para)
        class_id = label_list.index(taxonomy_names[i])
        mu = dist_pool_mu[class_id, :, :].squeeze()
        sigma = dist_pool_sigma[class_id, :, :].squeeze()
        F = np.random.choice(a=np.arange(k), size=num_para, replace=True, p=np.ones(k) / k)
        for j in range(num_para):
            L = int(F[j])
            if args.num_k == 1:
                sample[j] = a[j] * np.tanh(np.random.normal(loc=mu[j], scale=sigma[j], size=1)) + b[j]
            else:
                sample[j] = a[j] * np.tanh(np.random.normal(loc=mu[L, j], scale=sigma[L, j], size=1)) + b[j]  # 得到一个viewpoint的一组参数
        sample_all[i, :] = sample
        if args.pix3d:
            imgs_list.append(args.trainset_pix3d_path % (taxonomy_names[i], sample_names[i], 14))
        elif args.abo:
            imgs_list.append(args.abo_path % (taxonomy_names[i], sample_names[i], 6))
        else:
            imgs_list.append(args.trainset_path % (taxonomy_names[i], sample_names[i], 6))
    render_imgs = render_image(sample_all, imgs_list, taxonomy_names, sample_names, diffusion, transforms)
    # print(render_imgs.size()) # [batch_size, n_view, 3, 224, 224]
    # print(labels.size()) # [batch_size, 32, 32, 32]
    del img
    torch.cuda.empty_cache()

    return render_imgs

def MVTNSampler(img, vox, model, mvtn_model, taxonomy_names, sample_names, diffusion, transforms):
    device = torch.cuda.current_device()
    batch_size = img.size()[0]

    img, vox = img.to(device), vox.to(device)

    with torch.no_grad():

        model['encoder'].eval()
        model['decoder'].eval()
        model['merger'].eval()
        if args.attack_model == 'pix2vox':
            model['refiner'].eval()
        mvtn_model.eval()

        if args.attack_model == 'pix2vox':
            image_features = model['encoder'](img)
            raw_features, generated_volumes = model['decoder'](image_features)
            generated_volumes = model['merger'](raw_features, generated_volumes)
            generated_volumes = model['refiner'](generated_volumes)
        elif args.attack_model == 'lrgt':
            image_features = model['encoder'](img)
            generated_volumes = model['merger'](image_features)
            generated_volumes = model['decoder'](generated_volumes).squeeze(dim=1)
            generated_volumes = generated_volumes.clamp_max(1)
        generated_volumes = torch.ge(generated_volumes, 0.4).float()
        res = torch.pow(torch.sub(generated_volumes, vox), 2)

        elev, azim = mvtn_model(res)

    random_indices = torch.randperm(24)[:args.n_views]
    elev, azim = elev[:, random_indices], azim[:, random_indices]

    sample_all = torch.stack([elev, azim], dim=2).cpu().detach().numpy()
    imgs_list = []
    for i in range(batch_size):
        if args.pix3d:
            imgs_list.append(args.trainset_pix3d_path % (taxonomy_names[i], sample_names[i], 14))
        elif args.abo:
            imgs_list.append(args.abo_path % (taxonomy_names[i], sample_names[i], 6))
        else:
            imgs_list.append(args.trainset_path % (taxonomy_names[i], sample_names[i], 6))
    render_imgs = render_image(sample_all, imgs_list, taxonomy_names, sample_names, diffusion, transforms)
    render_imgs = torch.cat([render_imgs.to(device), img[:, :2]], dim=1)

    del img, vox
    torch.cuda.empty_cache()
    return render_imgs.cpu()

def RandomSampler(img, taxonomy_names, sample_names, diffusion, transforms):
    num_para = args.search_num
    batch_size = img.size()[0]

    a = [180.0, 360.0]
    b = [-90.0, -180.0]

    sample_all = np.zeros([batch_size, args.n_views, num_para])
    imgs_list = []
    for i in range(batch_size):
        sample = np.zeros([args.n_views, num_para])
        for j in range(args.n_views):
            sample[j, 0] = a[0] * np.random.random(1) + b[0]
            sample[j, 1] = a[1] * np.random.random(1) + b[1]
        sample_all[i, :, :] = sample
        if args.pix3d:
            imgs_list.append(args.trainset_pix3d_path % (taxonomy_names[i], sample_names[i], 14))
        elif args.abo:
            imgs_list.append(args.abo_path % (taxonomy_names[i], sample_names[i], 6))
        else:
            imgs_list.append(args.trainset_path % (taxonomy_names[i], sample_names[i], 6))
    render_imgs = render_image(sample_all, imgs_list, taxonomy_names, sample_names, diffusion, transforms)

    del img
    torch.cuda.empty_cache()

    return render_imgs

def NatureSampler(img, taxonomy_names, sample_names, diffusion, transforms):
    num_para = args.search_num
    batch_size = img.size()[0]

    a = [0.0, 360.0]
    b = [0.0, -180.0]

    sample_all = np.zeros([batch_size, args.n_views, num_para])
    imgs_list = []
    for i in range(batch_size):
        sample = np.zeros([args.n_views, num_para])
        for j in range(args.n_views):
            sample[j, 0] = a[0] * np.random.random(1) + b[0]
            sample[j, 1] = a[1] * np.random.random(1) + b[1]
        sample_all[i, :, :] = sample
        if args.pix3d:
            imgs_list.append(args.trainset_pix3d_path % (taxonomy_names[i], sample_names[i], 14))
        elif args.abo:
            imgs_list.append(args.abo_path % (taxonomy_names[i], sample_names[i], 6))
        else:
            imgs_list.append(args.trainset_path % (taxonomy_names[i], sample_names[i], 6))
    render_imgs = render_image(sample_all, imgs_list, taxonomy_names, sample_names, diffusion, transforms)

    del img
    torch.cuda.empty_cache()

    return render_imgs

def OursSampler(rotation_pool, img, taxonomy_names, sample_names, diffusion, transforms):
    num_para = args.search_num
    batch_size = img.size()[0]

    sample_all = np.zeros([batch_size, args.n_views, num_para])
    imgs_list = []
    for i in range(batch_size):
        '''if taxonomy_names[i] in rotation_pool and sample_names[i] in rotation_pool[taxonomy_names[i]]:
            tran = random.sample(rotation_pool[taxonomy_names[i]][sample_names[i]], 1)[0]'''
        if taxonomy_names[i] in rotation_pool:
            tran = random.sample(rotation_pool[taxonomy_names[i]], 1)[0]
        else:
            tran = np.zeros((args.n_views, num_para))
            for j in range(args.n_views):
                tran[j][0] = 180 * np.random.random(1) - 90
                tran[j][1] = 360 * np.random.random(1) - 180
        sample_all[i, :] = tran
        if args.pix3d:
            imgs_list.append(args.trainset_pix3d_path % (taxonomy_names[i], sample_names[i], 14))
        elif args.abo:
            imgs_list.append(args.abo_path % (taxonomy_names[i], sample_names[i], 6))
        else:
            imgs_list.append(args.trainset_path % (taxonomy_names[i], sample_names[i], 6))
    render_imgs = render_image(sample_all, imgs_list, taxonomy_names, sample_names, diffusion, transforms)
    # print(render_imgs.size()) # [batch_size, n_view, 3, 224, 224]
    # print(labels.size()) # [batch_size, 32, 32, 32]
    del img
    torch.cuda.empty_cache()

    return render_imgs

# ------------------------------------------------------------------------------------------------------------- #
def generate_dataset(dataloader, model=None, transforms=None, dist_pool_mu=None, dist_pool_sigma=None, rotation_pool=None, mvtn_model=None):
    rot_data = {}
    rot_size = int(len(dataloader) * args.ft_rate)

    diffusion = set_diffusion()
    for_tqdm = tqdm(enumerate(dataloader), total=rot_size, position=0, desc='generate novel views')
    for batch_idx, (taxonomy_names, sample_names, data, target) in for_tqdm:
        if batch_idx < rot_size:
            if args.AT_type == 'VIAT':
                data = GMSampler(dist_pool_mu, dist_pool_sigma, data, taxonomy_names, sample_names, diffusion, transforms)
            elif args.AT_type == 'random':
                data = RandomSampler(data, taxonomy_names, sample_names, diffusion, transforms)
            elif args.AT_type == 'nature':
                data = NatureSampler(data, taxonomy_names, sample_names, diffusion, transforms)
            elif args.AT_type == 'MVTN':
                data = MVTNSampler(data, target, model, mvtn_model, taxonomy_names, sample_names, diffusion, transforms)
            elif args.AT_type == 'ours':
                data = OursSampler(rotation_pool, data, taxonomy_names, sample_names, diffusion, transforms)
            if batch_idx == 0:
                rot_data['taxonomy_names'] = [taxonomy_names]
                rot_data['sample_names'] = [sample_names]
                rot_data['data'] = [data]
                rot_data['target'] = [target]
            else:
                rot_data['taxonomy_names'].append(taxonomy_names)
                rot_data['sample_names'].append(sample_names)
                rot_data['data'].append(data)
                rot_data['target'].append(target)
        else:
            break

    return rot_data

def train(args, model, train_loader, optimizer, scheduler,
          epoch, dist_pool_mu=None, dist_pool_sigma=None, mvtn_model=None, rotation_pool=None, writer=None, transforms=None):
    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    dice_loss = DiceLoss()

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', 'finetune')
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'

    rot_data = generate_dataset(train_loader, model, transforms, dist_pool_mu, dist_pool_sigma, rotation_pool, mvtn_model)

    n_batches = len(train_loader)
    rot_size = int(n_batches * args.ft_rate)

    for epoch_idx in range(epoch, epoch + args.epochs_fintune):
        # Tick / tock
        epoch_start_time = time()

        losses = AverageMeter()

        model['encoder'].train()
        model['decoder'].train()
        model['merger'].train()
        if args.attack_model == 'pix2vox':
            model['refiner'].train()

        for batch_idx, (taxonomy_names, sample_names, data,
                        target) in enumerate(train_loader):
            if batch_idx < rot_size:
                d, t = rot_data['data'][batch_idx], rot_data['target'][batch_idx]
                # data, target = torch.concat([data[:, :5, :, :, :], d], dim=1), t
                data, target = d, t

            device = torch.cuda.current_device()
            data, target = data.to(device), target.to(device)

            # Train the encoder, decoder, refiner, and merger
            if args.attack_model == 'pix2vox':
                image_features = model['encoder'](data)
                raw_features, generated_volumes = model['decoder'](image_features)
                generated_volumes = model['merger'](raw_features, generated_volumes)
                generated_volumes = model['refiner'](generated_volumes)
                loss = bce_loss(generated_volumes, target)

                # Gradient decent
                model['encoder'].zero_grad()
                model['decoder'].zero_grad()
                model['refiner'].zero_grad()
                model['merger'].zero_grad()

                loss.backward()

                optimizer['encoder'].step()
                optimizer['decoder'].step()
                optimizer['refiner'].step()
                optimizer['merger'].step()

            elif args.attack_model == 'lrgt':
                image_features = model['encoder'](data)
                # merger
                context = model['merger'](image_features)
                # decoder
                generated_volumes = model['decoder'](context).squeeze(dim=1)
                # Loss
                loss = dice_loss(generated_volumes, target)

                # Gradient decent
                model['encoder'].zero_grad()
                model['decoder'].zero_grad()
                model['merger'].zero_grad()

                loss.backward()

                optimizer['encoder'].step()
                optimizer['decoder'].step()
                optimizer['merger'].step()

            if args.world_size >= 2:
                dist.all_reduce(loss)
                loss /= args.world_size

            if torch.distributed.get_rank() == 0:
                losses.update(loss.item())
                # Append loss to TensorBoard
                n_itr = epoch_idx * n_batches + batch_idx
                writer.add_scalar('BatchLoss', loss.item(), n_itr)

        torch.cuda.synchronize(torch.device(torch.cuda.current_device()))

        # Adjust learning rate
        scheduler['encoder'].step()
        scheduler['decoder'].step()
        scheduler['merger'].step()
        if args.attack_model == 'pix2vox':
            scheduler['refiner'].step()

        if torch.distributed.get_rank() == 0:
            epoch_end_time = time()
            writer.add_scalar('EpochLoss', losses.avg, epoch_idx + 1)
            print(
                '[Epoch %d/%d] EpochTime = %.3f (s) Loss = %.4f'
                % (epoch_idx + 1, args.epochs, epoch_end_time - epoch_start_time, losses.avg))
            if args.attack_model == 'pix2vox':
                print('LearningRate:\tencoder: %f | decoder: %f | merger: %f | refiner: %f' %
                      (scheduler['encoder'].optimizer.param_groups[0]['lr'],
                       scheduler['decoder'].optimizer.param_groups[0]['lr'],
                       scheduler['merger'].optimizer.param_groups[0]['lr'],
                       scheduler['refiner'].optimizer.param_groups[0]['lr']))
            elif args.attack_model == 'lrgt':
                print('LearningRate:\tencoder: %f | decoder: %f | merger: %f' %
                      (scheduler['encoder'].optimizer.param_groups[0]['lr'],
                       scheduler['decoder'].optimizer.param_groups[0]['lr'],
                       scheduler['merger'].optimizer.param_groups[0]['lr']))

        torch.cuda.empty_cache()

    model['encoder'].zero_grad()
    model['decoder'].zero_grad()
    model['merger'].zero_grad()
    if args.attack_model == 'pix2vox':
        model['refiner'].zero_grad()

    # Save weights to file
    if torch.distributed.get_rank() == 0:
        file_name = f'checkpoint_{epoch_idx}.pth'

        path = args.model_dir % (args.attack_model, args.interval_degree) + args.AT_type
        if args.pix3d:
            path = path + '_pix3d'
        output_path = os.path.join(path, file_name)
        if not os.path.exists(path):
            os.makedirs(path)

        checkpoint = {
            'epoch_idx': epoch_idx,
            'encoder_state_dict': model['encoder'].state_dict(),
            'decoder_state_dict': model['decoder'].state_dict(),
            'merger_state_dict': model['merger'].state_dict()
        }
        if args.attack_model == 'pix2vox':
            checkpoint['refiner_state_dict'] = model['refiner'].state_dict()

        torch.save(checkpoint, output_path)
        print('Saved checkpoint to %s ...' % output_path)

    torch.distributed.barrier()

    if args.AT_type == 'ours':
        return model, optimizer, scheduler, rot_data

    return model, optimizer, scheduler


def eval_test(model, test_loader, epoch=-1, best_iou=-1, writer=None):

    device = torch.cuda.current_device()
    test_iou = []
    losses = AverageMeter()
    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    dice_loss = DiceLoss()

    model['encoder'].eval()
    model['decoder'].eval()
    model['merger'].eval()
    if args.attack_model == 'pix2vox':
        model['refiner'].eval()
    for_tqdm = tqdm(enumerate(test_loader), total=len(test_loader), position=0)
    for batch_idx, (taxonomy_names, sample_names, data,
                    target) in for_tqdm:
        with torch.no_grad():
            data, target = data.to(device), target.to(device)

            if args.attack_model == 'pix2vox':
                image_features = model['encoder'](data)
                raw_features, generated_volumes = model['decoder'](image_features)
                generated_volumes = model['merger'](raw_features, generated_volumes)
                generated_volumes = model['refiner'](generated_volumes)
                loss = bce_loss(generated_volumes, target)
            elif args.attack_model == 'lrgt':
                image_features = model['encoder'](data)
                generated_volumes = model['merger'](image_features)
                generated_volumes = model['decoder'](generated_volumes).squeeze(dim=1)
                generated_volumes = generated_volumes.clamp_max(1)
                loss = dice_loss(generated_volumes, target)

            if args.world_size >= 2:
                dist.all_reduce(loss)
                loss /= args.world_size

            losses.update(loss.item())

            _volume = torch.ge(generated_volumes, 0.4).float()
            intersection = torch.sum(_volume.mul(target)).float()
            union = torch.sum(torch.ge(_volume.add(target), 1)).float()
            test_iou.append((intersection / union).unsqueeze(dim=0))

            del data, target
            torch.cuda.empty_cache()

    test_iou = torch.cat(test_iou, dim=0).unsqueeze(dim=1)
    all_test_iou = [torch.zeros_like(test_iou) for _ in range(args.world_size)]
    torch.distributed.all_gather(all_test_iou, test_iou)
    if torch.distributed.get_rank() == 0:
        redundancy = len(test_loader) % args.world_size
        if redundancy == 0:
            redundancy = args.world_size
        for i in range(args.world_size):
            all_test_iou[i] = all_test_iou[i] \
                if i < redundancy else all_test_iou[i][:-1, :]
        all_test_iou = torch.cat(all_test_iou, dim=0).cpu().numpy()  # [sample_num, 1]

    torch.cuda.synchronize(torch.device(torch.cuda.current_device()))
    if torch.distributed.get_rank() == 0:
        mean_iou = np.sum(all_test_iou, axis=0) / len(all_test_iou)
        print('[Epoch %d] IoU = %.4f' % (epoch, mean_iou))

        writer.add_scalar('EpochLoss', losses.avg, epoch)
        writer.add_scalar('IoU', mean_iou, epoch)

        if mean_iou > best_iou:
            best_iou = mean_iou
            file_name = f'checkpoint_best.pth'
            path = args.model_dir % (args.attack_model, args.interval_degree) + args.AT_type
            if args.pix3d:
                path = path + '_pix3d'
            output_path = os.path.join(path, file_name)
            if not os.path.exists(path):
                os.makedirs(path)
            checkpoint = {
                'epoch_idx': epoch,
                'encoder_state_dict': model['encoder'].state_dict(),
                'decoder_state_dict': model['decoder'].state_dict(),
                'merger_state_dict': model['merger'].state_dict()
            }
            if args.attack_model == 'pix2vox':
                checkpoint['refiner_state_dict'] = model['refiner'].state_dict()
            torch.save(checkpoint, output_path)
            print('Saved checkpoint to %s ...' % output_path)

    torch.distributed.barrier()

    return best_iou


# ----------------------------------------------------------------------------------------------------------#

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.cuda.current_device()

    # setup data loader
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    if args.attack_model == 'pix2vox':
        train_transforms = dataset.data_transforms.Compose([
            dataset.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
            dataset.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
            dataset.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
            dataset.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
            dataset.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            dataset.data_transforms.RandomFlip(),
            dataset.data_transforms.RandomPermuteRGB(),
            dataset.data_transforms.ToTensor(),
        ])
        val_transforms = dataset.data_transforms.Compose([
            dataset.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            dataset.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            dataset.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            dataset.data_transforms.ToTensor(),
        ])
    elif args.attack_model == 'lrgt':
        train_transforms = dataset.data_transforms.Compose([
            dataset.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            dataset.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
            dataset.data_transforms.ToTensor(),
            dataset.data_transforms.normalize
        ])
        val_transforms = dataset.data_transforms.Compose([
            dataset.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            dataset.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            dataset.data_transforms.ToTensor(),
            dataset.data_transforms.normalize
        ])

    if args.test:
        # setup data loader
        test_transforms = dataset.data_transforms.Compose([
            dataset.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            dataset.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            dataset.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            dataset.data_transforms.ToTensor(),
        ])
        if args.pix3d:
            test_dataset = Pix3dDataLoader(args).get_dataset(
                DatasetType.VAL, args.n_views, test_transforms)
        elif args.abo:
            test_dataset = ABODataLoader(args).get_dataset(
                DatasetType.TEST, args.n_views, test_transforms)
        else:
            test_dataset = ShapeNetDataLoader(args, rotation=args.test_rot, half=args.half).get_dataset(
                DatasetType.TEST, args.n_views, test_transforms)

        test_dataset_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size,
                                                       num_workers=cfg.CONST.NUM_WORKER,
                                                       pin_memory=True, shuffle=False, sampler=test_dataset_sampler)
        test_net(args, test_data_loader)

    else:
        if args.pix3d:
            train_dataset = ShapeNetDataLoader(args, rotation=False, pix3d=True).get_dataset(
                DatasetType.TRAIN, args.n_views, train_transforms)
            val_dataset = Pix3dDataLoader(args).get_dataset(
                DatasetType.VAL, args.n_views, val_transforms)
        elif args.abo:
            train_dataset = ABODataLoader(args).get_dataset(
                DatasetType.TRAIN, args.n_views, train_transforms)
            val_dataset = ABODataLoader(args).get_dataset(
                DatasetType.VAL, args.n_views, val_transforms)
        else:
            train_dataset = ShapeNetDataLoader(args, rotation=False).get_dataset(
                DatasetType.TRAIN, args.n_views, train_transforms)
            val_dataset = ShapeNetDataLoader(args, rotation=True).get_dataset(
                DatasetType.TEST, args.n_views, val_transforms)

        train_dataset_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

        train_batch_sampler = torch.utils.data.BatchSampler(train_dataset_sampler, args.train_batch_size,
                                                            drop_last=True)

        val_dataset_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        num_workers=cfg.CONST.NUM_WORKER,
                                                        pin_memory=True,
                                                        batch_sampler=train_batch_sampler)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=args.test_batch_size,
                                                      num_workers=cfg.CONST.NUM_WORKER,
                                                      pin_memory=True,
                                                      sampler=val_dataset_sampler)

        if torch.distributed.get_rank() == 0:
            writer_path = args.logs_dir % (args.attack_model, args.interval_degree) + args.AT_type
            if args.pix3d:
                writer_path = writer_path + '_pix3d'
            train_writer = SummaryWriter(os.path.join(writer_path, 'train'))
            val_writer = SummaryWriter(os.path.join(writer_path, 'test'))
        else:
            train_writer = None
            val_writer = None

        init_epoch = 0
        best_iou = 0

        if args.attack_model == 'pix2vox':
            if init_epoch == 0:
                if args.pix3d:
                    ckpt = args.pix3d_ckpt_path % args.attack_model
                else:
                    ckpt = args.ckpt_path % args.attack_model
            else:
                ckpt = os.path.join(args.model_dir % (args.attack_model, args.interval_degree) + args.AT_type, 'checkpoint_%d.pth' % (init_epoch-1))
            model = pix2vox(args, ckpt)
            optimizer, scheduler = opt_pix2vox(args, model)
        elif args.attack_model == 'lrgt':
            if init_epoch == 0:
                if args.pix3d:
                    ckpt = args.pix3d_ckpt_path % args.attack_model
                else:
                    ckpt = args.ckpt_path % args.attack_model
            else:
                ckpt = os.path.join(args.model_dir % (args.attack_model, args.interval_degree) + args.AT_type, 'checkpoint_%d.pth' % (init_epoch-1))
            model = lrgt(args, ckpt)
            optimizer, scheduler = opt_lrgt(args, model)

        if args.AT_type == 'MVTN':
            mvtn_model = MVTN(nb_views=24, shape_features_size=1024)
            mvtn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mvtn_model)
            mvtn_model = torch.nn.parallel.DistributedDataParallel(mvtn_model.to(device), find_unused_parameters=False,
                                                                   device_ids=[device], output_device=device)
            if init_epoch != 0:
                ckpt = os.path.join(args.mvtn_path, 'checkpoint_%d.pth' % (init_epoch - 1))
                checkpoint = torch.load(ckpt, map_location='cpu')
                mvtn_model.load_state_dict(checkpoint['mvtn_state_dict'])

            if args.abo:
                mvtn_dataset_loader = ABODataLoader(args).get_dataset(
                    DatasetType.TRAIN, args.n_views, train_transforms, n_samples=10)
            else:
                mvtn_dataset_loader = ShapeNetDataLoader(args, rotation=False).get_dataset(
                    DatasetType.TRAIN, args.n_views, train_transforms, n_samples=10)
            mvtn_dataset_sampler = torch.utils.data.distributed.DistributedSampler(mvtn_dataset_loader, shuffle=True)
            mvtn_batch_sampler = torch.utils.data.BatchSampler(mvtn_dataset_sampler, args.train_batch_size,
                                                               drop_last=True)
            mvtn_data_loader = torch.utils.data.DataLoader(dataset=mvtn_dataset_loader,
                                                           num_workers=cfg.CONST.NUM_WORKER,
                                                           pin_memory=True,
                                                           batch_sampler=mvtn_batch_sampler)

        if args.AT_type == 'ours':
            if args.pix3d:
                ours_samples_batchsize = 10
                ours_dataset_loader = ShapeNetDataLoader(args, rotation=False, pix3d=True).get_dataset(
                    DatasetType.TRAIN, args.n_views, train_transforms, n_samples=10)
            elif args.abo:
                ours_samples_batchsize = 60
                ours_dataset_loader = ABODataLoader(args).get_dataset(
                    DatasetType.TRAIN, args.n_views, train_transforms)
            else:
                ours_samples_batchsize = 130
                ours_dataset_loader = ShapeNetDataLoader(args, rotation=False).get_dataset(
                    DatasetType.TRAIN, args.n_views, train_transforms, n_samples=60)
            ours_dataset_sampler = torch.utils.data.distributed.DistributedSampler(ours_dataset_loader, shuffle=True)
            ours_batch_sampler = torch.utils.data.BatchSampler(ours_dataset_sampler, ours_samples_batchsize,
                                                               drop_last=True)

            ours_data_loader = torch.utils.data.DataLoader(dataset=ours_dataset_loader,
                                                           num_workers=cfg.CONST.NUM_WORKER,
                                                           pin_memory=True,
                                                           batch_sampler=ours_batch_sampler)

        for epoch_idx in range(init_epoch, args.epochs, args.epochs_fintune):

            train_dataset_sampler.set_epoch(epoch_idx)

            # evaluation on natural examples
            # print('================================================================')
            #best_iou = eval_test(model, val_data_loader, epoch_idx, best_iou, writer=val_writer)

            if args.AT_type == 'VIAT':
                dist_pool_mu_path = os.path.join(args.dist_pool_path, 'dist_pool_mu_%s.npy' % args.attack_model)
                dist_pool_sigma_path = os.path.join(args.dist_pool_path, 'dist_pool_sigma_%s.npy' % args.attack_model)
                # warm_start if epoch>1
                if epoch_idx == 0:
                    # attack to generate dist_pool first
                    NES_GMM_search(args, model, dist_pool_mu=None, dist_pool_sigma=None, mood='init')
                    dist_pool_mu = np.load(dist_pool_mu_path)
                    dist_pool_sigma = np.load(dist_pool_sigma_path)

                else:
                    # load dist_pool first
                    dist_pool_mu = np.load(dist_pool_mu_path)
                    dist_pool_sigma = np.load(dist_pool_sigma_path)

                    NES_GMM_search(args, model, dist_pool_mu=dist_pool_mu, dist_pool_sigma=dist_pool_sigma,
                                   mood='warm_start')
                    dist_pool_mu = np.load(dist_pool_mu_path)
                    dist_pool_sigma = np.load(dist_pool_sigma_path)

                model, optimizer, scheduler = train(args, model, train_data_loader, optimizer, scheduler,
                                                    epoch_idx, dist_pool_mu=dist_pool_mu,
                                                    dist_pool_sigma=dist_pool_sigma, writer=train_writer, transforms=train_transforms)

            elif args.AT_type == 'random' or args.AT_type == 'nature':

                model, optimizer, scheduler = train(args, model, train_data_loader, optimizer, scheduler,
                                                    epoch_idx, writer=train_writer, transforms=train_transforms)

            elif args.AT_type == 'MVTN':
                mvtn_dataset_sampler.set_epoch(epoch_idx)

                mvtn_model = mvtn_search(args, model, mvtn_model, mvtn_data_loader, epoch_idx)

                model, optimizer, scheduler = train(args, model, train_data_loader, optimizer, scheduler,
                                                    epoch_idx, mvtn_model=mvtn_model, writer=train_writer, transforms=train_transforms)

            elif args.AT_type == 'ours':
                ours_dataset_sampler.set_epoch(epoch_idx)
                rot_path = args.rotation_pool_path % (args.interval_degree, args.attack_model)
                if args.pix3d:
                    rot_path = rot_path + '_pix3d'
                if not os.path.exists(rot_path):
                    os.makedirs(rot_path)
                rotation_path = os.path.join(rot_path, 'rotation_pool_%d.json' % dist.get_rank())
                if epoch_idx == init_epoch:
                    rot_data = None
                    rotation_pool = None
                    mood = 'init'
                    ours_search(args, model, ours_data_loader, rot_data, mood)
                    #ours_wo_pool_search(args, model, ours_data_loader, rot_data, mood, rotation_pool)
                else:
                    mood = 'warm_start'
                    ours_search(args, model, ours_data_loader, rot_data, mood)
                    #ours_wo_pool_search(args, model, ours_data_loader, rot_data, mood, rotation_pool)

                with open(rotation_path, 'r') as f:
                    rotation_pool = json.load(f)
                model, optimizer, scheduler, data = train(args, model, train_data_loader, optimizer, scheduler,
                                                    epoch_idx, rotation_pool=rotation_pool, writer=train_writer, transforms=train_transforms)
                rot_data = data




if __name__ == '__main__':
    main()