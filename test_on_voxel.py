import torch
from tqdm import tqdm
import numpy as np
import json
import os

from pix2vox.config import cfg as cfg_pix2vox
from pix2vox.models.encoder import Encoder as Encoder_pix2vox
from pix2vox.models.decoder import Decoder as Decoder_pix2vox
from pix2vox.models.refiner import Refiner as Refiner_pix2vox
from pix2vox.models.merger import Merger as Merger_pix2vox

from lrgt.config import cfg as cfg_lrgt
from lrgt.models.encoder.encoder import Encoder as Encoder_lrgt
from lrgt.models.merger.merger import Merger as Merger_lrgt
from lrgt.models.decoder.decoder import Decoder as Decoder_lrgt
def test_net(args, data_loader):
    ckpt = args.test_ckpt

    if args.pix3d:
        with open(args.pix3d_taxonomy_path, encoding='utf-8') as file:
            taxonomies = json.loads(file.read())
    elif args.abo:
        with open(args.abo_taxonomy_path, encoding='utf-8') as file:
            taxonomies = json.loads(file.read())
    else:
        with open(args.taxonomy_path, encoding='utf-8') as file:
            taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    if args.attack_model == 'pix2vox':
        encoder = Encoder_pix2vox(cfg_pix2vox)
        decoder = Decoder_pix2vox(cfg_pix2vox)
        refiner = Refiner_pix2vox(cfg_pix2vox)
        merger = Merger_pix2vox(cfg_pix2vox)

        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
        refiner = torch.nn.SyncBatchNorm.convert_sync_batchnorm(refiner)
        merger = torch.nn.SyncBatchNorm.convert_sync_batchnorm(merger)

        device = torch.cuda.current_device()

        encoder = torch.nn.parallel.DistributedDataParallel(encoder.to(device), find_unused_parameters=True, device_ids=[device], output_device=device)
        decoder = torch.nn.parallel.DistributedDataParallel(decoder.to(device), find_unused_parameters=True, device_ids=[device], output_device=device)
        refiner = torch.nn.parallel.DistributedDataParallel(refiner.to(device), find_unused_parameters=True, device_ids=[device], output_device=device)
        merger = torch.nn.parallel.DistributedDataParallel(merger.to(device), find_unused_parameters=True, device_ids=[device], output_device=device)

        print('Loading weights from %s ...' % (ckpt))
        checkpoint = torch.load(ckpt, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
        merger.load_state_dict(checkpoint['merger_state_dict'])

    elif args.attack_model == 'lrgt':
        encoder = Encoder_lrgt(cfg_lrgt)
        decoder = Decoder_lrgt(cfg_lrgt)
        merger = Merger_lrgt(cfg_lrgt)

        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
        merger = torch.nn.SyncBatchNorm.convert_sync_batchnorm(merger)

        device = torch.cuda.current_device()

        encoder = torch.nn.parallel.DistributedDataParallel(encoder.cuda(), device_ids=[device], output_device=device)
        decoder = torch.nn.parallel.DistributedDataParallel(decoder.cuda(), device_ids=[device], output_device=device)
        merger = torch.nn.parallel.DistributedDataParallel(merger.cuda(), device_ids=[device], output_device=device)

        print('Loading weights from %s ...' % (ckpt))
        checkpoint = torch.load(ckpt, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        merger.load_state_dict(checkpoint['merger_state_dict'])

    encoder.eval()
    decoder.eval()
    merger.eval()
    if args.attack_model == 'pix2vox':
        refiner.eval()

    n_samples = len(data_loader)
    test_iou = []
    taxonomies_list = []

    for_tqdm = tqdm(enumerate(data_loader), total=n_samples)
    for batch_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in for_tqdm:
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        rendering_images, ground_truth_volume = rendering_images.to(device), ground_truth_volume.to(device)

        with torch.no_grad():

            if args.attack_model == 'pix2vox':
                image_features = encoder(rendering_images)
                raw_features, generated_volume = decoder(image_features)
                generated_volume = merger(raw_features, generated_volume)
                generated_volume = refiner(generated_volume)
            elif args.attack_model == 'lrgt':
                image_features = encoder(rendering_images)
                context = merger(image_features)
                generated_volume = decoder(context).squeeze(dim=1)
                generated_volume = generated_volume.clamp_max(1)

            sample_iou = []
            for th in cfg_pix2vox.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).unsqueeze(dim=0))
            test_iou.append(torch.cat(sample_iou).unsqueeze(dim=0))
            taxonomies_list.append(torch.tensor(list(taxonomies.keys()).index(taxonomy_id)).unsqueeze(dim=0))

    test_iou = torch.cat(test_iou, dim=0)
    taxonomies_list = torch.cat(taxonomies_list).to(torch.cuda.current_device())

    world_size = int(os.environ['WORLD_SIZE'])
    all_test_iou = [torch.zeros_like(test_iou) for _ in range(world_size)]
    all_taxonomies_list = [torch.zeros_like(taxonomies_list) for _ in range(world_size)]
    torch.distributed.all_gather(all_test_iou, test_iou)
    torch.distributed.all_gather(all_taxonomies_list, taxonomies_list)
    if torch.distributed.get_rank() == 0:
        redundancy = n_samples % world_size
        if redundancy == 0:
            redundancy = world_size
        for i in range(world_size):
            all_test_iou[i] = all_test_iou[i] \
                if i < redundancy else all_test_iou[i][:-1, :]
            all_taxonomies_list[i] = all_taxonomies_list[i] \
                if i < redundancy else all_taxonomies_list[i][:-1]
        all_test_iou = torch.cat(all_test_iou, dim=0).cpu().numpy()  # [sample_num, 4]
        all_taxonomies_list = torch.cat(all_taxonomies_list).cpu().numpy()  # [sample_num]
        test_iou = dict()
        for taxonomy_id, sample_iou in zip(all_taxonomies_list, all_test_iou):
            if list(taxonomies.keys())[taxonomy_id] not in test_iou.keys():
                test_iou[list(taxonomies.keys())[taxonomy_id]] = {'n_samples': 0, 'iou': []}
            test_iou[list(taxonomies.keys())[taxonomy_id]]['n_samples'] += 1
            test_iou[list(taxonomies.keys())[taxonomy_id]]['iou'].append(sample_iou)

    torch.cuda.synchronize(torch.device(torch.cuda.current_device()))

    if torch.distributed.get_rank() == 0:
        # Output testing results
        mean_iou = []
        n_samples = 0
        for taxonomy_id in test_iou:
            test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
            mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
            n_samples += test_iou[taxonomy_id]['n_samples']
        mean_iou = np.sum(mean_iou, axis=0) / n_samples

        # Print header
        print('\n')
        print('============================ TEST RESULTS ============================')
        print('Taxonomy', end='\t')
        print('#Sample', end='\t')
        print('Baseline', end='\t')
        for th in cfg_pix2vox.TEST.VOXEL_THRESH:
            print('t=%.2f' % th, end='\t')
        print()
        # Print body
        for taxonomy_id in test_iou:
            print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
            print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
            print('N/a', end='\t\t')

            for ti in test_iou[taxonomy_id]['iou']:
                print('%.4f' % ti, end='\t')
            print()

        # Print mean IoU for each threshold
        print('Overall ', end='\t\t\t\t')
        for mi in mean_iou:
            print('%.4f' % mi, end='\t')
        print('\n')

        # Add testing results to TensorBoard
        max_iou = np.max(mean_iou)

        print('The IoU score of %d-view-input is %.4f\n' % (args.n_views, max_iou))