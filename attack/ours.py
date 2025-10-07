import numpy as np
import os

import torch
from tqdm import tqdm
import json
import torch.distributed as dist
import torch.nn.functional as F


def ours_search(args, model, dataloader, rot_data=None, mood='init'):
    device = torch.cuda.current_device()
    rotation_pool = {}

    interval_degree = args.interval_degree
    elevations_num_intervals = int(180 / interval_degree)
    azimuths_num_intervals = int(360 / interval_degree)

    elevations = np.linspace(-90.0, 90.0, elevations_num_intervals + 1)[:-1]
    azimuths = np.linspace(-180.0, 180.0, azimuths_num_intervals + 1)[:-1]

    model['encoder'].eval()
    model['decoder'].eval()
    if args.attack_model == 'pix2vox':
        model['refiner'].eval()
    model['merger'].eval()

    with torch.no_grad():

        if mood == 'warm_start':
            for temp in tqdm(range(1), total=1, desc='Searching for viewpoints for novel dataset'):
                taxonomy_names = []
                for id in range(len(rot_data['taxonomy_names'])):
                    taxonomy_names.extend(rot_data['taxonomy_names'][id])
                data = torch.cat(rot_data['data'], dim=0)
                target = torch.cat(rot_data['target'], dim=0)
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                voxel_size = target.size(-1)

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
                res = torch.abs(torch.sub(target, generated_volumes))
                mask_voxel = target * res
                voxel = mask_voxel.cpu().detach().unsqueeze(dim=1)

                total_sum = torch.zeros(batch_size, len(elevations) * len(azimuths))
                for elevation in elevations:
                    for azimuth in azimuths:
                        # 定义俯仰角和方位角（以弧度为单位）
                        elevation_angle = torch.deg2rad(torch.tensor(elevation + interval_degree / 2))
                        azimuth_angle = torch.deg2rad(torch.tensor(azimuth + interval_degree / 2))
                        # 定义旋转角以弧度为单位）
                        rotation_angle_y = torch.deg2rad(torch.tensor(0))
                        rotation_angle_z = torch.deg2rad(torch.tensor(90))

                        # 生成旋转矩阵
                        rotation_matrix_y = torch.tensor([[torch.cos(rotation_angle_y), 0, torch.sin(rotation_angle_y)],
                                                          [0, 1, 0],
                                                          [-torch.sin(rotation_angle_y), 0,
                                                           torch.cos(rotation_angle_y)]], dtype=torch.double)
                        rotation_matrix_z = torch.tensor(
                            [[torch.cos(rotation_angle_z), -torch.sin(rotation_angle_z), 0],
                             [torch.sin(rotation_angle_z), torch.cos(rotation_angle_z), 0],
                             [0, 0, 1]], dtype=torch.double)
                        rotation_matrix = torch.matmul(rotation_matrix_z, rotation_matrix_y)

                        elevation_matrix_angle = torch.tensor([[1, 0, 0],
                                                               [0, torch.cos(elevation_angle),
                                                                -torch.sin(elevation_angle)],
                                                               [0, torch.sin(elevation_angle),
                                                                torch.cos(elevation_angle)]], dtype=torch.double)
                        azimuth_matrix_angle = torch.tensor([[torch.cos(azimuth_angle), 0, torch.sin(azimuth_angle)],
                                                             [0, 1, 0],
                                                             [-torch.sin(azimuth_angle), 0, torch.cos(azimuth_angle)]],
                                                            dtype=torch.double)
                        view_rot_matrix = torch.matmul(azimuth_matrix_angle, elevation_matrix_angle)

                        rotation_matrix = torch.matmul(view_rot_matrix, rotation_matrix).unsqueeze(dim=0).expand(
                            batch_size, -1, -1)

                        # 将体素转换为二维平面上的投影
                        zero_column = torch.zeros(batch_size, 3, 1)
                        rotated_matrix = torch.cat([rotation_matrix, zero_column], dim=2)
                        grid = F.affine_grid(rotated_matrix, voxel.size(), align_corners=True)
                        rotation_voxel = F.grid_sample(voxel, grid.float(), align_corners=True)
                        rotation_voxel = rotation_voxel.squeeze(dim=1)

                        # 投影
                        projected_voxel = torch.zeros(batch_size, voxel_size, voxel_size)
                        for batch_i in range(batch_size):
                            for y in range(voxel_size):
                                for z in range(voxel_size):
                                    if_nonzero = (rotation_voxel[batch_i, :, y, z] > 0).any().item()
                                    if if_nonzero:
                                        positive_indices = torch.nonzero(
                                            rotation_voxel[batch_i, :, y, z] > 0).squeeze()  # 获取大于0值的索引
                                        max_positive_index = torch.max(positive_indices)  # 获取具有最大索引的值的索引
                                        projected_voxel[batch_i, y, z] = rotation_voxel[batch_i, :, y, z][
                                            max_positive_index]  # 获取具有最大索引的值

                        # 将体素转换为二维平面上的投影
                        sum = torch.sum(projected_voxel, dim=(1, 2), dtype=torch.float64)
                        non_zero_counts = torch.sum(projected_voxel != 0, dim=(1, 2))
                        averages = sum / non_zero_counts
                        total_sum[:,
                        np.where(elevations == elevation)[0][0] * len(azimuths) + np.where(azimuths == azimuth)[0][
                            0]] = averages

                        '''if dist.get_rank() == 0:
                            print(sample_names[0])

                            fig = plt.figure()
                            projection = torch.rot90(projected_voxel[0], 1, (0, 1))
                            plt.imshow(projection, cmap='gray')
                            plt.axis('off')
                            plt.savefig(
                                os.path.join('/home/ubuntu/zhouhang/zero123-main/zero123/attack/rotation_pool',
                                             'projection_%d_%d.png' % (elevation, azimuth)))
                            x = rotation_voxel[0].__ge__(0.01).float()
                            fig = plt.figure()
                            ax = fig.add_axes(Axes3D(fig))
                            ax.set_aspect('auto')
                            ax.voxels(x, edgecolor="k", facecolors='r')
                            plt.tight_layout()
                            plt.savefig(os.path.join('/home/ubuntu/zhouhang/zero123-main/zero123/attack/rotation_pool',
                                'voxel_%d_%d.png' % (elevation, azimuth)))

                            print('elevation: %d, azimuth: %d, sum: %f, non_zero_count: %f, average: %f' % (elevation, azimuth, sum[0], non_zero_counts[0], averages[0]))
                        torch.distributed.barrier()'''

                index_sort = torch.argsort(total_sum, dim=1, descending=True)
                for batch_id in range(batch_size):
                    trans = np.zeros((args.n_views, 2))
                    for i_view in range(args.n_views):
                        elevation_id = int(index_sort[batch_id][i_view] / len(azimuths))
                        azimuth_id = int(index_sort[batch_id][i_view] % len(azimuths))
                        trans[i_view][0] = np.random.normal(
                            loc=elevations[elevation_id] + interval_degree / 2,
                            scale=interval_degree / 6)
                        trans[i_view][1] = np.random.normal(
                            loc=azimuths[azimuth_id] + interval_degree / 2,
                            scale=interval_degree / 6)

                    trans = trans.tolist()

                    if taxonomy_names[batch_id] in rotation_pool:
                        rotation_pool[taxonomy_names[batch_id]].append(trans)
                    else:
                        rotation_pool[taxonomy_names[batch_id]] = [trans]

        for_tqdm = tqdm(enumerate(dataloader), total=len(dataloader), desc='Searching for viewpoints for origin dataset')
        for batch_idx, (taxonomy_names, sample_names, data,
                        target) in for_tqdm:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            voxel_size = target.size(-1)

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
            res = torch.abs(torch.sub(target, generated_volumes))
            mask_voxel = target * res
            voxel = mask_voxel.cpu().detach().unsqueeze(dim=1)

            total_sum = torch.zeros(batch_size, len(elevations) * len(azimuths))
            for elevation in elevations:
                for azimuth in azimuths:
                    # 定义俯仰角和方位角（以弧度为单位）
                    elevation_angle = torch.deg2rad(torch.tensor(elevation + interval_degree / 2))
                    azimuth_angle = torch.deg2rad(torch.tensor(azimuth + interval_degree / 2))
                    # 定义旋转角以弧度为单位）
                    rotation_angle_y = torch.deg2rad(torch.tensor(0))
                    rotation_angle_z = torch.deg2rad(torch.tensor(90))

                    # 生成旋转矩阵
                    rotation_matrix_y = torch.tensor([[torch.cos(rotation_angle_y), 0, torch.sin(rotation_angle_y)],
                                                      [0, 1, 0],
                                                      [-torch.sin(rotation_angle_y), 0, torch.cos(rotation_angle_y)]], dtype=torch.double)
                    rotation_matrix_z = torch.tensor([[torch.cos(rotation_angle_z), -torch.sin(rotation_angle_z), 0],
                                                      [torch.sin(rotation_angle_z), torch.cos(rotation_angle_z), 0],
                                                      [0, 0, 1]], dtype=torch.double)
                    rotation_matrix = torch.matmul(rotation_matrix_z, rotation_matrix_y)

                    elevation_matrix_angle = torch.tensor([[1, 0, 0],
                                                           [0, torch.cos(elevation_angle), -torch.sin(elevation_angle)],
                                                           [0, torch.sin(elevation_angle), torch.cos(elevation_angle)]], dtype=torch.double)
                    azimuth_matrix_angle = torch.tensor([[torch.cos(azimuth_angle), 0, torch.sin(azimuth_angle)],
                                                         [0, 1, 0],
                                                         [-torch.sin(azimuth_angle), 0, torch.cos(azimuth_angle)]], dtype=torch.double)
                    view_rot_matrix = torch.matmul(azimuth_matrix_angle, elevation_matrix_angle)

                    rotation_matrix = torch.matmul(view_rot_matrix, rotation_matrix).unsqueeze(dim=0).expand(batch_size, -1, -1)

                    # 将体素转换为二维平面上的投影
                    zero_column = torch.zeros(batch_size, 3, 1)
                    rotated_matrix = torch.cat([rotation_matrix, zero_column], dim=2)
                    grid = F.affine_grid(rotated_matrix, voxel.size(), align_corners=True)
                    rotation_voxel = F.grid_sample(voxel, grid.float(), align_corners=True)
                    rotation_voxel = rotation_voxel.squeeze(dim=1)

                    #投影
                    projected_voxel = torch.zeros(batch_size, voxel_size, voxel_size)
                    for batch_i in range(batch_size):
                        for y in range(voxel_size):
                            for z in range(voxel_size):
                                if_nonzero = (rotation_voxel[batch_i, :, y, z] > 0).any().item()
                                if if_nonzero:
                                    positive_indices = torch.nonzero(rotation_voxel[batch_i, :, y, z] > 0).squeeze()  # 获取大于0值的索引
                                    max_positive_index = torch.max(positive_indices)  # 获取具有最大索引的值的索引
                                    projected_voxel[batch_i, y, z] = rotation_voxel[batch_i, :, y, z][max_positive_index]  # 获取具有最大索引的值

                    # 将体素转换为二维平面上的投影
                    sum = torch.sum(projected_voxel, dim=(1, 2), dtype=torch.float64)
                    non_zero_counts = torch.sum(projected_voxel != 0, dim=(1, 2))
                    averages = sum / non_zero_counts
                    total_sum[:, np.where(elevations == elevation)[0][0] * len(azimuths) + np.where(azimuths == azimuth)[0][0]] = averages

                    '''if dist.get_rank() == 0:
                        print(sample_names[0])

                        fig = plt.figure()
                        projection = torch.rot90(projected_voxel[0], 1, (0, 1))
                        plt.imshow(projection, cmap='gray')
                        plt.axis('off')
                        plt.savefig(
                            os.path.join('/home/ubuntu/zhouhang/zero123-main/zero123/attack/rotation_pool',
                                         'projection_%d_%d.png' % (elevation, azimuth)), bbox_inches='tight', pad_inches=0)
                        x = rotation_voxel[0].__ge__(0.01).float()
                        fig = plt.figure()
                        ax = fig.add_axes(Axes3D(fig))
                        ax.set_aspect('auto')
                        ax.voxels(x, edgecolor="k", facecolors='r')
                        plt.tight_layout()
                        plt.savefig(os.path.join('/home/ubuntu/zhouhang/zero123-main/zero123/attack/rotation_pool',
                            'voxel_%d_%d.png' % (elevation, azimuth)))

                        print('elevation: %d, azimuth: %d, sum: %f, non_zero_count: %f, average: %f' % (elevation, azimuth, sum[0], non_zero_counts[0], averages[0]))
                    torch.distributed.barrier()'''

            index_sort = torch.argsort(total_sum, dim=1, descending=True)
            for batch_id in range(batch_size):
                trans = np.zeros((args.n_views, 2))
                for i_view in range(args.n_views):
                    elevation_id = int(index_sort[batch_id][i_view] / len(azimuths))
                    azimuth_id = int(index_sort[batch_id][i_view] % len(azimuths))
                    trans[i_view][0] = np.random.normal(
                        loc=elevations[elevation_id] + interval_degree / 2,
                        scale=interval_degree / 6)
                    trans[i_view][1] = np.random.normal(
                        loc=azimuths[azimuth_id] + interval_degree / 2,
                        scale=interval_degree / 6)

                trans = trans.tolist()

                if taxonomy_names[batch_id] in rotation_pool:
                    rotation_pool[taxonomy_names[batch_id]].append(trans)
                else:
                    rotation_pool[taxonomy_names[batch_id]] = [trans]

        rot_path = args.rotation_pool_path % (args.interval_degree, args.attack_model)
        if args.pix3d:
            rot_path = rot_path + '_pix3d'
        rotation_path = os.path.join(rot_path, 'rotation_pool_%d.json' % dist.get_rank())
        if torch.distributed.get_rank() == 0:
            if not os.path.exists(rot_path):
                os.makedirs(rot_path)
        torch.cuda.synchronize(torch.device(torch.cuda.current_device()))
        with open(rotation_path, 'w') as f:
            json.dump(rotation_pool, f, indent=4, ensure_ascii=False)

    torch.distributed.barrier()

    return

