import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist
from torchvision import transforms

import torch

from .evaluate_forAT import comput_fitness
from .solver import PEPG
from zero123.dataset.binvox_rw import read_as_3d_array
from zero123.diffusion_model import set_diffusion

np.set_printoptions(precision=4,  linewidth=100, suppress=True)

class batch_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, images, poses, labels):
        self.images = images
        self.poses = poses
        self.labels = labels

    def __len__(self):
        return self.images.size()[0]

    def __getitem__(self, item):
        return self.images[item], self.poses[item], self.labels[item]

def NES_GMM_search_step(args, model, image, target, taxonomy_names, sample_names, mood, diffusion, mu_start=None, sigma_start=None):

    iteration = args.iteration
    popsize = args.popsize
    num_paras = args.search_num
    num_k = args.num_k
    device = torch.cuda.current_device()

    batchsize = popsize-1 + num_k

    image = Image.open(image)
    image.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    image = image.resize([256, 256], Image.Resampling.LANCZOS)
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = transforms.ToTensor()(image).unsqueeze(0)
    image = image * 2 - 1
    image = transforms.functional.resize(image, [256, 256])
    image = image.expand(batchsize, args.n_views, -1, -1, -1)

    with open(target, 'rb') as f:
        target = read_as_3d_array(f)
        target = target.data.astype(np.float32)

    target = torch.from_numpy(target)
    target = target.expand(batchsize, -1, -1, -1)

    taxonomy_names = [taxonomy_names for _ in range(batchsize)]
    sample_names = [sample_names for _ in range(batchsize)]

    # 搜索三维空间，th phi gamma
    solver = PEPG(num_params=num_paras,  # number of model parameters
                  num_k = num_k,
                  sigma_init=0.1,  # initial standard deviation
                  sigma_update=True,  # 不大幅更新sigma
                  learning_rate=args.lr,  # learning rate for standard deviation
                  learning_rate_decay=0.99, # don't anneal the learning rate
                  learning_rate_limit=0,
                  popsize=popsize,  # population size
                  average_baseline=False,  # set baseline to average of batch
                  weight_decay=0.00,  # weight decay coefficient
                  rank_fitness=True,  # use rank rather than fitness numbers
                  forget_best=False,
                  mu_lambda=args.mu_lamba,
                  sigma_lambda=args.sigma_lamba,
                  omiga_lamba=args.omiga_lamba,
                  random_begin=args.random_begin,
                  omiga_alpha=0.02,
                  mood = mood,
                  mu_start=mu_start,
                  sigma_start=sigma_start
                  )
    history = []
    fitness_origin = []
    for j in range(iteration):
        solutions = solver.ask()
        mu_entropy_grad, sigma_entropy_grad, omiga_entropy_grad = solver.comput_entropy()

        # gamma (-90,90)
        solutions[:, 0] = 90 * np.tanh(solutions[:, 0])
        # th (-180,180)
        solutions[:, 1] = 180 * np.tanh(solutions[:, 1])

        #  多进程工作
        # with joblib.Parallel(n_jobs=N_JOBS) as parallel:
        #   #for i in tqdm(range(solver.popsize)):
        #     #fitness_list[i] = comput_fitness(solutions[i])

        #   fitness_list = parallel(joblib.delayed(comput_fitness)(solutions[i], solver.sigma) for i in tqdm(range(solver.batch_size*2+solver.num_k)))

        fitness_list = comput_fitness(args, model, image, target, taxonomy_names, sample_names, solutions, diffusion)

        solver.tell(fitness_list, mu_entropy_grad, sigma_entropy_grad, omiga_entropy_grad)
        result = solver.result()  # first element is the best solution, second element is the best fitness

        fitness_origin.append(np.max(fitness_list))
        history.append(result[1])
        average_fitness = np.mean(fitness_list)

    result = torch.tensor(np.array(result)).to(device)
    dist.broadcast(result, src=0)
    mu = result[0]
    sigma = result[3]
    Entropy = solver.entropy

    return mu.cpu().numpy(), sigma.cpu().numpy(), Entropy

class GMFool:

    def __init__(self, args, dist_pool_mu=None, dist_pool_sigma=None, mood='init'):

        self.img_path = args.trainset_path.split('%s')[0]
        self.vox_path = args.voxel_path.split('%s')[0]
        all_class = os.listdir(self.img_path)
        all_class.sort()
        class_num = len(all_class)
        self.class_num = class_num
        self.all_class = all_class
        self.num_para = args.search_num

        self.dist_pool_mu = dist_pool_mu
        self.dist_pool_sigma = dist_pool_sigma


    def step(self, args, class_id, model, mood, diffusion):
        mu_result = np.zeros([args.num_k, self.num_para])
        sigma_result = np.zeros([args.num_k, self.num_para])
        Entropy_class = 0

        all_object = os.listdir(self.img_path+ '/' + self.all_class[class_id] + '/')
        all_object.sort()

        object_id = int(np.random.randint(0, len(all_object), size=1))

        image = self.img_path + self.all_class[class_id] + '/' + all_object[object_id] + '/06.png'
        target = self.vox_path + self.all_class[class_id] + '/' + all_object[object_id] + '/model.binvox'

        if mood == 'init':
            mu, sigma, Entropy = NES_GMM_search_step(args, model, image, target, self.all_class[class_id], all_object[object_id], mood, diffusion)
            Entropy_class += Entropy

            mu_result[:, :] = mu
            sigma_result[:, :] = sigma

        elif mood == 'warm_start':

            mu_start = self.dist_pool_mu[class_id, :, :]
            sigma_start = self.dist_pool_sigma[class_id, :, :]

            mu, sigma, Entropy = NES_GMM_search_step(args, model, image, target, self.all_class[class_id], all_object[object_id], mood, diffusion, mu_start, sigma_start)

            Entropy_class += Entropy

            mu_result[:, :] = mu
            sigma_result[:, :] = sigma

        return mu_result, sigma_result, Entropy_class


def NES_GMM_search(args, model, dist_pool_mu, dist_pool_sigma, mood):

    GMFool_solver = GMFool(args, dist_pool_mu=dist_pool_mu, dist_pool_sigma=dist_pool_sigma, mood=mood)
    diffusion = set_diffusion()
    diffusion.eval()

    if mood == 'init':
        dist_pool_mu_return = np.zeros([GMFool_solver.class_num, args.num_k, args.search_num])
        dist_pool_sigma_return = np.zeros([GMFool_solver.class_num, args.num_k, args.search_num])
    elif mood == 'warm_start':
        dist_pool_mu_return = dist_pool_mu
        dist_pool_sigma_return = dist_pool_sigma

    #-----------------------------------串行-------------------------------------------------------------
    Entropy_classes = []
    for class_id in tqdm(range(GMFool_solver.class_num)):
        mu_result, sigma_result, Entropy_class = GMFool_solver.step(args, class_id, model, mood, diffusion)
        Entropy_classes.append(Entropy_class)
        dist_pool_mu_return[class_id, :, :] = mu_result
        dist_pool_sigma_return[class_id, :, :] = sigma_result

    #-----------------------------------------------------------------------------------------------------------
    if dist.get_rank() == 0:
        average_entropy = sum(Entropy_classes) / len(Entropy_classes)
        print("average_entropy:", average_entropy)

        if(os.path.exists(args.dist_pool_path) == False):
            os.makedirs(args.dist_pool_path)
        np.save(os.path.join(args.dist_pool_path, 'dist_pool_mu_%s.npy' % args.attack_model), dist_pool_mu_return)
        np.save(os.path.join(args.dist_pool_path, 'dist_pool_sigma_%s.npy' % args.attack_model), dist_pool_sigma_return)

    torch.distributed.barrier()

    return

