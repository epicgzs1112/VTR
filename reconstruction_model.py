import torch

from pix2vox.config import cfg as cfg_pix2vox
from pix2vox.models.encoder import Encoder as Encoder_pix2vox
from pix2vox.models.decoder import Decoder as Decoder_pix2vox
from pix2vox.models.refiner import Refiner
from pix2vox.models.merger import Merger as Merger_pix2vox

from lrgt.config import cfg as cfg_lrgt
from lrgt.models.encoder.encoder import Encoder as Encoder_lrgt
from lrgt.models.merger.merger import Merger as Merger_lrgt
from lrgt.models.decoder.decoder import Decoder as Decoder_lrgt



def lrgt(args, ckpt):
    model = dict()
    model['encoder'] = Encoder_lrgt(cfg_lrgt)
    model['decoder'] = Decoder_lrgt(cfg_lrgt)
    model['merger'] = Merger_lrgt(cfg_lrgt)

    model['encoder'] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model['encoder'])
    model['decoder'] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model['decoder'])
    model['merger'] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model['merger'])

    device = torch.cuda.current_device()

    model['encoder'] = torch.nn.parallel.DistributedDataParallel(model['encoder'].to(device),
                                                                find_unused_parameters=True, device_ids=[device],
                                                                 output_device=device)
    model['decoder'] = torch.nn.parallel.DistributedDataParallel(model['decoder'].to(device),device_ids=[device],
                                                                 output_device=device)
    model['merger'] = torch.nn.parallel.DistributedDataParallel(model['merger'].to(device),device_ids=[device],
                                                                output_device=device)

    torch.distributed.barrier()

    print('Loading weights from %s ...' % (ckpt))
    checkpoint = torch.load(ckpt, map_location='cpu')
    model['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
    model['decoder'].load_state_dict(checkpoint['decoder_state_dict'])
    model['merger'].load_state_dict(checkpoint['merger_state_dict'])

    return model

def opt_lrgt(args, model):
    optimizer = dict()
    scheduler = dict()
    optimizer['encoder'] = torch.optim.Adam(model['encoder'].parameters(),
                                            lr=cfg_lrgt.TRAIN.ENCODER_LEARNING_RATE,
                                            betas=cfg_lrgt.TRAIN.BETAS)
    optimizer['decoder'] = torch.optim.Adam(model['decoder'].parameters(),
                                            lr=cfg_lrgt.TRAIN.DECODER_LEARNING_RATE,
                                            betas=cfg_lrgt.TRAIN.BETAS)
    optimizer['merger'] = torch.optim.Adam(model['merger'].parameters(),
                                           lr=cfg_lrgt.TRAIN.MERGER_LEARNING_RATE,
                                           betas=cfg_lrgt.TRAIN.BETAS)

    # Set up learning rate scheduler to decay learning rates dynamically
    scheduler['encoder'] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer['encoder'], milestones=[lr for lr in cfg_lrgt.TRAIN.MILESTONESLR.ENCODER_LR_MILESTONES],
                gamma=cfg_lrgt.TRAIN.MILESTONESLR.GAMMA)
    scheduler['decoder'] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer['decoder'], milestones=[lr for lr in cfg_lrgt.TRAIN.MILESTONESLR.DECODER_LR_MILESTONES],
                gamma=cfg_lrgt.TRAIN.MILESTONESLR.GAMMA)
    scheduler['merger'] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer['merger'], milestones=[lr for lr in cfg_lrgt.TRAIN.MILESTONESLR.MERGER_LR_MILESTONES],
                gamma=cfg_lrgt.TRAIN.MILESTONESLR.GAMMA)
    return optimizer, scheduler

def pix2vox(args, ckpt):
    model = dict()
    model['encoder'] = Encoder_pix2vox(cfg_pix2vox)
    model['decoder'] = Decoder_pix2vox(cfg_pix2vox)
    model['refiner'] = Refiner(cfg_pix2vox)
    model['merger'] = Merger_pix2vox(cfg_pix2vox)

    model['encoder'] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model['encoder'])
    model['decoder'] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model['decoder'])
    model['refiner'] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model['refiner'])
    model['merger'] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model['merger'])

    device = torch.cuda.current_device()

    model['encoder'] = torch.nn.parallel.DistributedDataParallel(model['encoder'].to(device),
                                                                 find_unused_parameters=False, device_ids=[device], output_device=device)
    model['decoder'] = torch.nn.parallel.DistributedDataParallel(model['decoder'].to(device),
                                                                 find_unused_parameters=False, device_ids=[device], output_device=device)
    model['refiner'] = torch.nn.parallel.DistributedDataParallel(model['refiner'].to(device),
                                                                 find_unused_parameters=False, device_ids=[device], output_device=device)
    model['merger'] = torch.nn.parallel.DistributedDataParallel(model['merger'].to(device),
                                                                 find_unused_parameters=False, device_ids=[device], output_device=device)

    torch.distributed.barrier()

    print('Loading weights from %s ...' % (ckpt))
    checkpoint = torch.load(ckpt, map_location='cpu')
    model['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
    model['decoder'].load_state_dict(checkpoint['decoder_state_dict'])
    model['refiner'].load_state_dict(checkpoint['refiner_state_dict'])
    model['merger'].load_state_dict(checkpoint['merger_state_dict'])

    return model

def opt_pix2vox(args, model):
    optimizer = dict()
    scheduler = dict()

    optimizer['encoder'] = torch.optim.Adam(model['encoder'].parameters(),
                                      lr=cfg_pix2vox.TRAIN.ENCODER_LEARNING_RATE,
                                      betas=cfg_pix2vox.TRAIN.BETAS)
    optimizer['decoder'] = torch.optim.Adam(model['decoder'].parameters(),
                                      lr=cfg_pix2vox.TRAIN.DECODER_LEARNING_RATE,
                                      betas=cfg_pix2vox.TRAIN.BETAS)
    optimizer['merger'] = torch.optim.Adam(model['merger'].parameters(),
                                      lr=cfg_pix2vox.TRAIN.MERGER_LEARNING_RATE,
                                      betas=cfg_pix2vox.TRAIN.BETAS)

    optimizer['refiner'] = torch.optim.Adam(model['refiner'].parameters(),
                                            lr=cfg_pix2vox.TRAIN.REFINER_LEARNING_RATE,
                                            betas=cfg_pix2vox.TRAIN.BETAS)

    # Set up learning rate scheduler to decay learning rates dynamically
    scheduler['encoder'] = torch.optim.lr_scheduler.MultiStepLR(optimizer['encoder'],
                                                                milestones=cfg_pix2vox.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg_pix2vox.TRAIN.GAMMA)
    scheduler['decoder'] = torch.optim.lr_scheduler.MultiStepLR(optimizer['decoder'],
                                                                milestones=cfg_pix2vox.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=cfg_pix2vox.TRAIN.GAMMA)
    scheduler['refiner'] = torch.optim.lr_scheduler.MultiStepLR(optimizer['refiner'],
                                                                milestones=cfg_pix2vox.TRAIN.REFINER_LR_MILESTONES,
                                                                gamma=cfg_pix2vox.TRAIN.GAMMA)
    scheduler['merger'] = torch.optim.lr_scheduler.MultiStepLR(optimizer['merger'],
                                                               milestones=cfg_pix2vox.TRAIN.MERGER_LR_MILESTONES,
                                                               gamma=cfg_pix2vox.TRAIN.GAMMA)
    return optimizer, scheduler