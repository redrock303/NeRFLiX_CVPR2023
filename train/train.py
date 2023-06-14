import argparse
import os
import os.path as osp
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import sys


sys.path.append('/newdata/kunzhou/project/package')
sys.path.append('/newdata/kunzhou/project/package_3090')

from config import config
from utils import common, dataloader, solver, model_opr


from dataset.mix_dataset import MixDataset as TrainDataset
from dataset.vimeo import Vimeo_Synthetic as BaseDataset

from model import SimpleNet
from validate import validate
import torchvision

import glob

# torch.backends.cudnn.enabled = False

def init_dist(local_rank):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    print('local_rank',local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method='env://')
    dist.barrier()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    rank = 0
    # initialization
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        num_gpu = int(os.environ['WORLD_SIZE'])
        distributed = num_gpu > 1
    if distributed:
        rank = args.local_rank
        init_dist(rank)
    common.init_random_seed(config.DATASET.SEED)
    # if config.DATASET.JPEGCOMPRESS is not None:
    #     model_name = '_v_{}_jc[{}-{}]_sharp_{}'.format(config.model_version,config.DATASET.JPEGCOMPRESS[0],\
    #               config.DATASET.JPEGCOMPRESS[1],config.DATASET.LABEL_SHARP )
    # else:
        # model_name = '_v_{}_jc[{}-{}]_sharp_{}'.format(config.model_version,100,100,config.DATASET.LABEL_SHARP )
    model_name = config.model_version
    # set up dirs and log
    
    root_dir = config.SOLVER.ROOT_PATH


    log_dir = osp.join(root_dir, model_name)
    model_dir = osp.join(log_dir, 'models')
    solver_dir = osp.join(log_dir, 'solvers')
    if rank <= 0:
        common.mkdir(log_dir)
        
        
        common.mkdir(model_dir)
        common.mkdir(solver_dir)
        save_dir = osp.join(log_dir, 'saved_imgs')
        common.mkdir(save_dir)
        tb_dir = osp.join(log_dir, 'tb_log')
        tb_writer = SummaryWriter(tb_dir)
        common.setup_logger('base', log_dir, 'train', level=logging.INFO, screen=True, to_file=True)
        logger = logging.getLogger('base')

    

    
    
    train_dataset = TrainDataset(patch_width=config.DATASET.PATCH_WIDTH, patch_height=config.DATASET.PATCH_HEIGHT,rank=rank,nframes = config.DATASET.NFRAME)
    train_loader = dataloader.train_loader(train_dataset, config, rank=rank,is_dist=distributed)
    if rank <= 0:
        val_dataset = BaseDataset(split='val')
        val_loader = dataloader.val_loader(val_dataset, config, rank, 1)



    # model
    
    model =SimpleNet(nbr = config.DATASET.NFRAME-1)
    print("model have {:.3f}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000.0))

    if config.INIT_MODEL:
        model_opr.load_model(model, config.INIT_MODEL, strict=False, cpu=True)


    device = torch.device(config.MODEL.DEVICE)
    model.to(device)
    # sisr_net.to(device)
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],find_unused_parameters=True) # 
        # sisr_net = torch.nn.parallel.DistributedDataParallel(sisr_net, device_ids=[torch.cuda.current_device()],find_unused_parameters=True)


    # solvers
    optimizer = solver.make_optimizer(config, model)  # lr without X num_gpu
    lr_scheduler = solver.make_lr_scheduler(config, optimizer)
    iteration = 0



    for batch_data in train_loader:
        model.train()
        iteration = iteration + 1
        lr_img = batch_data[0].to(device)
        hr_img = batch_data[1].to(device)

        noise_prior = batch_data[2].to(device)
        
        loss_dict = model(lr_img,noise_prior,hr_img)

        total_loss = sum(loss for loss in loss_dict.values())


        # if float(total_loss.item()) < 0.3: # for strange loss here ...
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if rank <= 0:
            if iteration % config.LOG_PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                log_str = 'Iter: %d, LR: %.3e, ' % (iteration, optimizer.param_groups[0]['lr'])
                for key in loss_dict:
                    tb_writer.add_scalar(key, loss_dict[key].mean(), global_step=iteration)
                    log_str += key + ': %.4f, ' % float(loss_dict[key])
                logger.info(log_str)

            if iteration % config.SAVE_PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                logger.info('[Saving] Iter: %d' % iteration)
                # model_path = osp.join(model_dir, 'latest_ft_rf.pth')
                # # solver_path = osp.join(solver_dir, '%d.solver' % iteration)
                # model_opr.save_model(model, model_path)
                
                model_path = osp.join(model_dir, 'iteration{}.pth'.format(iteration))
                model_opr.save_model(model, model_path)
                # model_opr.save_solver(optimizer, lr_scheduler, iteration, solver_path)

            if iteration % config.VAL.PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                logger.info('[Validating] Iter: %d' % iteration)
                model.eval()
                psnr, ssim = validate(model, val_loader, device, iteration, down=4,sisr_net=None, save_path=save_dir, save_img=True,
                                      max_num=5)
                logger.info('[Val Result] Iter: %d, PSNR: %.4f, SSIM: %.4f' % (iteration, psnr, ssim))

            if iteration >= config.SOLVER.MAX_ITER:
                logger.info('Finish training process!')
                break


if __name__ == '__main__':
    main()
