import torch
from torch.utils.data import DataLoader
import timm
from tensorboardX import SummaryWriter
# from models.vmunet.vmunet import VMUNet
from vmunet_v2 import VMUNetV2

from datetime import datetime
from engine_vmunet import *
import os
import sys
from utils_vmunet import *
from hydra.utils import instantiate, call

import warnings
warnings.filterwarnings("ignore")

import argparse

def training_preprocess(config, cid):
    print(torch.cuda.is_available())
    print('#----------Creating logger----------#')
    os.chdir('..') #change to the parent path 

    work_dir = 'drive/MyDrive/kvasir/results/' + 'vmunet-v2' + '_' + 'kvasir' + str(cid) + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'


    sys.path.append(work_dir + '/')
    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs) 

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    # print('#----------Preparing dataset----------#')
    # # train_dataset = NPY_datasets(config.data_path, config, train=True)
    # train_dataset = Polyp_datasets(parser.data_path, config, train=True)
    
    # train_loader = DataLoader(train_dataset,
    #                             batch_size=config.batch_size, 
    #                             shuffle=True,
    #                             pin_memory=True,
    #                             num_workers=config.num_workers)
    
    
    # # val_dataset = NPY_datasets(parser.data_path, config, train=False)
    
    


    # val_dataset = Polyp_datasets(parser.data_path, config, train=False)
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=1,
    #                         shuffle=False,
    #                         pin_memory=True, 
    #                         num_workers=config.num_workers,
    #                         drop_last=True)
        




    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet-v2':
        # model = VMUNetV2(
        #     num_classes=model_cfg['num_classes'],
        #     input_channels=model_cfg['input_channels'],
        #     depths=model_cfg['depths'],
        #     depths_decoder=model_cfg['depths_decoder'],
        #     drop_path_rate=model_cfg['drop_path_rate'],
        #     #load_ckpt_path=model_cfg['load_ckpt_path'],
        #     load_ckpt_path=parser.pretrained_weight_path, # comment so we can use transfer learning
        #     deep_supervision = model_cfg['deep_supervision'],
        # )

        model = instantiate(model_cfg)

        # model.load_from()    
    else: raise Exception('network in not right!')

    device = 'cpu:0'
    print(torch.cuda.is_available())
    device = 'cuda:0'

    model = model.cuda()

    cal_params_flops(model, 256, logger)

    return model, resume_model, checkpoint_dir, logger, writer, work_dir    


def train_vmunet(model_cfg, train_loader, val_loader, model, epochs, checkpoint_dir, logger, writer, work_dir):

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = instantiate(model_cfg.criterion)
    optimizer = get_optimizer(model_cfg, model)
    scheduler = get_scheduler(model_cfg, optimizer)
    model = model



    # print('#----------Set other params----------#')
    # min_loss = 999
    # start_epoch = 1
    # min_epoch = 1


    # if os.path.exists(resume_model):
    #     print('#----------Resume Model and Other params----------#')
    #     checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     saved_epoch = checkpoint['epoch']
    #     start_epoch += saved_epoch
    #     min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

    #     log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
    #     print(log_info)
    #     logger.info(log_info)



    step = 0
    print('#----------Training----------#')
    for epoch in range(epochs):

    
        torch.cuda.empty_cache()
        step = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                step,
                logger,
                model_cfg,
                writer
        )

        loss_all = []
        #for name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            #val_loader_t = val_loader_dict[name]
            
        loss_t = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                model_cfg,
                val_data_name='Kvasir'
            )
        loss_all.append(loss_t)
            
        loss = np.mean(loss_all)
        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            print(f'this epoch save the model in best.pth with loss = {loss}')
            min_loss = loss
            min_epoch = epoch

        # torch.save(
        #     {
        #         'epoch': epoch,
        #         'min_loss': min_loss,
        #         'min_epoch': min_epoch,
        #         'loss': loss,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #     }, os.path.join(checkpoint_dir, 'latest.pth')) 

    # # 测试的步骤：也要按照 Polyp 的文件夹列表
    # if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
    #     print('#----------Testing----------#')
    #     best_weight = torch.load(work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
    #     model.load_state_dict(best_weight)
    #     #for name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    #         #val_loader_t = val_loader_dict[name]
    #     loss = test_one_epoch(
    #         val_loader,
    #         model,
    #         criterion,
    #         logger,
    #         model_cfg,
    #         test_data_name='Kvasir'
    #     )
    #     os.rename(
    #         os.path.join(checkpoint_dir, 'best.pth'),
    #         os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
    #     )      


# if __name__ == '__main__':
#     config = setting_config
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str, default=None)
#     parser.add_argument('--pretrained_weight_path', type=str, default=None)
#     parser = parser.parse_args()
#     main(config, parser)