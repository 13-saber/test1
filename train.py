import datetime
import os
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.ssd import SSD300
from nets.ssd_train import (MultiboxLoss, get_lr_scheduler,
                               set_optimizer_lr, weights_init)
from utils.anchors import get_anchors
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SSDDataset, ssd_dataset_collate
from utils.utils import download_weights, get_classes, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":

    Cuda=True
    distributed     = False
    sync_bn=False
    fp16            = False
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关 

    classes_path    = 'model_data/voc_classes.txt'
    # model_path      ='model_data/mobilenetv2_ssd_weights.pth'
    # model_path      = 'model_data/best_epoch_weights.pth'
    model_path= ''
    input_shape     = [300, 300]
    backbone        = "mobilenetv2"
    # backbone        = "vgg"

    #   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    pretrained      = True
    
    #   可用于设定先验框的大小，默认的anchors_size
    anchors_size    = [30, 60, 111, 162, 213, 264, 315]

    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 16

    UnFreeze_Epoch      = 1000
    Unfreeze_batch_size = 4

    Freeze_Train        = True

    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=6e-4
    #                   当使用SGD优化器时建议设置   Init_lr=2e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    Init_lr             = 2e-3
    Min_lr              = Init_lr * 0.01

    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=6e-4
    #                   当使用SGD优化器时建议设置   Init_lr=2e-3
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    lr_decay_type       = 'cos'
    save_period         = 10
    #   save_dir        权值与日志文件保存的文件夹
    save_dir            = 'logs'
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   安装pycocotools库后，评估体验更佳。
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    eval_flag           = True
    eval_period         = 50
    num_workers         = 4
    #   train_annotation_path   训练图片路径和标签
    #   val_annotation_path     验证图片路径和标签
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    #   设置用到的显卡
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    #是否预训练
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)
    
    #   获取classes和anchor
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    #默认先验框
    anchors = get_anchors(input_shape, anchors_size, backbone)

    model = SSD300(num_classes, backbone, pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
         if local_rank == 0:
            print('Load weights {}.'.format(model_path))
         #   根据预训练权重的Key和模型的Key进行加载
         model_dict      = model.state_dict()
         pretrained_dict = torch.load(model_path, map_location = device)
         load_key, no_load_key, temp_dict = [], [], {}
         for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
         model_dict.update(temp_dict)
         model.load_state_dict(model_dict)
         #   显示没有匹配上的Key
         if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #   获得损失函数
    criterion       = MultiboxLoss(num_classes, neg_pos_ratio=3.0)
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    #   多卡同步Bn
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #   读取数据集对应的txt
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines) 

    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数 
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
       
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
    
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            if backbone == "vgg":
                for param in model.vgg[:28].parameters():
                    param.requires_grad=False
            else:
                for param in model.mobilenet.parameters():
                    param.requires_grad = False
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        #   判断当前batch_size，自适应调整学习率
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-5
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #   根据optimizer_type选择优化器
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #   获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        #   判断每一个世代的长度
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
        #训练集和测试集
        train_dataset   = SSDDataset(train_lines, input_shape, anchors, batch_size, num_classes, train = True)
        val_dataset     = SSDDataset(val_lines, input_shape, anchors, batch_size, num_classes, train = False)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True
        
        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=ssd_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=ssd_dataset_collate, sampler=val_sampler)
        #   记录eval的map曲线
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, anchors, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        #   开始模型训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #   判断当前batch_size，自适应调整学习率
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-5
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                #   获得学习率下降的公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                if backbone == "vgg":
                    for param in model.vgg[:28].parameters():
                        param.requires_grad = True
                else:
                    for param in model.mobilenet.parameters():
                        param.requires_grad = True
                
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen         = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=ssd_dataset_collate, sampler=train_sampler)
                gen_val     = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=ssd_dataset_collate, sampler=val_sampler)
                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, criterion, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
                        
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()