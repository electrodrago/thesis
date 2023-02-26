import sys, os
import torch
import math
import time

sys.path.append(os.path.dirname(__file__))

from dataset.REDS_dataset import REDSRecurrentDataset
from dataset.data_sampler import EnlargedSampler
from dataset.REDS_test_dataset import REDSVideoTestDataset
from dataset.build_dataloader import build_dataloader
from dataset.prefetcher import CUDAPrefetcher, CPUPrefetcher

from network.net import Net


torch.backends.cudnn.benchmark = True

train_loader, val_loaders = None, []
train_dataset_opt = dict(
    dataroot_gt="D:\\VSR_dataset\\val_sharp\\val\\val_sharp",
    dataroot_lq="D:\\VSR_dataset\\val_sharp\\val\\val_sharp_bicubic\\X4",
    meta_info_file=".\\meta_info\\meta_info_REDS_GT.txt",
    val_partition="REDS4",
    io_backend="disk",
    num_frame=15,
    gt_size=256,
    interval_list=[1],
    random_reverse=False,
    use_hflip=True,
    use_rot=True,
    scale=4,
    test_mode=False
)
train_set = REDSRecurrentDataset(**train_dataset_opt)

train_sampler_opt = dict(
    dataset=train_set,
    num_replicas=1,
    rank=0,
    ratio=200
)
train_sampler = EnlargedSampler(**train_sampler_opt)

train_loader_opt = dict(
    dataset=train_set,
    phase="train",
    batch_size=4,
    num_workers=2,
    num_gpu=1,
    sampler=train_sampler
)
train_loader = build_dataloader(**train_loader_opt)

dataset_enlarge_ratio = 200
batch_size = 4
num_workers = 2
world_size = 1
total_iters = 300000

num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / (batch_size * world_size))
total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
print('Training statistics:'
    f'\n\tNumber of train images: {len(train_set)}'
    f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
    f'\n\tBatch size per gpu: {batch_size}'
    f'\n\tWorld size (gpu number): {world_size}'
    f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
    f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')


# val_dataset_opt = dict(
#     dataroot_gt="D:\\VSR_dataset\\val_sharp\\val\\val_sharp",
#     dataroot_lq="D:\\VSR_dataset\\val_sharp\\val\\val_sharp_bicubic\\X4",
#     io_backend=dict(type='disk'),
#     cache_data=True,
#     name='REDS4',
#     num_frame=-1,
#     meta_info_file=None
# )
# val_set = REDSVideoTestDataset(**val_dataset_opt)

# val_loader_opt = dict(
#     dataset=val_set,
#     phase="val"
# )
# val_loader = build_dataloader(**val_dataset_opt)

# print(f'Number of val images/folders in REDS4: {len(val_set)}')
# val_loaders.append(val_loader)

model_opt = dict(
    network_g=dict(
        num_feat=64,
        num_block=10,
        spynet_path=""
    ),
    ckpt=None
)

train_opt = dict(
    ema_decay=0,        # 0: not use ema
    optim_g=dict(
        type='Lion',    # Adam
        lr=0.0002,
        weight_decay=0,
        betas=[0.9, 0.99]
    ),
    pixel_opt=dict(     # Charbonnier loss
        loss_weight=1.0,
        reduction='mean'
    ),
    cleaning_opt=dict(  # L1 loss
        loss_weight=1.0,
        reduction='mean'
    )
)

start_epoch = 0
current_iter = 0
# TODO: build model and resume training code
model = Net(model_opt, train_opt, is_train=True)

# model.print_network(model.net_g)

# model.resume_training("1200.ckpt")
# prefetcher = CPUPrefetcher(train_loader)  # CUDAPrefetcher(train_loader)

# for epoch in range(start_epoch, total_epochs + 1):
#     train_sampler.set_epoch(epoch)
#     prefetcher.reset()
#     train_data = prefetcher.next()

#     while train_data is not None:
#         current_iter += 1
#         if current_iter > total_iters:
#             break

#         print(f'Epoch: {epoch}, iter: {current_iter}')
#         # Feed to model
#         model.feed_data(train_data)

#         # Compute loss, back_prop
#         model.optimize_parameters(current_iter)

#         # Evaluate
#         for val_loader in val_loaders:
#             model.validation_psnr(val_loader, current_iter, 'path_to_save_img')

#         if current_iter % 1000 == 0:
#             model.save_training_state(epoch, current_iter, 'path_to_save_ckpt')

#         train_data = prefetcher.next()
