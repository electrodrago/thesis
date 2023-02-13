from dataset.REDS_dataset import REDSRecurrentDataset
from dataset.data_sampler import EnlargedSampler
from dataset.REDS_test_dataset import REDSVideoRecurrentTestDataset
from dataset.build_dataloader import build_dataloader
import torch
import math

torch.backends.cudnn.benchmark = True

train_loader, val_loaders = None, []
# Train loaders:
# dataroot_gt, dataroot_lq, meta_info_file, val_partition, io_backend, num_frame, gt_size, interval_list, random_reverse, use_hflip, use_rot, scale, test_mode
train_set = REDSRecurrentDataset("/content/drive/MyDrive/1THESIS/train/train_sharp", "/content/drive/MyDrive/1THESIS/train/train_sharp_bicubic/X4", "/content/thesis/meta_info/meta_info_REDS_GT.txt", "REDS4", 'disk', 15, 256, [1], False, True, True, 4, False)
# train_set = REDSRecurrentDataset()
# train_sampler = EnlargedSampler()
world_size = 1
rank = 0
dataset_enlarge_ratio = 1
train_sampler = EnlargedSampler(train_set, 1, rank, dataset_enlarge_ratio)
train_loader = build_dataloader()

dataset_enlarge_ratio = 1
batch_size = 4
num_workers = 2
world_size = 1
total_iter = 300000

num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / (batch_size * world_size))
total_iters = total_iter
total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
print('Training statistics:'
    f'\n\tNumber of train images: {len(train_set)}'
    f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
    f'\n\tBatch size per gpu: {batch_size}'
    f'\n\tWorld size (gpu number): {world_size}'
    f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
    f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

val_name = 'REDS4'
val_set = REDSVideoRecurrentTestDataset()
val_loader = build_dataloader()
print(f'Number of val images/folders in {val_name}: {len(val_set)}')
val_loaders.append(val_loader)

# train_loader, train_sampler, val_loaders, total_epochs, total_iters
