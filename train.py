from dataset.REDS_dataset import REDSRecurrentDataset
from dataset.data_sampler import EnlargedSampler
from dataset.REDS_test_dataset import REDSVideoRecurrentTestDataset
from dataset.build_dataloader import build_dataloader
import torch
import math

torch.backends.cudnn.benchmark = True

train_loader, val_loaders = None, []
# Train loaders:
train_set = REDSRecurrentDataset()
train_sampler = EnlargedSampler()
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
