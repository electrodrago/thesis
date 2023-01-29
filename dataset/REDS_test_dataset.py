import glob
import torch
from os import path as osp
from torch.utils import data as data
from utils.data_utils import generate_frame_indices, read_img_seq, scandir


# Test dataset builder
class REDSVideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    ::f

        dataroot
        ├── subfolder1
            ├── frame000
            ├── frame001
            ├── ...
        ├── subfolder2
            ├── frame000
            ├── frame001
            ├── ...
        ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        io_backend (dict): IO backend type and other kwarg.
        cache_data (bool): Whether to cache testing datasets.
        name (str): Dataset name.
        meta_info_file (str): The path to the file storing the list of test folders. If not provided, all the folders
            in the dataroot will be used.
        num_frame (int): Window size for input frames.
        padding (str): Padding mode.
    """

    def __init__(self, dataroot_gt, dataroot_lq, io_backend, cache_data, name, num_frame, padding, meta_info_file=None):
        super(REDSVideoTestDataset, self).__init__()
        self.cache_data = cache_data
        self.gt_root = dataroot_gt
        self.lq_root = dataroot_lq
        self.num_frame = num_frame
        self.padding = padding

        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}

        # File client (io backend)
        self.file_client = None
        self.io_backend = io_backend

        print(f'Generate data info for REDSVideoTestDataset - {name}')
        self.imgs_lq, self.imgs_gt = {}, {}
        if meta_info_file:
            with open(meta_info_file, 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        # Split into each folder and ground truth
        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # Get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
            img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                      f' and gt folders ({len(img_paths_gt)})')

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.num_frame // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # Cache data or save the frame list
            if self.cache_data:
                print(f'Cache {subfolder_name} for REDSVideoTestDataset...')
                self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.num_frame, padding=self.padding)

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq(self.imgs_gt[folder][idx])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


class REDSVideoRecurrentTestDataset(REDSVideoTestDataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.
    """
    def __init__(self, dataroot_gt, dataroot_lq, io_backend, cache_data, meta_info_file, name, num_frame, padding):
        super(REDSVideoTestDataset, self).__init__(dataroot_gt, dataroot_lq, io_backend, cache_data, meta_info_file, name, num_frame, padding)
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
        else:
            raise NotImplementedError('Without cache_data is not implemented.')

        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)
