import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
from utils.img_utils import imfrombytes, img2tensor
from utils.FileClient import FileClient
from transforms import augment, paired_random_crop


# Train set build
class REDSRecurrentDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        train_data_opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        dmeta_info_fileataroot_flow (str, optional): Data root path for flow.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, train_data_opt):
        super(REDSRecurrentDataset, self).__init__()
        self.train_data_opt = train_data_opt
        self.gt_root, self.lq_root = Path(train_data_opt['dataroot_gt']), Path(train_data_opt['dataroot_lq'])
        self.num_frame = train_data_opt['num_frame']

        self.keys = []
        with open(train_data_opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if train_data_opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif train_data_opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {train_data_opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if train_data_opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_train_data_opt = train_data_opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_train_data_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_train_data_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_train_data_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_train_data_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_train_data_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = train_data_opt.get('interval_list', [1])
        self.random_reverse = train_data_opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_train_data_opt.pop('type'), **self.io_backend_train_data_opt)

        scale = self.train_data_opt['scale']
        gt_size = self.train_data_opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.train_data_opt['use_hflip'], self.train_data_opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
