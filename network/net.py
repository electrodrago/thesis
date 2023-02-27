import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path as oisp
from tqdm import tqdm
import numpy as np
import time
from collections import OrderedDict
from lion_pytorch import Lion
import sys, os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.metrics_utils import psnr
from utils.img_utils import tensor2img, imwrite
from utils.losses_utils import CharbonnierLoss, L1Loss
# TODO: setup wandb


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Net():
    def __init__(self, model_opt, train_val_opt, is_train=True):
        self.model_opt = model_opt
        self.train_val_opt = train_val_opt
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            # raise Exception("No GPU available")

        self.is_train = is_train
        self.optimizers = []

        self.net_g = None
        self.net_g = self.model_to_device(self.net_g)

        if self.is_train:
            self.init_training_settings()

        # Load pretrained network
        load_path = self.model_opt['ckpt']
        if load_path is not None:
            self.resume_training(load_path)

    def model_to_device(self, net):
        net = net.to(self.device)
        return net

    def print_network(self, net):
        net_cls_str = f'{net.__class__.__name__}'

        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        print(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        print(net_str)

    def init_training_settings(self):
        self.net_g.train()

        self.ema_decay = self.train_val_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            print('Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            self.net_g_ema = self.model_to_device(None)
            load_path = self.model_opt['ckpt']
            if load_path is None:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.pixel_loss = CharbonnierLoss(**self.train_val_opt['pixel_opt']).to(self.device)

        self.cleaning_loss = L1Loss(**self.train_val_opt['cleaning_opt']).to(self.device)

        # set up optimizers
        self.setup_optimizers()

    def model_ema(self, decay=0.999):
        net_g_params = dict(self.net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def setup_optimizers(self):
        optim_params = self.net_g.parameters()
        optim_type = self.train_val_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **self.train_val_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        elif optim_type == 'Lion':
            optimizer = Lion(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output, lq_clean = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = self.pixel_loss(self.output, self.gt)
        l_total += l_pix
        loss_dict['l_pix'] = l_pix

        # cleaning loss
        n, t, c, h, w = self.gt.size()
        gt_clean = self.gt.clone()
        gt_clean = gt_clean.view(-1, c, h, w)
        gt_clean = F.interpolate(gt_clean, scale_factor=0.25, mode='area')
        gt_clean = gt_clean.view(n, t, c, h // 4, w // 4)

        l_clean = self.cleaning_loss(lq_clean, gt_clean)
        l_total += l_clean
        loss_dict['l_clean'] = l_clean

        l_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output, _ = self.net_g(self.lq)
        self.net_g.train()

    def save_training_state(self, epoch, current_iter, ckpt_folder_path):
        state = {'epoch': epoch, 'iter': current_iter, 'optimizers': []}
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())

        ema_state_dict = self.net_g_ema.state_dict() if self.ema_decay > 0 else None
        model_state_dict = self.net_g.state_dict()
        state['net_g_ema'] = ema_state_dict
        state['net_g'] = model_state_dict

        save_filename = f'{current_iter}.ckpt'
        save_path = os.path.join(ckpt_folder_path, save_filename)

        torch.save(state, save_path)

    def resume_training(self, load_path):
        resume_state = torch.load(load_path, map_location=lambda storage, loc: storage.cuda())

        resume_optimizers = resume_state['optimizers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)

        self.net_g.load_state_dict(resume_state['net_g'])
        self.net_g = self.model_to_device(self.net_g)

        if resume_state['net_g_ema'] is not None and self.ema_decay > 0:
            self.net_g_ema.load_state_dict(resume_state['net_g_ema'])
            self.net_g_ema = self.model_to_device(self.net_g_ema)
            self.net_g_ema.eval()

    def validation_psnr(self, dataloader, current_iter, save_image: str):
        dataset = dataloader.dataset
        if not hasattr(self, 'psnr_results'):
            self.psnr_results = dict()

        pbar = tqdm(total=len(dataset), unit='folder')

        # Run evaluation on each validation folder
        for i in range(len(dataset)):
            # Prepare to feed data to model
            val_data = dataset[i]

            # Tensor lq [100, 3, 180, 320] -> [1, 100, 3, 180, 320]
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)

            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)

            name = val_data['folder']
            self.test()

            # Give images to cpu to compute PSNR and save images
            # Return shape [1, 100, 3, 720, 1280] for output
            # lq = self.lq.detach().cpu()
            output = self.output.detach().cpu()
            gt = self.gt.detach().cpu()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.gt
            torch.cuda.empty_cache()

            # Check separate image, size at first dim -> REDS = 100
            # Run on each image then average
            psnr_folder = []
            for j in range(output.size(1)):
                result = output[0, j, :, :, :]
                result_img = tensor2img([result])  # uint8, bgr
                gt_j = gt[0, j, :, :, :]
                gt_j_img = tensor2img([gt_j])  # uint8, bgr

                psnr_folder.append(psnr(result_img, gt_j_img))

                if save_image and not self.is_train:
                    safe_mkdir(save_image)
                    imwrite(result_img, oisp.join(save_image, f'{name}_{j}.png'))
            # Get the mean of each run
            # self.psnr_results['000'] = [29.04, 28.04, ...] with each
            # number is the average PSNR of each validation iter
            locate = str(current_iter) + "_" + name
            self.psnr_results[locate] = np.mean(psnr_folder)  # psnr_folder
            pbar.update(1)
            pbar.set_description(f'Folder: {name}')

        pbar.close()
