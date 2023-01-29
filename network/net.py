# from utils.img_utils import imwrite, tensor2img
import torch
import torch.nn as nn
# from collections import Counter
# from os import path as osp
# from tqdm import tqdm
# from copy import deepcopy
import os
import time
from collections import OrderedDict

# from utils.metrics_utils import calculate_psnr
# from utils.img_utils import tensor2img, imwrite
from model.BasicVSR_arch import BasicVSR
from utils.losses_utils import CharbonnierLoss
import utils.lr_scheduler as lr_scheduler


model_opt = dict(
    network_g=dict(
        num_feat=64,
        num_block=30,
        spynet_path=""
    ),
    path=dict(
        pretrained=None,
        strict_load_g=True,
        resume_state=None,
        log="",
        visualization="",
        training_states=""
    )
)

train_opt = dict(
    ema_decay=0.999,
    optim_g=dict(
        type='Adam',
        lr=0.0002,
        weight_decay=0,
        betas=[0.9, 0.99]
    ),
    scheduler=dict(  # CosineAnnealingRestartLR
        periods=[300000],
        restart_weights=[1],
        eta_min=1e-07
    ),
    total_iter=300000,
    warmup_iter=-1,
    fix_flow=5000,
    flow_lr_mul=0.125,
    pixel_opt=dict(  # charbonnier loss
        loss_weight=1.0,
        reduction='mean'
    )
)

val_opt = dict(
    metrics=dict(

    )
)


class Net():
    def __init__(self, model_opt, train_val_opt, is_train=True):
        self.model_opt = model_opt
        self.train_val_opt = train_val_opt
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            raise Exception("No GPU available")

        self.is_train = is_train
        self.schedulers = []
        self.optimizers = []

        self.net_g = BasicVSR(**self.model_opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # Load pretrained network
        load_path = self.model_opt['path']['pretrained']
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.model_opt['path']['strict_load_g'], 'params')

        if self.is_train:
            self.init_training_settings()
            self.fix_flow_iter = self.train_val_opt['fix_flow']

    def model_to_device(self, net):
        """Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        return net

    def print_network(self, net):
        """Args:
            net (nn.Module)
        """
        net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        print(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        print(net_str)

    def get_bare_model(self, net: nn.Module):
        return net

    def init_training_settings(self):
        self.net_g.train()

        self.ema_decay = self.train_val_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            print('Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            self.net_g_ema = self.model_to_device(BasicVSR(**self.model_opt['network_g']).to(self.device))
            # TODO: rewrite load pretrained model
            load_path = self.model_opt['path']['pretrained']
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.model_opt['path']['strict_load_g'], 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.cri_pix = CharbonnierLoss(**self.train_val_opt['pixel_opt']).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def setup_optimizers(self):
        flow_lr_mul = self.train_val_opt.get('flow_lr_mul', 1)
        print('Multiply the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': self.train_val_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': self.train_val_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

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
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_schedulers(self):
        """Set up schedulers."""
        self.train_val_opt = self.train_val_opt
        self.train_val_opt['scheduler'].pop('type')
        for optimizer in self.optimizers:
            self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **self.train_val_opt['scheduler']))

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # Choose parameters to optimize, after self.fix_flow_iter, train all
        if self.fix_flow_iter:
            if current_iter == 1:
                print(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                print('Train all the parameters.')
                self.net_g.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = self.cri_pix(self.output, self.gt)
        l_total += l_pix
        loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        pass

    def save_training_state(self, epoch, current_iter, ckpt_folder_path):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())

        ema_state_dict = self.net_g_ema.state_dict() if self.ema_decay > 0 else None
        model_state_dict = self.net_g.state_dict()
        state['net_g_ema'] = ema_state_dict
        state['net_g'] = model_state_dict

        save_filename = f'{current_iter}.ckpt'
        save_path = os.path.join(ckpt_folder_path, save_filename)

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(state, save_path)
            except Exception as e:
                print(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            print(f'Still cannot save {save_path}. Just ignore it.')

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        self.net_g.load_state_dict(resume_state['net_g'])
        self.net_g = self.model_to_device(self.net_g)

        if resume_state['net_g_ema'] is not None:
            self.net_g_ema.load_state_dict(resume_state['net_g_ema'])
            self.net_g_ema = self.model_to_device(self.net_g_ema)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.
        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()
            return log_dict

    def validation_psnr(self, dataloader, current_iter, save_img):
        pass
