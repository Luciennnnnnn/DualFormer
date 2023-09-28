import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils import DiffJPEG
from basicsr.utils.registry import MODEL_REGISTRY

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils.img_process_util import filter2D

from .sr_model import SRModel


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

@MODEL_REGISTRY.register()
class SPSRHardGatedDegradationModel(SRModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.queue_size = opt.get('queue_size', 180)

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'skip')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # define network net_grad_d
        self.net_grad_d = build_network(self.opt['net_grad_d'])
        self.net_grad_d = self.model_to_device(self.net_grad_d)
        self.print_network(self.net_grad_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()
        self.net_grad_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)
        else:
            self.cri_ldl = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('pix_grad_opt'):
            self.cri_pix_grad = build_loss(train_opt['pix_grad_opt']).to(self.device)
        else:
            self.cri_pix_grad = None

        if train_opt.get('pix_branch_opt'):
            self.cri_pix_branch = build_loss(train_opt['pix_branch_opt']).to(self.device)
        else:
            self.cri_pix_branch = None

        self.d_weight = self.opt['train']['gan_opt'].get('d_weight', 1)
        self.grad_d_weight = self.opt['train']['gan_opt'].get('grad_d_weight', 1)

        self.branch_pretrain = train_opt.get('branch_pretrain', 1)
        self.branch_init_iters = train_opt.get('branch_init_iters', 5000)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        self.get_grad = Get_gradient()
        self.get_grad_nopadding = Get_gradient_nopadding()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)
        # optimizer spectral_d
        optim_type = train_opt['optim_grad_d'].pop('type')
        self.optimizer_grad_d = self.get_optimizer(optim_type, self.net_grad_d.parameters(), **train_opt['optim_grad_d'])
        self.optimizers.append(self.optimizer_grad_d)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('simple_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)

            self.kernel = data['kernel'].to(self.device)

            # blur
            if np.random.uniform() < 0.5: # gated
                out = filter2D(self.gt, self.kernel)
            else:
                out = self.gt

            # dowm sample
            if self.opt['scale'] != 1:
                out = F.interpolate(out, scale_factor=1/self.opt['scale'], mode='bicubic')

            # add noise
            if np.random.uniform() < 0.5: # gated
                gray_noise_prob = self.opt['gray_noise_prob']
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

            # JPEG compression
            if np.random.uniform() < 0.5: # gated
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
                out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out = self.jpeger(out, quality=jpeg_p)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger):
        # do not use the synthetic process during validation
        self.is_train = False
        super().nondist_validation(dataloader, current_iter, tb_logger)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        for p in self.net_grad_d.parameters():
            p.requires_grad = False

        if(self.branch_pretrain):
            if(current_iter < self.branch_init_iters):
                for k, v in self.net_g.named_parameters():
                    if 'f_' not in k :
                        v.requires_grad=False
            else:
                for k, v in self.net_g.named_parameters():
                    if 'f_' not in k :
                        v.requires_grad=True

        self.optimizer_g.zero_grad()
        self.fake_H_branch, self.fake_H, self.grad_LR = self.net_g(self.lq)

        self.fake_H_grad = self.get_grad(self.fake_H)
        self.real_H_grad = self.get_grad(self.gt)
        self.real_H_grad_nopadding = self.get_grad_nopadding(self.gt)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.fake_H, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.fake_H, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # gradient pixel loss
            if self.cri_pix_grad:
                l_g_pix_grad = self.cri_pix_grad(self.fake_H_grad, self.real_H_grad)
                l_g_total += l_g_pix_grad
                loss_dict['l_g_pix_grad'] = l_g_pix_grad

            # gradient pixel loss
            if self.cri_pix_branch:
                l_g_pix_grad_branch = self.cri_pix_branch(self.fake_H_branch, self.real_H_grad_nopadding)
                l_g_total += l_g_pix_grad_branch
                loss_dict['l_g_pix_grad_branch'] = l_g_pix_grad_branch

            # gan loss (relativistic gan)
            real_d_pred = self.net_d(self.gt).detach()
            fake_d_pred = self.net_d(self.fake_H)
            l_g_real_d = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), False, is_disc=False)
            l_g_fake_d = self.cri_gan(fake_d_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan_d = self.d_weight * (l_g_real_d + l_g_fake_d) / 2

            real_grad_d_pred = self.net_grad_d(self.real_H_grad).detach()
            fake_grad_d_pred = self.net_grad_d(self.fake_H_grad)
            l_g_real_grad_d = self.cri_gan(real_grad_d_pred - torch.mean(fake_grad_d_pred), False, is_disc=False)
            l_g_fake_grad_d = self.cri_gan(fake_grad_d_pred - torch.mean(real_grad_d_pred), True, is_disc=False)
            l_g_gan_grad_d = self.grad_d_weight * (l_g_real_grad_d + l_g_fake_grad_d) / 2

            l_g_gan = l_g_gan_d + l_g_gan_grad_d

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            loss_dict['l_g_gan_d'] = l_g_gan_d
            loss_dict['l_g_gan_grad_d'] = l_g_gan_grad_d

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.fake_H.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        # optimize net_spectral_d
        for p in self.net_grad_d.parameters():
            p.requires_grad = True

        self.optimizer_grad_d.zero_grad()
        # real
        real_grad_d_pred = self.net_grad_d(self.real_H_grad)
        l_grad_d_real = self.cri_gan(real_grad_d_pred, True, is_disc=True)
        loss_dict['l_grad_d_real'] = l_grad_d_real
        loss_dict['out_grad_d_real'] = torch.mean(real_grad_d_pred.detach())
        l_grad_d_real.backward()
        # fake
        fake_grad_d_pred = self.net_grad_d(self.fake_H_grad.detach())
        l_grad_d_fake = self.cri_gan(fake_grad_d_pred, False, is_disc=True)
        loss_dict['l_grad_d_fake'] = l_grad_d_fake
        loss_dict['out_grad_d_fake'] = torch.mean(fake_grad_d_pred.detach())
        l_grad_d_fake.backward()
        self.optimizer_grad_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_network(self.net_grad_d, 'net_grad_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                _, self.output, _ = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                _, self.output, _ = self.net_g(self.lq)
            self.net_g.train()