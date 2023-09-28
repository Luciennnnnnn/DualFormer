import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
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
class SPSRModel(SRModel):
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