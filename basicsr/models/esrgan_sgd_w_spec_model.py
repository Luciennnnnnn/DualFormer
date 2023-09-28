from functools import partial
from collections import OrderedDict

import torch

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger

from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.models.esrgan_sgd_w_model import ESRGANSimpleGatedDegradationWModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.diffaug import DiffAugment
from basicsr.losses.losses import gradient_penalty_loss_2


@MODEL_REGISTRY.register(suffix='basicsr')
class ESRGANSimpleGatedDegradationWSpecModel(ESRGANSimpleGatedDegradationWModel):
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
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        # define network net_spectral_d
        self.net_spectral_d = build_network(self.opt['net_spectral_d'])
        self.net_spectral_d = self.model_to_device(self.net_spectral_d)
        self.print_network(self.net_spectral_d)

        self.net_g.train()
        self.net_d.train()
        self.net_spectral_d.train()

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

        if train_opt.get('frequency_opt'):
            self.cri_frequency = build_loss(train_opt['frequency_opt']).to(self.device)
        else:
            self.cri_frequency = None

        if self.opt['train']['gan_opt'].get('diffaug', False):
            self.diff_aug = partial(DiffAugment, policy=self.opt['train']['gan_opt']['aug_policy'])
        else:
            self.diff_aug = lambda input: input

        self.d_weight = self.opt['train']['gan_opt'].get('d_weight', 1)
        self.spectral_d_weight = self.opt['train']['gan_opt'].get('spectral_d_weight', 1)

        self.adv_loss_version = self.opt['train']['gan_opt'].get('adv_loss_version', 'v1')

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

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
        optim_type = train_opt['optim_spectral_d'].pop('type')
        self.optimizer_spectral_d = self.get_optimizer(optim_type, self.net_spectral_d.parameters(), **train_opt['optim_spectral_d'])
        self.optimizers.append(self.optimizer_spectral_d)

    def optimize_parameters(self, current_iter, tb_logger=None):
        l1_gt = self.gt
        percep_gt = self.gt
        gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        for p in self.net_spectral_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        if self.cri_ldl:
            self.output_ema = self.net_g_ema(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            if self.cri_ldl:
                pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
                l_g_ldl = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # frequency loss
            if self.cri_frequency:
                l_g_freq = self.cri_frequency(self.output, self.gt) * 0.5
                l_g_total += l_g_freq
                loss_dict['l_g_freq'] = l_g_freq
            # gan loss
            fake_g_pred_d = self.net_d(self.diff_aug(self.output))
            fake_g_pred_spectral_d = self.net_spectral_d(self.diff_aug(self.output))

            if self.adv_loss_version == 'v1':
                l_g_gan_d = self.d_weight * self.cri_gan(fake_g_pred_d, True, is_disc=False)
                l_g_gan_spectral_d = self.spectral_d_weight * self.cri_gan(fake_g_pred_spectral_d, True, is_disc=False)
                l_g_gan = l_g_gan_d + l_g_gan_spectral_d

                loss_dict['l_g_gan_d'] = l_g_gan_d
                loss_dict['l_g_gan_spectral_d'] = l_g_gan_spectral_d
            elif self.adv_loss_version == 'v2':
                fake_g_pred = self.d_weight * fake_g_pred_d + self.spectral_d_weight * fake_g_pred_spectral_d
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)

                loss_dict['out_g_d'] = torch.mean(self.d_weight * fake_g_pred_d.detach())
                loss_dict['out_g_spectral_d'] = torch.mean(self.spectral_d_weight * fake_g_pred_spectral_d.detach())

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.diff_aug(gan_gt))
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()

        # fake
        fake_d_pred = self.net_d(self.diff_aug(self.output.detach().clone()))
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()

        loss_dict['d_w_distance'] = loss_dict['out_d_real'] - loss_dict['out_d_fake']

        # gradient penalty
        l_gp = gradient_penalty_loss_2(self.net_d, gan_gt, self.output.detach(), lambda_gp=10, grad_trick=True)
        loss_dict['l_gp'] = l_gp
        l_gp.backward()

        self.optimizer_d.step()

        # optimize net_spectral_d
        for p in self.net_spectral_d.parameters():
            p.requires_grad = True

        self.optimizer_spectral_d.zero_grad()

        # real
        real_spectral_d_pred = self.net_spectral_d(self.diff_aug(gan_gt))
        l_spectral_d_real = self.cri_gan(real_spectral_d_pred, True, is_disc=True)
        loss_dict['l_spectral_d_real'] = l_spectral_d_real
        loss_dict['out_spectral_d_real'] = torch.mean(real_spectral_d_pred.detach())
        l_spectral_d_real.backward()
        # fake
        fake_spectral_d_pred = self.net_spectral_d(self.diff_aug(self.output.detach().clone()))  # clone for pt1.9
        l_spectral_d_fake = self.cri_gan(fake_spectral_d_pred, False, is_disc=True)
        loss_dict['l_spectral_d_fake'] = l_spectral_d_fake
        loss_dict['out_spectral_d_fake'] = torch.mean(fake_spectral_d_pred.detach())
        l_spectral_d_fake.backward()

        loss_dict['spectral_d_w_distance'] = loss_dict['out_spectral_d_real'] - loss_dict['out_spectral_d_fake']

        # gradient penalty
        l_spectral_gp = gradient_penalty_loss_2(self.net_spectral_d, gan_gt, self.output.detach(), lambda_gp=10, grad_trick=True)
        loss_dict['l_spectral_gp'] = l_spectral_gp
        l_spectral_gp.backward()

        self.optimizer_spectral_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_network(self.net_spectral_d, 'net_spectral_d', current_iter)
        self.save_training_state(epoch, current_iter)

        # self.delete_unnecessaries(['net_g', 'net_d', 'net_spectral_d'])
        self.delete_unnecessaries(['net_d', 'net_spectral_d'])