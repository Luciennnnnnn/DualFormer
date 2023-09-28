import torch
from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.utils import get_root_logger

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.srgan_spec_model import SRGANSpecModel


@MODEL_REGISTRY.register()
class ESRGANSpecModel(SRGANSpecModel):
    def optimize_parameters(self, current_iter):
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
                l_g_pix = self.cri_pix(self.output, self.gt) * (1 if self.cri_frequency is None else 0.5)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            if self.cri_ldl:
                pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
                l_g_ldl = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
            # frequency loss
            if self.cri_frequency:
                l_g_freq = self.cri_frequency(self.output, self.gt) * 0.5
                l_g_total += l_g_freq
                loss_dict['l_g_freq'] = l_g_freq

            # gan loss (relativistic gan)
            real_d_pred = self.net_d(self.gt).detach()
            fake_d_pred = self.net_d(self.output)
            l_g_real_d = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), False, is_disc=False)
            l_g_fake_d = self.cri_gan(fake_d_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan_d = self.d_weight * (l_g_real_d + l_g_fake_d) / 2

            real_spectral_d_pred = self.net_spectral_d(self.gt).detach()
            fake_spectral_d_pred = self.net_spectral_d(self.output)
            l_g_real_spectral_d = self.cri_gan(real_spectral_d_pred - torch.mean(fake_spectral_d_pred), False, is_disc=False)
            l_g_fake_spectral_d = self.cri_gan(fake_spectral_d_pred - torch.mean(real_spectral_d_pred), True, is_disc=False)
            l_g_gan_spectral_d = self.spectral_d_weight * (l_g_real_spectral_d + l_g_fake_spectral_d) / 2

            l_g_gan = l_g_gan_d + l_g_gan_spectral_d

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            loss_dict['l_g_gan_d'] = l_g_gan_d
            loss_dict['l_g_gan_spectral_d'] = l_g_gan_spectral_d

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        fake_d_pred = self.net_d(self.output).detach()
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()

        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        # optimize net_spectral_d
        for p in self.net_spectral_d.parameters():
            p.requires_grad = True

        self.optimizer_spectral_d.zero_grad()
        # real
        fake_spectral_d_pred = self.net_spectral_d(self.output).detach()
        real_spectral_d_pred = self.net_spectral_d(self.gt)
        l_spectral_d_real = self.cri_gan(real_spectral_d_pred - torch.mean(fake_spectral_d_pred), True, is_disc=True) * 0.5
        l_spectral_d_real.backward()
        # fake
        fake_spectral_d_pred = self.net_spectral_d(self.output.detach())
        l_spectral_d_fake = self.cri_gan(fake_spectral_d_pred - torch.mean(real_spectral_d_pred.detach()), False, is_disc=True) * 0.5
        l_spectral_d_fake.backward()

        self.optimizer_spectral_d.step()

        loss_dict['l_spectral_d_real'] = l_spectral_d_real
        loss_dict['l_spectral_d_fake'] = l_spectral_d_fake
        loss_dict['out_spectral_d_real'] = torch.mean(real_spectral_d_pred.detach())
        loss_dict['out_spectral_d_fake'] = torch.mean(fake_spectral_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)