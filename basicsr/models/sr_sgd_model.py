import numpy as np

import torch
from torch.nn import functional as F

from basicsr.data.degradations import random_add_gaussian_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.img_process_util import filter2D
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class SRSimpleGatedDegradationModel(SRModel):
    """Base SR model for single image super-resolution under simple gated degradation model."""

    def __init__(self, opt):
        super().__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.queue_size = opt.get('queue_size', 180)

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
            out = F.interpolate(out, scale_factor=1/self.opt['scale'], mode='bicubic')

            # add noise
            if np.random.uniform() < 0.5: # gated
                gray_noise_prob = self.opt['gray_noise_prob']
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)

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