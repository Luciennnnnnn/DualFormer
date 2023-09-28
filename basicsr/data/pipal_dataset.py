import os
import torch
import numpy as np
from torchvision.transforms.functional import normalize

from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class PIPAL(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(PIPAL, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.dataroot = opt['dataroot']
        label_info_file = opt['label_info_file']

        ref_pathes, dis_pathes, scores = [], [], []
        with open(label_info_file, 'r') as fin:
            for line in fin:
                dis_name, score = line.strip().split(',')
                dis_path = os.path.join(self.dataroot, dis_name)

                basename, extension = os.path.splitext(os.path.basename(dis_name))

                ref_name = basename.split('_')[0] + extension
                ref_path = os.path.join(self.dataroot, 'Train_Ref', ref_name)

                score = float(score)
                ref_pathes.append(ref_path)
                dis_pathes.append(dis_path)
                scores.append(score)

        scores = np.array(scores, dtype=np.float32)
        scores = self.normalization(scores)

        self.data = []
        for dis_path, ref_path, score in zip(dis_pathes, ref_pathes, list(scores)):
            self.data.append(dict([('dis_path', dis_path), ('ref_path', ref_path), ('score', score)]))

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        ref_path = self.data[index]['ref_path']
        img_bytes = self.file_client.get(ref_path, 'ref')
        img_ref = imfrombytes(img_bytes, float32=True)
        dis_path = self.data[index]['dis_path']
        img_bytes = self.file_client.get(dis_path, 'dis')
        img_dis = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            size = self.opt['size']
            # random crop
            img_ref, img_dis = paired_random_crop(img_ref, img_dis, size, 1, ref_path)
            # flip, rotation
            img_ref, img_dis = augment([img_ref, img_dis], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_ref, img_dis = img2tensor([img_ref, img_dis], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_dis, self.mean, self.std, inplace=True)
            normalize(img_ref, self.mean, self.std, inplace=True)

        score = torch.tensor(self.data[index]['score'], dtype=torch.float32)

        return {'dis': img_dis, 'ref': img_ref, 'dis_path': dis_path, 'ref_path': ref_path, 'score': score}


@DATASET_REGISTRY.register()
class PIPALVal(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(PIPALVal, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.dataroot_dis = opt['dataroot_dis']
        self.dataroot_ref = opt['dataroot_ref']

        dis_pathes = sorted(list(scandir(self.dataroot_dis, full_path=True)))

        ref_pathes = []
        for dis_path in dis_pathes:
            basename, extension = os.path.splitext(os.path.basename(dis_path))

            ref_name = basename.split('_')[0] + extension
            ref_path = os.path.join(self.dataroot_ref, ref_name)

            ref_pathes.append(ref_path)

        self.data = []
        for dis_path, ref_path in zip(dis_pathes, ref_pathes):
            self.data.append(dict([('dis_path', dis_path), ('ref_path', ref_path)]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        ref_path = self.data[index]['ref_path']
        img_bytes = self.file_client.get(ref_path, 'ref')
        img_ref = imfrombytes(img_bytes, float32=True)
        dis_path = self.data[index]['dis_path']
        img_bytes = self.file_client.get(dis_path, 'dis')
        img_dis = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            size = self.opt['size']
            # random crop
            img_ref, img_dis = paired_random_crop(img_ref, img_dis, size, 1, ref_path)
            # flip, rotation
            img_ref, img_dis = augment([img_ref, img_dis], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_ref, img_dis = img2tensor([img_ref, img_dis], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_dis, self.mean, self.std, inplace=True)
            normalize(img_ref, self.mean, self.std, inplace=True)

        return {'dis': img_dis, 'ref': img_ref, 'dis_path': dis_path, 'ref_path': ref_path}