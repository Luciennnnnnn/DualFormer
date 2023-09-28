import os

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as transforms_f


class IQATestSet(Dataset):
    def __init__(self,
                 path: str,
                 guide_file: str):
        """
        Load IQA test set which has (image, MOS) pairs.
        :param path:
            The path of image set.
        :param guide_file:
            A text file which has image and MOS pair info.
            Used to locate the image file and give the ground-truth MOS.
        """
        super(IQATestSet, self).__init__()

        self.path = path

        pairs = [line.rstrip() for line in open(guide_file, mode='r')]
        self.iqa_pairs = [
            one_pair.split(',') for one_pair in pairs
        ]  # It's a list containing many (image, MOS) pairs.

    def __getitem__(self, index):
        image_name, mos_str = tuple(self.iqa_pairs[index])
        image = Image.open(os.path.join(self.path, image_name)).convert("RGB")
        mos = float(mos_str)

        image = transforms_f.to_tensor(image)

        return {"image": image, "MOS": mos, "image_name": image_name}

    def __len__(self):
        return len(self.iqa_pairs)


class BappsSuperresTestSet(Dataset):
    def __init__(self,
                 path: str,
                 guide_file: str):
        """
        Load 2afc->val->superres part in BAPPS dataset as test set.
        In this test set, each sample have 1 reference patch, 2 distorted patches and a perceptual preference.
        The human preference index is in range [0,1],
        here 0 = prefer 1st distorted patch and 1 = prefer 2nd distorted patch.
        Ref:
            https://github.com/richzhang/PerceptualSimilarity#c-about-the-dataset
            Zhang et al. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR 2018.
        :param path:
            The path of image set.
        :param guide_file:
            A text file which has image and preference tuple info.
            Used to locate the patch files and give the ground-truth preference.
        """
        super(BappsSuperresTestSet, self).__init__()

        self.path = path

        samples = [line.rstrip() for line in open(guide_file, mode='r')]
        self.samples = [
            one_sample.split(',') for one_sample in samples
        ]  # Sequence: 1st distorted patch "p0", 2nd distorted patch "p1", reference patch "ref", judgement "prefer".

    def __getitem__(self, index):
        p0, p1, ref, prefer = tuple(self.samples[index])
        p0 = Image.open(self.path + p0).convert("RGB")
        p1 = Image.open(self.path + p1).convert("RGB")
        ref = Image.open(self.path + ref).convert("RGB")

        prefer = float(prefer)
        prefer = int(0 if prefer < 0.5 else 1)

        p0 = transforms_f.to_tensor(p0)
        p1 = transforms_f.to_tensor(p1)
        ref = transforms_f.to_tensor(ref)

        return {"p0": p0, "p1": p1, "ref": ref, "prefer": prefer}

    def __len__(self):
        return len(self.samples)


class SRPairImageTestSet(Dataset):
    def __init__(self,
                 path: str,
                 guide_file: str,
                 scale_factor: int = 4):
        """
        Load super-resolution test set which has many paired images.
        :param path:
            The path of dataset images.
        :param guide_file:
            A text file which has paired images info.
        :param scale_factor:
            Upscale factor.
            Default: 4.
        """
        super(SRPairImageTestSet, self).__init__()

        self.path = path

        self.scale_factor = scale_factor

        pairs = [line.rstrip() for line in open(guide_file, mode='r')]
        self.image_pairs = [
            one_pair.split(',') for one_pair in pairs
        ]  # It's a list containing many (HR, LR) pairs, and each pair is a list of 2 filenames.

    def __getitem__(self, index):
        hr_image, lr_image = tuple(self.image_pairs[index])
        hr_image = Image.open(self.path + hr_image)
        lr_image = Image.open(self.path + lr_image)

        hr_w, hr_h = hr_image.size
        lr_w, lr_h = lr_image.size
        assert hr_w == self.scale_factor * lr_w and hr_h == self.scale_factor * lr_h, (
            f"The size difference of HR and LR images are not X{self.scale_factor} scale factor."
        )

        lr_image = transforms_f.to_tensor(lr_image)
        hr_image = transforms_f.to_tensor(hr_image)

        return {"LR": lr_image, "HR": hr_image}

    def __len__(self):
        return len(self.image_pairs)


class SRSingleImageTestSet(Dataset):
    def __init__(self,
                 path: str,
                 guide_file: str,
                 scale_factor: int = 4):
        """
        Load super-resolution test set which has many HR images but no LR image.
        :param path:
            The path of dataset.
        :param guide_file:
            A text file which has HR images info.
        :param scale_factor:
            Upscale factor.
            Default: 4.
        """
        super(SRSingleImageTestSet, self).__init__()

        self.path = path
        self.scale_factor = scale_factor
        self.images = [line.rstrip() for line in open(guide_file, mode='r')]

    def __getitem__(self, index):
        from utils import matlab_functions

        # Read HR image.
        hr_image = self.images[index]
        hr_image = cv2.imread(self.path + hr_image, cv2.IMREAD_COLOR)
        hr_image = cv2.cvtColor(
            hr_image,
            code=cv2.COLOR_BGR2RGB
        )  # Convert from BGR to RGB, unified with the Pillow library.

        # Resize HR image if its height or width is illegal.
        if hr_image.shape[0] % self.scale_factor != 0:
            new_h = (hr_image.shape[0] // self.scale_factor + 1) * self.scale_factor
        else:
            new_h = hr_image.shape[0]
        if hr_image.shape[1] % self.scale_factor != 0:
            new_w = (hr_image.shape[1] // self.scale_factor + 1) * self.scale_factor
        else:
            new_w = hr_image.shape[1]
        if (new_w != hr_image.shape[0]) or (new_w != hr_image.shape[1]):
            hr_image = cv2.resize(hr_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Get LR image.
        lr_image = matlab_functions.imresize(
            hr_image,
            scale=1.0 / self.scale_factor,
            antialiasing=True
        )  # Generate LR patch by bicubic down-sampling (following MATLAB).
        lr_image = np.round(lr_image).astype(hr_image.dtype)

        # From numpy array to pytorch tensor.
        lr_image = transforms_f.to_tensor(lr_image)
        hr_image = transforms_f.to_tensor(hr_image)

        return {"LR": lr_image, "HR": hr_image}

    def __len__(self):
        return len(self.images)


def get_dataloader(choice: str,
                   path: str,
                   guide_file: str,
                   scale_factor: int,
                   batch_size: int = 1,
                   shuffle: bool = False,
                   num_workers: int = 0):
    """
    Return a dataloader for test.
    :param choice:
        Choose the test set type.
        Support: "SR_Pair", "SR_Single", "IQA", "BappsSuperres".
    :param path:
        The path of dataset.
    :param guide_file:
        A text file which has image info.
    :param scale_factor:
        Upscale factor.
        Unused If conduct IQA test.
    :param batch_size:
        The size of one batch.
    :param shuffle:
        Indicate if shuffle when load samples.
    :param num_workers:
        How many subprocesses to use for data loading.
    """
    from torch.utils.data import DataLoader

    if choice == "SR_Pair":
        test_set = SRPairImageTestSet(path, guide_file, scale_factor=scale_factor)
    elif choice == "SR_Single":
        test_set = SRSingleImageTestSet(path, guide_file, scale_factor=scale_factor)
    elif choice == "IQA":
        test_set = IQATestSet(path, guide_file)
    elif choice == "BappsSuperres":
        test_set = BappsSuperresTestSet(path, guide_file)
    else:
        raise NotImplementedError(f"The choice = {choice} is illegal.")

    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )


if __name__ == '__main__':
    pass