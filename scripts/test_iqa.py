"""
Used to test discriminator on the IQA dataset.
Support:
    1. Test ImageGAN discriminator on IQA dataset.
    2. Test PatchGAN discriminator on IQA dataset.
    3. Test optimized PatchGAN discriminator (optimized by using weight matrix) on IQA dataset.
"""
import argparse
import os
import glob
import yaml

from tqdm import tqdm

import numpy as np

import torch

from basicsr.data import test_set
from basicsr.archs import build_network
from basicsr.metrics import correlation as corr
from basicsr.utils.img_process_util import patchify
from basicsr.utils.options import ordered_yaml

from pytorch_lightning import seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--experiment_name', type=str, default="train_RealESRGANPx4plus_400k_B12G4")
    parser.add_argument('--test_set_name', type=str, default="KonIQ-10k")
    parser.add_argument('--test_set', type=str, default="datasets/KonIQ-10k/1024x768/")
    parser.add_argument('--test_set_guider', type=str, default="datasets/IQA_guider/KonIQ-10k_all_pair.txt")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--img_size', type=int, nargs="+", default=[192, 192])
    parser.add_argument('--spatial_window_size', type=int, default=0)
    parser.add_argument('--spectral_window_size', type=int, default=0)

    parser.add_argument('--stride', type=int, default=64)

    parser.add_argument('--test_protocol_spatial', type=str, default="default")
    parser.add_argument('--test_protocol_spectral', type=str, default="patchify")
    parser.add_argument('--n_ensemble', type=int, default=20)

    parser.add_argument('--CLIVE_crop_spatial', action='store_true')

    args = parser.parse_args()

    if args.test_set_name == 'KonIQ-10k':
        args.n_ensemble = 40

    current_work_dir = os.path.dirname(os.path.dirname(__file__))
    experiment_dir = os.path.join(current_work_dir, "experiments", args.experiment_name)

    print(f"test: {args.experiment_name} {args.test_set_name}")

    seed_everything(args.seed)

    opt_path = glob.glob(os.path.join(experiment_dir, '*.yml'))
    # print(f"{os.path.join(experiment_dir, '*.yml')=}")

    assert len(opt_path) == 1

    opt_path = opt_path[0]

    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    if args.spatial_window_size != 0:
        if 'window_size' in opt['network_d']:
            opt['network_d']['window_size'] = args.spatial_window_size

    if args.spectral_window_size != 0:
        if 'net_spectral_d' in opt and 'window_size' in opt['net_spectral_d']:
            opt['net_spectral_d']['window_size'] = args.spectral_window_size

    if opt['network_d']['type'] == 'SwinTransformer' or (opt['network_d']['type'] == 'SpectralDiscriminator2D' and opt['network_d']['version'] == 'swin'):
        opt['network_d']['img_size'] = args.img_size
    if 'net_spectral_d' in opt and (opt['net_spectral_d']['type'] == 'SwinTransformer' or (opt['net_spectral_d']['type'] == 'SpectralDiscriminator2D' and opt['net_spectral_d']['version'] == 'swin')):
        opt['net_spectral_d']['img_size'] = args.img_size

    # region -------- Initialize Discriminator (predictor) --------

    spatial_discriminator_ckpt = os.path.join(experiment_dir, 'models', f'net_d.pth')
    spatial_discriminator = build_network(opt['network_d'])

    state_dict = torch.load(spatial_discriminator_ckpt)['params']

    spatial_discriminator.load_state_dict(state_dict)
    spatial_discriminator = spatial_discriminator.cuda()
    spatial_discriminator.eval()

    if 'net_spectral_d' in opt:
        spectral_discriminator_ckpt = os.path.join(experiment_dir, 'models', f'net_spectral_d.pth')
        spectral_discriminator = build_network(opt['net_spectral_d'])

        state_dict = torch.load(spectral_discriminator_ckpt)['params']

        spectral_discriminator.load_state_dict(state_dict)
        spectral_discriminator = spectral_discriminator.cuda()
        spectral_discriminator.eval()
    # endregion

    # region -------- Dataloader --------
    test_loader = test_set.get_dataloader(
        choice="IQA",
        path=os.path.join(current_work_dir, args.test_set),
        guide_file=os.path.join(current_work_dir, args.test_set_guider),
        scale_factor=-1,  # Upscale factor is useless in IQA test.
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"There are {len(test_loader)} batches in IQA test set. "
          f"The batch size: {test_loader.batch_size}.")
    # endregion

    # region -------- Test --------
    spatial_d_test_res = {"image_name": [], "MOS": [], "pred_score": []}
    spectral_d_test_res = {"image_name": [], "MOS": [], "pred_score": []}
    test_res = {"image_name": [], "MOS": [], "pred_score": []}
    with torch.no_grad():
        for i, samples in enumerate(tqdm(test_loader)):
            image = samples["image"].cuda()  # The tensor with shape = [N, C, H, W].
            mos = samples["MOS"]  # The tensor with shape = [N].
            name = samples["image_name"]  # The list with shape = [N].

            if args.CLIVE_crop_spatial:
                image_spatial = image[:, :, :496, :496]
            else:
                image_spatial = image

            if args.test_protocol_spatial == 'patchify':
                image_patches = patchify(image_spatial, patch_size=args.img_size, stride=args.stride)
                spatial_d_pred_score = spatial_discriminator(image_patches).view(image.size(0), -1).mean(dim=1)
            elif args.test_protocol_spatial == 'random':
                spatial_d_pred_score = 0
                for _ in range(args.n_ensemble):
                    top = np.random.randint(0, image_spatial.shape[2] - args.img_size[0])
                    left = np.random.randint(0, image_spatial.shape[3] - args.img_size[1])
                    spatial_d_pred_score += spatial_discriminator(image_spatial[:,:, top: top+args.img_size[0], left: left+args.img_size[1]]).view(image_spatial.size(0), -1).mean(dim=1)
                spatial_d_pred_score /= args.n_ensemble
            else:
                spatial_d_pred_score = spatial_discriminator(image_spatial).view(image_spatial.size(0), -1).mean(dim=1)

            mos = mos.view(mos.size(0))  # Reshape to [N].
            spatial_d_pred_score = spatial_d_pred_score.view(spatial_d_pred_score.size(0))  # Reshape to [N].
            assert len(mos.size()) == 1 and len(spatial_d_pred_score.size()) == 1  # Check the tensor shapes.

            spatial_d_test_res["MOS"].extend(mos.cpu().tolist())
            spatial_d_test_res["pred_score"].extend(spatial_d_pred_score.tolist())
            spatial_d_test_res["image_name"].extend(name)

            if 'net_spectral_d' in opt:
                if args.test_protocol_spectral == 'patchify':
                    image_patches = patchify(image, patch_size=args.img_size, stride=args.stride)
                    spectral_d_pred_score = spectral_discriminator(image_patches).view(image.size(0), -1).mean(dim=1)
                elif args.test_protocol_spectral == 'random':
                    spectral_d_pred_score = 0
                    for _ in range(args.n_ensemble):
                        top = np.random.randint(0, image.shape[2] - args.img_size[0])
                        left = np.random.randint(0, image.shape[3] - args.img_size[1])
                        spectral_d_pred_score += spectral_discriminator(image[:,:, top: top+args.img_size[0], left: left+args.img_size[1]]).view(image.size(0), -1).mean(dim=1)
                    spectral_d_pred_score /= args.n_ensemble
                else:
                    spectral_d_pred_score = spectral_discriminator(image).view(image.size(0), -1).mean(dim=1)

                spectral_d_pred_score = spectral_d_pred_score.view(spectral_d_pred_score.size(0))  # Reshape to [N].

                spectral_d_test_res["MOS"].extend(mos.cpu().tolist())
                spectral_d_test_res["pred_score"].extend(spectral_d_pred_score.tolist())
                spectral_d_test_res["image_name"].extend(name)
                # print(f"{spatial_d_weight * spatial_d_pred_score}----{spectral_d_weight * spectral_d_pred_score}-----{mos=}")
                pred_score = spatial_d_pred_score + spectral_d_pred_score

                test_res["MOS"].extend(mos.cpu().tolist())
                test_res["pred_score"].extend(pred_score.tolist())
                test_res["image_name"].extend(name)

    # Calculate PLCC, SRCC and KRCC.
    assert len(spatial_d_test_res["MOS"]) == len(spatial_d_test_res["pred_score"]) == len(spatial_d_test_res["image_name"])
    spatial_d_plcc = corr.plcc(spatial_d_test_res["pred_score"], spatial_d_test_res["MOS"])
    spatial_d_fitted_plcc = corr.fitted_plcc(spatial_d_test_res["pred_score"], spatial_d_test_res["MOS"])
    spatial_d_srcc = corr.srcc(spatial_d_test_res["pred_score"], spatial_d_test_res["MOS"])
    spatial_d_krcc = corr.krcc(spatial_d_test_res["pred_score"], spatial_d_test_res["MOS"])

    # Save predictions.
    if 'net_spectral_d' in opt:
        assert len(spectral_d_test_res["MOS"]) == len(spectral_d_test_res["pred_score"]) == len(spectral_d_test_res["image_name"])
        spectral_d_plcc = corr.plcc(spectral_d_test_res["pred_score"], spectral_d_test_res["MOS"])
        spectral_d_fitted_plcc = corr.fitted_plcc(spectral_d_test_res["pred_score"], spectral_d_test_res["MOS"])
        spectral_d_srcc = corr.srcc(spectral_d_test_res["pred_score"], spectral_d_test_res["MOS"])
        spectral_d_krcc = corr.krcc(spectral_d_test_res["pred_score"], spectral_d_test_res["MOS"])

        assert len(test_res["MOS"]) == len(test_res["pred_score"]) == len(test_res["image_name"])
        plcc = corr.plcc(test_res["pred_score"], test_res["MOS"])
        fitted_plcc = corr.fitted_plcc(test_res["pred_score"], test_res["MOS"])
        srcc = corr.srcc(test_res["pred_score"], test_res["MOS"])
        krcc = corr.krcc(test_res["pred_score"], test_res["MOS"])
        print(f"PLCC, fitted_PLCC, SRCC, KRCC:")
        print(f"{plcc:.8f},{fitted_plcc:.8f},{srcc:.8f},{krcc:.8f}")
    else:
        print(f"PLCC, fitted_PLCC, SRCC, KRCC:")
        print(f"{spatial_d_plcc:.8f},{spatial_d_fitted_plcc:.8f},{spatial_d_srcc:.8f},{spatial_d_krcc:.8f}")

    with open(os.path.join(experiment_dir, f"iqa_{args.test_set_name}_{args.stride}_{args.batch_size}_results.txt"), 'w') as iqa_test_logger:
        print("image_name,mos,pred", file=iqa_test_logger)
        for i in range(len(spatial_d_test_res["image_name"])):
            img = spatial_d_test_res["image_name"][i]
            mos = spatial_d_test_res["MOS"][i]
            pred = spatial_d_test_res["pred_score"][i]
            print(f"{img},{mos},{pred}", file=iqa_test_logger)

        if 'net_spectral_d' in opt:
            for i in range(len(spectral_d_test_res["image_name"])):
                img = spectral_d_test_res["image_name"][i]
                mos = spectral_d_test_res["MOS"][i]
                pred = spectral_d_test_res["pred_score"][i]
                print(f"{img},{mos},{pred}", file=iqa_test_logger)

            for i in range(len(test_res["image_name"])):
                img = test_res["image_name"][i]
                mos = test_res["MOS"][i]
                pred = test_res["pred_score"][i]
                print(f"{img},{mos},{pred}", file=iqa_test_logger)
            print(f"PLCC,fitted_PLCC,SRCC,KRCC", file=iqa_test_logger)
            print(f"{plcc:.12f},{fitted_plcc:.12f},{srcc:.12f},{krcc:.12f}", file=iqa_test_logger)
        else:
            print(f"PLCC,fitted_PLCC,SRCC,KRCC", file=iqa_test_logger)
            print(f"{spatial_d_plcc:.12f},{spatial_d_fitted_plcc:.12f},{spatial_d_srcc:.12f},{spatial_d_krcc:.12f}", file=iqa_test_logger)
    # endregion


if __name__ == '__main__':
    main()