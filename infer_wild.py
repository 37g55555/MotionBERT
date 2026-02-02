import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, default="../data/mb_res/", help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixel coordinates')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

def main():
    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print('Loading checkpoint', opts.evaluate)
    checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone
    model_pos.eval()
    testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
    }

    os.makedirs(opts.out_path, exist_ok=True)

    if opts.pixel:
        # Keep relative scale with pixel coornidates
        vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
        vid_size = vid.get_meta_data()['size']
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None)
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1])

    test_loader = DataLoader(wild_dataset, **testloader_params)

    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
            else:
                predicted_3d_pos[:,0,0,2]=0
                pass
            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            predicted_3d_pos[...,1] *= -1.0  # flip Y-axis for unity
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    if opts.pixel:
        # Convert to pixel coordinates
        results_all = results_all * (min(vid_size) / 2.0)
        results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0

    # Convert to CSV
    T, nBones, _ = results_all.shape
    rows = []
    for t in range(T):
        for b in range(nBones):
            x, y, z = results_all[t, b]
            rows.append([t, b, x, y, z, 1.0])
    basename = os.path.splitext(os.path.basename(opts.json_path))[0]
    csv_file = os.path.join(opts.out_path, f'{basename}.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "landmark", "x", "y", "z", "visibility"])
        writer.writerows(rows)
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    main()