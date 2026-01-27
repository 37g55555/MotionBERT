import os
import sys
import cv2
import numpy as np
import argparse
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_npy import WildDetDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, default="extract/csv", help='output path')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    #smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def main(data_path):
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

    # Scale to [-1,1]
    wild_dataset = WildDetDataset(data_path, clip_len=opts.clip_len, scale_range=[1,1])
    filename = os.path.basename(data_path)
    basename = os.path.splitext(filename)[0]

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
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)

    # Convert to CSV
    T, nBones, _ = results_all.shape
    rows = []
    for t in range(T):
        for b in range(nBones):
            x, y, z = results_all[t, b]
            rows.append([t, b, x, y, z, 1.0])

    csv_file = os.path.join(opts.out_path, f'{basename}.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "landmark", "x", "y", "z", "visibility"])
        writer.writerows(rows)
    print(f"Results saved to {csv_file}")

def load_raw_keypoints(raw_data):
    """Load BlazePose raw data (T,33,3)."""
    data = np.load(raw_data, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame","landmark","x","y","z","visibility"])

    pts_seq = []

    for f, group in df.groupby("frame"):
        group = group.sort_values("landmark")
        pts = group[["x","y","z"]].values  # (33,3)
        pts_seq.append(pts)

    return np.array(pts_seq)   # (T,33,3)

def extract_3d_keypoints():
    """extract video keypoint ["frame","landmark","x","y","z","visibility"]."""
    NPY_PATH = "extract/npy"
    os.makedirs(NPY_PATH, exist_ok=True)
    filename = os.path.basename(opts.vid_path)
    basename = os.path.splitext(filename)[0]

    cap = cv2.VideoCapture(opts.vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_rows = []

    pbar = tqdm(total=total_frames, desc="Extracting BlazePose 3D", dynamic_ncols=False)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)

        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                pose_rows.append({
                    "frame": frame_idx,
                    "landmark": i,
                    "x": lm.x,
                    "y": -lm.y,      # flip for unity-like coords
                    "z": lm.z,
                    "visibility": lm.visibility,
                })

        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    df = pd.DataFrame(pose_rows)
    raw_data = f"{NPY_PATH}/{basename}.npz"
    np.savez(raw_data, data=df.to_numpy())
    
    reshape_seq = load_raw_keypoints(raw_data)      # (T,33,3)
    reshape_data = f"{NPY_PATH}/{basename}.npy"
    np.save(reshape_data, reshape_seq)
    print(f"Saved raw → {NPY_PATH}")
    os.remove(raw_data)

    return reshape_data

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    data_path = extract_3d_keypoints()
    main(data_path)