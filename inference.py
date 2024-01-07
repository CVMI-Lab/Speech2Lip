import os, sys
from pickle import TRUE

# sys.path.append(os.path.abspath("../../../"))
# sys.path.insert(0, './flow_tool/')
# sys.path.append(os.path.abspath("../../../flow_tool"))
import flowlib as fl
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange
import re 
import cv2
import timeit

from src import config
from src.face_simple.models import TalkingFace
from src.checkpoints import CheckpointIO
from src.face_simple.rendering import density2outputs, get_coords
from src.data.someones_lip_dataset import SomeonesLipDataset
from src.face_simple.models.utils import *

import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sorted_by_number(v_list):
    return sorted(v_list, key=str2int)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a 3D reconstruction model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--output_dir', type=str, help='output dir name', default="test")
    parser.add_argument('--change_pose', default=-1, type=int, help='controllable')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training ')
    parser.add_argument('--model_iter', default=None, type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--use_new_audio', action='store_true')
    
    args = parser.parse_args()
    return args 

def inference():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args = parse_args()
    abs_path = os.path.abspath("./")
    # config module mainly to load config, dataset, network
    cfg = config.load_config(args.config, 'configs/default.yaml', abs_path=abs_path)
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu", args.local_rank)
    
    height = cfg['data']['height']
    width = cfg['data']['width']
    out_dir = os.path.join(abs_path, cfg['training']['out_dir'])
    batch_size = cfg['training']['batch_rays']
    use_coords2audio = cfg['model']['use_coords2audio']
    use_delta_uv = cfg['model']['use_delta_uv']
    use_head_pose = cfg['model']['use_head_pose']
    use_head_pose_net = cfg['model']['use_head_pose_net']
    use_audio = cfg['model']['use_audio']
    use_coords_mapping = cfg['training']['use_coords_mapping']
    use_time = cfg['model']['use_time']
    use_post_fusion = cfg['model']['use_post_fusion']
    use_post_fusion_wface = cfg['model']['use_post_fusion_wface']
    
    # eval
    fusion_lip_only = cfg['training']['fusion_lip_only']

    model = TalkingFace(device=device, cfg=cfg, mode='eval')
    model = model.eval()
    audio_dims = model.audio_dims

    checkpoint_io = CheckpointIO(out_dir, model=model)
    
    # load model
    if args.model_path is not None: 
        load_dict = checkpoint_io.load(args.model_path, device=device)
    else:
        model_list = os.listdir(out_dir)
        model_list = [model for model in model_list if '.pt' in model and 'model_' in model and 'model_0.pt' not in model]
        model_list = sorted_by_number(model_list)
        try:
            if args.model_iter is not None:
                key = 'model_'+args.model_iter+'.pt'
                load_dict = checkpoint_io.load(key, device=device)
            else:
                key = model_list[-1]
                load_dict = checkpoint_io.load(key, device=device)
            print('load '+key+'...')
        except: 
            load_dict = checkpoint_io.load('model.pt', device=device)
            print('load model.pt...')

    print("Successfully load model!")

    # load data
    dataset_folder = os.path.join(abs_path, cfg['data']['path'])
    if args.use_new_audio:
        test_set = 'test'
    else:
        test_set = 'val'
    dataset = SomeonesLipDataset(dataset_folder, test_set, cfg=cfg, img_ext='.jpg', change_pose=args.change_pose)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # define output path
    test_output_dir = os.path.join("rendering_result", args.output_dir)
    os.makedirs(test_output_dir, exist_ok=True)
    
    if use_post_fusion:
        test_output_post_dir = os.path.join(test_output_dir, "postfusion")
        os.makedirs(test_output_post_dir, exist_ok=True)
    
    seed = 0 
    for data, index in tqdm(test_loader):
        for key, value in data.items():
            data[key] = value.to(device)
            
        audio = data['audio'].tile(batch_size, 1, 1) # [b, 16, 29]
        pose = None
        coords = get_coords(int(width), int(height), device)

        rgb_zero = data['rgb_zero'].reshape(-1, 3)
        with torch.no_grad():
            coords_batch = coords
            audio_batch = model.audio_merge_forward(audio) 
            uv_audio_rays = torch.cat([coords_batch[:, None, :], audio_batch[:, None, :]], -1) # [N, 1, 64*2] 
            feature_length = audio_dims + 2
            time_pts = data['index'] + seed # add noise
            lms = None 
            text_pts = None
            
            outputs = model.rgb_forward(uv_audio_rays.view(-1, feature_length), time_pts=time_pts, rgb_pts=rgb_zero, lms_pts=lms, text_pts=text_pts)
            rgb_map = outputs[:, :3]
            
            if use_post_fusion:
                rgb_lip = rgb_map.reshape((1, height, width, 3))
                rgb_face_canonical = data['rgb_face_zero']
            
                # in canonical space 
                rgb_face_gt = data['rgb_face_ori']
                rgb_face_recon, rgb_face_recon_before, rgb_merged_canonical = model.post_fusion2_onlylip(rgb_lip, rgb_face_canonical, rgb_face_gt, data['mask_lip_canonical'], 
                                                        data['lip_lefttop_x'], data['lip_lefttop_y'], data['coord'],
                                                        use_canonical_space=True, change_pose=args.change_pose, 
                                                        mask_face_canonical=None) # , wav2lip=data['wav2lip'] for wav2lip's lip, use_canonical_space == True results are better
                
        rgb_img = rgb_map.reshape(1, height, width, 3)
        rgb_img = rgb_img.cpu().numpy()[0]
        if use_post_fusion:
            rgb_face_recon = rgb_face_recon[0].cpu().numpy()
            rgb_face_recon = cv2.cvtColor(rgb_face_recon, cv2.COLOR_RGB2BGR)
            output_path = test_output_post_dir + "/{:05d}.jpg".format(int(index.data)+1)
            cv2.imwrite(output_path, rgb_face_recon * 255)
        
if __name__ == "__main__":
    inference()