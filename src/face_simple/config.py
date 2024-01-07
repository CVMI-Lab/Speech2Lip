import os

import numpy as np
import torch
from torch import nn
import torch.distributions as dist
from torchvision import transforms

from src.face_simple import models, training
from src.face_simple.models.tf_nerf import TalkingFace


def get_model(cfg, device=None, dataset=None, **kwargs):
    '''
    Return the Network model. (Option: NeRF or UNISURF)

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    ''' 
    model = TalkingFace(device=device, cfg=cfg)
    return model

def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    lindisp = cfg['training']['lindisp']
    perturb = cfg['training']['perturb']
    raw_noise_std = cfg['training']['raw_noise_std']
    n_sample_points = cfg['training']['n_sample_points']
    n_sample_points_fine = cfg['training']['n_sample_points_fine']
    lambda_rgb = cfg['model']['lambda_rgb']
    multi_gpu = cfg['training']['multi_gpu']
    local_rank = cfg['training']['local_rank']
    batch_rays = cfg['training']['batch_rays']
    use_audio_net = cfg['model']['audio_net']
    use_coords2audio = cfg['model']['use_coords2audio']
    use_delta_uv = cfg['model']['use_delta_uv']
    use_canonical_loss = cfg['training']['use_canonical_loss']
    use_temp_consist = cfg['training']['use_temp_consist']
    use_head_pose = cfg['model']['use_head_pose']
    use_head_pose_net = cfg['model']['use_head_pose_net']
    use_audio = cfg['model']['use_audio']
    use_loss_bg = cfg['training']['use_loss_bg']
    use_loss_face = cfg['training']['use_loss_face']
    use_loss_facewoaudio = cfg['training']['use_loss_facewoaudio']
    use_loss_lip = cfg['training']['use_loss_lip']
    use_coords_mapping = cfg['training']['use_coords_mapping']
    add_noise_uv = cfg['training']['add_noise_uv']
    add_noise_audio = cfg['training']['add_noise_audio']
    use_time = cfg['model']['use_time']
    use_post_fusion = cfg['model']['use_post_fusion']
    w_post_fusion = cfg['training']['w_post_fusion']
    use_perceptual_loss = cfg['training']['use_perceptual_loss']
    w_perceptual_loss = cfg['training']['w_perceptual_loss']
    use_syncloss = cfg['training']['use_syncloss']
    w_syncloss = cfg['training']['w_syncloss']
    use_fusion_face = cfg['training']['use_fusion_face']
    use_c_lip = cfg['training']['use_c_lip']
    fusion_lip_only = cfg['training']['fusion_lip_only'] # fusion lip instead of whole face
    
    trainer = training.Trainer(model, optimizer, device=device, 
        out_dir=out_dir, cfg=cfg,
        threshold=threshold, raw_noise_std=raw_noise_std, 
        n_sample_points=n_sample_points, n_sample_points_fine=n_sample_points_fine, 
        lindisp=lindisp, perturb=perturb, lambda_rgb=lambda_rgb, multi_gpu=multi_gpu, 
        local_rank=local_rank, batch_rays=batch_rays, use_audio_net=use_audio_net, 
        use_coords2audio=use_coords2audio,
        use_delta_uv=use_delta_uv,
        use_canonical_loss=use_canonical_loss,
        use_temp_consist=use_temp_consist,
        use_head_pose=use_head_pose,
        use_head_pose_net=use_head_pose_net,
        use_audio=use_audio, 
        use_loss_bg=use_loss_bg, use_loss_face=use_loss_face, use_loss_facewoaudio=use_loss_facewoaudio, use_loss_lip=use_loss_lip,
        use_coords_mapping=use_coords_mapping,
        add_noise_uv=add_noise_uv, add_noise_audio=add_noise_audio,
        use_time=use_time,
        use_post_fusion=use_post_fusion, w_post_fusion=w_post_fusion,
        use_perceptual_loss=use_perceptual_loss, w_perceptual_loss=w_perceptual_loss,
        use_syncloss=use_syncloss, w_syncloss=w_syncloss, use_fusion_face=use_fusion_face, use_c_lip=use_c_lip,
        fusion_lip_only=fusion_lip_only)

    return trainer

