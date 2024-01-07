import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import math
import random
import torch.nn.functional as F
from .SimpleUnetLight import SimpleUnetLight
from .utils import *

class TalkingFace(nn.Module):
    def __init__(self, device, cfg, mode='train',
                use_viewdirs=False, coord_merge_audio=False, 
                W=256, D=8, coord_D=4, skips=[4],
                uv_audio_dims=66, uv_dims=2, audio_dims=29, head_pose_dims=3,
                time_multires=10, 
                output_ch=3, **args):

        super(TalkingFace, self).__init__()
        # Define parameters
        # abandon
        self.use_viewdirs = use_viewdirs
        self.coord_merge_audio = coord_merge_audio  
        self.uv_audio_dims = uv_audio_dims
        self.use_attention = cfg['model']['use_attention']
        # self.use_delta_uv = cfg['model']['use_delta_uv']
        # self.use_coords2audio = cfg['model']['use_coords2audio']
        # default
        self.use_audio_net = cfg['model']['audio_net']
        self.use_uv_audio_sep = cfg['model']['use_uv_audio_sep']
        self.audio_not_embed = cfg['model']['audio_not_embed']
        self.cfg = cfg
        # else
        self.skips = skips
        self.uv_dims = uv_dims
        if cfg['model']['use_audio_mel']:
            self.audio_dims = 80
        else:
            self.audio_dims = audio_dims
        self.head_pose_dims = head_pose_dims
        self.device = device
        self.use_audio = cfg['model']['use_audio']
        self.N_sample = cfg['training']['n_sample_points']
        self.use_head_pose = cfg['model']['use_head_pose']
        self.use_head_pose_net = cfg['model']['use_head_pose_net']
        self.use_time = cfg['model']['use_time']
        self.use_post_fusion = cfg['model']['use_post_fusion']
        self.use_lms = cfg['model']['use_lms']
        self.use_text = cfg['model']['use_text']
        audio_multires = cfg['model']['audio_embed']
        uv_multires = cfg['model']['uv_embed']
        head_pose_multires = cfg['model']['head_pose_multires']
        self.data_path = cfg['data']['path']
        
        if self.use_post_fusion:
            self.use_light_unet = cfg['model']['use_light_unet']
            self.use_resnet = cfg['model']['use_resnet']
            self.post_fusion_channel = cfg['model']['post_fusion_channel']
            self.post_fusion_unet = SimpleUnetLight(cfg=cfg, n_channels=self.post_fusion_channel).to(self.device)
                
        self.expand_lip_mask = cfg['model']['expand_lip_mask']

        if self.use_audio_net:
            self.audio_dims = 64
        self.MLP_version = cfg['model']['MLP_version']
        
        # Define embedder
        if self.use_audio:
            if not self.audio_not_embed:
                self.audio_embedder = Embedder(audio_multires, input_dims=self.audio_dims)
        self.uv_embedder = Embedder(uv_multires, input_dims=self.uv_dims)
        if self.use_head_pose:
            self.head_pose_embedder = Embedder(head_pose_multires, input_dims=self.head_pose_dims)
        if self.use_time:
            input_ch_time = 2 * time_multires
            self.time_embedder_new = PositionalEncodingTime(self.device, input_ch_time)
        if self.use_lms:
            lms_multires = 4
            lms_dims = 40 # 20 x 2
            self.lms_embedder = Embedder(lms_multires, input_dims=lms_dims)

        # Define network
        # audio_net
        if self.use_audio:
            if self.use_audio_net:
                if cfg['model']['use_audio_mel']:
                    input_audio_net = 80
                else:
                    input_audio_net = 29
                self.encoder_conv = nn.Sequential(  # n x 29 x 16
                    nn.Conv1d(input_audio_net, 32, kernel_size=3, stride=2,
                            padding=1, bias=True),  # n x 32 x 8
                    nn.LeakyReLU(0.02, True),
                    nn.Conv1d(32, 32, kernel_size=3, stride=2,
                            padding=1, bias=True),  # n x 32 x 4
                    nn.LeakyReLU(0.02, True),
                    nn.Conv1d(32, 64, kernel_size=3, stride=2,
                            padding=1, bias=True),  # n x 64 x 2
                    nn.LeakyReLU(0.02, True),
                    nn.Conv1d(64, 64, kernel_size=3, stride=2,
                            padding=1, bias=True),  # n x 64 x 1
                    nn.LeakyReLU(0.02, True),
                )
                self.encoder_fc1 = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.LeakyReLU(0.02, True),
                    nn.Linear(64, self.audio_dims),
                )

        if self.use_head_pose:
            if self.use_head_pose_net:
                self.encoder_conv_head_pose = nn.Sequential(  # n x 3 x 5
                    nn.Conv1d(3, 8, kernel_size=3, stride=2,
                            padding=1, bias=True),
                    nn.LeakyReLU(0.02, True),
                    nn.Conv1d(8, 8, kernel_size=3, stride=2,
                            padding=1, bias=True), 
                    nn.LeakyReLU(0.02, True),
                    nn.Conv1d(8, 8, kernel_size=3, stride=2,
                            padding=1, bias=True), 
                    nn.LeakyReLU(0.02, True),
                )
                self.encoder_fc1_head_pose = nn.Sequential(
                    nn.Linear(8, 3),
                    nn.LeakyReLU(0.02, True),
                    nn.Linear(3, 3),
                )

        coord_input = 2
        self.coord_linears = nn.ModuleList(
            [nn.Linear(coord_input, W)] +
            [nn.Linear(W, W) for i in range(coord_D - 1)] +
            [nn.Linear(W, self.audio_dims)]
        )

        input_ch_pts = self.uv_embedder.out_dims
        if self.use_audio:
            if not self.audio_not_embed:
                input_ch_audio = self.audio_embedder.out_dims
            else:
                input_ch_audio = self.audio_dims

        self.output_linear = nn.Linear(W, output_ch)
        
        if self.MLP_version == 'v2':
            # new MLP, add
            hidden_size = W
            self.fc_uv = nn.Linear(input_ch_pts, hidden_size)
            self.fc_uv_skip = nn.Linear(input_ch_pts, W)
            if self.use_audio:
                self.fc_audio = nn.Linear(input_ch_audio, hidden_size)
                self.fc_audio_skip = nn.Linear(input_ch_audio, W)
            if self.use_head_pose:
                input_ch_pose = self.head_pose_embedder.out_dims
                self.fc_pose = nn.Linear(input_ch_pose, hidden_size)
                self.fc_pose_skip = nn.Linear(input_ch_pose, W)
            if self.use_time:
                self.fc_time = nn.Linear(input_ch_time, hidden_size)
                self.fc_time_skip = nn.Linear(input_ch_time, W)
            if self.use_lms:
                input_ch_lms = self.lms_embedder.out_dims
                self.fc_lms = nn.Linear(input_ch_lms, hidden_size)
                self.fc_lms_skip = nn.Linear(input_ch_lms, W)
            
            if self.use_text:
                input_ch_text = 3584
                self.fc_text = nn.Linear(input_ch_text, hidden_size)
                self.fc_text_skip = nn.Linear(input_ch_text, W)
            self.pts_linears = nn.ModuleList(
                [nn.Linear(hidden_size, W)] + 
                [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + hidden_size, W) for i in range(D-1)])
        
        if cfg['model']['use_canonical_depth'] == True:
            if 'canonical_depth_init_path' in cfg['model']:
                # empirical setting
                if 'obama2' in cfg['data']['path']:
                    canonical_idx = 12
                else:
                    canonical_idx = 0

                canonical_depth_head_init = np.load(cfg['model']['canonical_depth_init_path'])
                canonical_depth_head_init = torch.from_numpy(canonical_depth_head_init).float()
                canonical_depth_head = canonical_depth_head_init.clone()
                canonical_depth_head[canonical_depth_head==0] = canonical_depth_head[canonical_depth_head>0].mean() # depth4
                # mask_path = os.path.join(cfg['data']['path'], "masks_head", "{:05d}.jpg".format(int(canonical_idx+1)))
                mask_path = os.path.join(cfg['data']['path'], 'canonical_head_mask.jpg')
                mask_head_canonical = cv2.imread(mask_path) / 255
                mask_head_canonical[mask_head_canonical>0] = 1
                mask_head_canonical = torch.from_numpy(mask_head_canonical[:, :, 0]).int()
                canonical_depth_head[mask_head_canonical==0] = 0 
                canonical_depth_head[canonical_depth_head_init > 0] = canonical_depth_head_init[canonical_depth_head_init > 0]
            else:
                canonical_depth_head = torch.randn((cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width'])) # H, W
            self.canonical_depth_head = torch.nn.Parameter(canonical_depth_head, requires_grad=True)
            
    def audio_merge_forward(self, audio):
        '''
        Args:
            audio: batch x 29 x 16
        '''
        if self.use_audio_net:
            if audio.shape[2] == 16:
                merged_feature = audio # batch x 80 x 16
            else:
                # use DeepSpeech in stage1. Or stage2.
                merged_feature = audio.permute([0, 2, 1]) # batch x 29 x 16

            merged_feature = self.encoder_conv(merged_feature).squeeze(-1)
            merged_feature = self.encoder_fc1(merged_feature)
            return merged_feature
        else:
            return audio

    def head_pose_merge_forward(self, head_pose):
        '''
        Args:
            head_pose_merge_forward: batch x 3 x 5
        '''
        merged_feature = head_pose.permute([0, 2, 1])
        merged_feature = self.encoder_conv_head_pose(merged_feature).squeeze(-1)
        merged_feature = self.encoder_fc1_head_pose(merged_feature)
        return merged_feature
    
    def rgb_forward(self, uv_audio_pts, 
                    time_pts=None, head_pose_pts=None,
                    rgb_pts=None, lms_pts=None,
                    text_pts=None):
        '''
        Args:
            N: batch_size
            uv_audio_pts: [N x 66] or [N x 128]
            head_pose_pts: [N x 3]
        '''
        uv_pts = uv_audio_pts[:, :self.uv_dims]
        if self.use_audio:
            audio_pts = uv_audio_pts[:, self.uv_dims:]
        
        # audio points embedder
        uv_pts = self.uv_embedder(uv_pts) # [N, 42]
        if self.use_audio:
            if not self.audio_not_embed:
                audio_pts = self.audio_embedder(audio_pts) # [N, 832]
        if self.use_head_pose:
            head_pose_pts = self.head_pose_embedder(head_pose_pts) # [N, 63]
        if self.use_time:
            time_pts = self.time_embedder_new(time_pts) # [N, 21]
        if self.use_lms:
            lms_pts = self.lms_embedder(lms_pts) # [N, 360]
        
        if self.MLP_version == 'v2':
            net = self.fc_uv(uv_pts)
            if self.use_audio:
                net += self.fc_audio(audio_pts)
            if self.use_head_pose:
                net += self.fc_pose(head_pose_pts)
            if self.use_time:
                net += self.fc_time(time_pts)
            if self.use_lms:
                net += self.fc_lms(lms_pts)
            if self.use_text:
                net += self.fc_text(text_pts)

            h = net
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h_skip = self.fc_uv_skip(uv_pts)
                    if self.use_audio:
                        h_skip += self.fc_audio_skip(audio_pts)
                    if self.use_head_pose:
                        h_skip += self.fc_pose_skip(head_pose_pts)
                    if self.use_time:
                        h_skip += self.fc_time_skip(time_pts)        
                    if self.use_lms:
                        h_skip += self.fc_lms_skip(lms_pts)
                    if self.use_text:
                        h_skip += self.fc_text_skip(text_pts)

                    h = torch.cat([h_skip, h], -1)

        outputs = self.output_linear(h)
        
        return outputs

    def post_fusion2_onlylip(self, rgb_lip_warped, rgb_face_canonical, rgb_gt, mask_lip_canonical, lip_lefttop_x, lip_lefttop_y, 
                            coord, use_canonical_space=False, change_pose=-1, 
                            mask_face_canonical=None, wav2lip=None,
                            mask_head_observed=None, use_post_fusion_blackaug=False):
        '''
        Args
            rgb_lip_warped: [B, H, W, C]
            rgb_face_canonical: [B, H, W, C]
            rgb_gt: [B, H, W, C]
            mask_lip_canonical: [B, H, W, C]
            change_pose: only works in inference
        Returns
            rgb_recon: [B, H, W, C] -> [B, C, H, W] -> [B, H, W, C]
        '''
        if self.use_light_unet:
            return self.post_fusion2_onlylip_light(rgb_lip_warped, rgb_face_canonical, rgb_gt, 
                                                    mask_lip_canonical, lip_lefttop_x, lip_lefttop_y, coord,
                                                    mask_head_observed, use_post_fusion_blackaug)
    
    def add_black_hole(self, input_img, mask=None):
        output_img = input_img.clone()
        noise = torch.randn(input_img.shape).to(input_img.device)[:, :1, :, :]
        noise[noise < 0.000001] = 0
        noise[noise >= 0.000001] = 1
        
        output_img = output_img * noise

        if mask is not None:
            output_img = output_img * mask + input_img * (1-mask)
            noise = noise * mask + torch.ones_like(noise) * (1-mask)
        
        return output_img, noise

    def post_fusion2_onlylip_light(self, rgb_lip_warped, rgb_face_canonical, 
                                        rgb_gt, mask_lip_canonical, 
                                        lip_lefttop_x, lip_lefttop_y, coord,
                                        mask_head_observed=None, use_post_fusion_blackaug=False):
        '''
        Args
            rgb_lip_warped: [B, H, W, C]
            rgb_face_canonical: [B, H, W, C]
            rgb_gt: [B, H, W, C]
            mask_lip_canonical: [B, H, W, C]
        
        Returns
            rgb_recon: [B, H, W, C] -> [B, C, H, W] -> [B, H, W, C]
        '''
        h, w = rgb_face_canonical.shape[1:3]
        lip_h, lip_w = rgb_lip_warped.shape[1:3]
        rgb_merged = rgb_face_canonical.clone()
        mask_lip_warped = mask_lip_canonical.clone()

        left_padding = lip_lefttop_x - 1
        right_padding = w - (left_padding + lip_w)
        up_padding = lip_lefttop_y - 1
        down_padding = h - (up_padding + lip_h)

        # empirical setting
        if 'macron' in self.data_path or 'obama_adnerf' in self.data_path or 'obama2_face_crop' in self.data_path:
            rgb_lip_warped_padding = F.pad(input=rgb_lip_warped.permute(0, 3, 1, 2), pad=(left_padding+1, right_padding-1, up_padding+1, down_padding-1), mode='constant', value=0).permute(0, 2, 3, 1)
        elif 'may' in self.data_path: 
            rgb_lip_warped_padding = F.pad(input=rgb_lip_warped.permute(0, 3, 1, 2), pad=(left_padding+1, right_padding-1, up_padding+1, down_padding-1), mode='constant', value=0).permute(0, 2, 3, 1)
        else:
            rgb_lip_warped_padding = F.pad(input=rgb_lip_warped.permute(0, 3, 1, 2), pad=(left_padding, right_padding, up_padding, down_padding), mode='constant', value=0).permute(0, 2, 3, 1)
        
        rgb_merged_canonical = mask_lip_warped * rgb_lip_warped_padding + (1-mask_lip_warped) * rgb_merged 

        if self.expand_lip_mask == True:
            # empirical setting
            if 'obama2_face_crop' in self.data_path:
                padding_size = lip_w // 12
            else:
                padding_size = lip_w // 5
            
            mask_lip_warped = torch.ones_like(mask_lip_canonical)
            mask_lip_warped_tmp = torch.zeros_like(mask_lip_canonical)
            mask_lip_warped_tmp[:, lip_lefttop_y-padding_size:lip_lefttop_y+lip_h+2*padding_size, lip_lefttop_x-padding_size:lip_lefttop_x+lip_w+padding_size, :] = 1 
            mask_lip_warped = mask_lip_warped * mask_lip_warped_tmp
        
        rgb_merged = F.grid_sample(rgb_merged_canonical.permute(0, 3, 1, 2), coord, align_corners=False)
        mask_lip_warped = F.grid_sample(mask_lip_warped.float().permute(0, 3, 1, 2), coord, align_corners=False)
        mask_lip_warped[mask_lip_warped!=0] = 1
        mask_lip_warped_new = mask_lip_warped.int()
        
        if use_post_fusion_blackaug == True and random.random() > 0.5:
            mask_face_canonical = (rgb_face_canonical > 0).float()
            mask_face_observed = F.grid_sample(mask_face_canonical.permute(0, 3, 1, 2), coord, align_corners=False)
            # mask_face_observed = mask_face_observed * (mask_head_observed.permute(0, 3, 1, 2))
            mask_face_observed[mask_face_observed != 1] = 0
            
            _, noise_1 = self.add_black_hole(rgb_merged, mask_face_observed)
            noise_1[noise_1 != 0] = 1
            _, noise_2 = self.add_black_hole(rgb_gt.permute(0, 3, 1, 2), mask_face_observed)
            noise_2[noise_2 != 0] = 1
            
            rgb_merged_before = rgb_merged.clone()
            rgb_merged = noise_1 * rgb_merged_before + (1 - noise_1) * rgb_gt.permute(0, 3, 1, 2)
            rgb_gt = (noise_2 * rgb_gt.permute(0, 3, 1, 2) + (1 - noise_2) * rgb_merged_before).permute(0, 2, 3, 1)

        rgb_merged_new = mask_lip_warped_new * rgb_merged + (1-mask_lip_warped_new) * rgb_gt.permute(0, 3, 1, 2)
        rgb_recon = self.post_fusion_unet(rgb_merged_new) 
        
        return rgb_recon.permute(0, 2, 3, 1), rgb_merged_new.permute(0, 2, 3, 1), rgb_merged_canonical
        
class Embedder(object):
    def __init__(self, multires, input_dims=29, include_input=True, log_sampling=True):
        self.multires = multires
        self.input_dims = input_dims
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.max_freq_log2 = multires - 1
        self.num_freqs = multires
        if self.include_input:
            self.out_dims = input_dims + 2 * self.num_freqs * input_dims
        else:
            self.out_dims = 2 * self.num_freqs * input_dims

    def __call__(self, inputs):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d
        
        periodic_fns = [torch.sin, torch.cos]
        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
        
        embed_results = torch.cat([fn(inputs) for fn in embed_fns], -1)

        return embed_results

class PositionalEncodingTime(object):
    def __init__(self, device, out_dims):
        self.out_dims = out_dims
        self.device = device
        self.div_term = torch.exp((torch.arange(0, out_dims, 2, dtype=torch.float) *
                                -(math.log(10000.0) / out_dims))).to(device)

    def __call__(self, position):
        # position: [N, 1]
        # output: [N, out_dims]
        pe = torch.zeros(self.out_dims).to(self.device)
        
        pe[0::2] = torch.sin(position[0].float() * self.div_term)
        pe[1::2] = torch.cos(position[0].float() * self.div_term)
        
        return pe