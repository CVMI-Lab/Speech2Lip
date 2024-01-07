from logging.config import valid_ident
import os
import sys
sys.path.insert(0, './flow_tool/')
import flowlib as fl

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.face_simple.rendering import density2outputs, get_coords
from src.training import BaseTrainer
from src.common import check_weights, tensor2array
from src.face_simple.models.utils import *

import lpips
from torchvision import transforms

class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, device, out_dir, cfg=None, batch_rays=10000,
                 threshold=0.5, n_sample_points=64, n_sample_points_fine=64, 
                 use_audio_net=False, use_head_pose_net=False,
                 lindisp=False, raw_noise_std=0., perturb=True, lambda_rgb=1., multi_gpu=False, local_rank=0, update_pose=False,
                 use_coords2audio=False, use_delta_uv=False,
                 use_canonical_loss=False, use_temp_consist=False, 
                 use_head_pose=False, use_audio=True,
                 use_loss_bg=False, use_loss_face=False, use_loss_facewoaudio=False, use_loss_lip=False,
                 use_coords_mapping=False, 
                 add_noise_uv=False, add_noise_audio=False,
                 use_time=False, use_merge_loss=False,
                 use_post_fusion=False, w_post_fusion=1.0,
                 use_perceptual_loss=False, w_perceptual_loss=1.0,
                 use_syncloss=False, w_syncloss=1.0, use_fusion_face=True, use_c_lip=False,
                 fusion_lip_only=False):
        self.model = model.to(device)
        self.cfg = cfg
        self.audio_dims = self.model.audio_dims
        if multi_gpu:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
        self.multi_gpu = multi_gpu
        self.optimizer = optimizer
        self.device = device
        self.out_dir = out_dir
        self.n_sample_points = n_sample_points
        self.n_sample_points_fine = n_sample_points_fine
        self.lindisp = lindisp
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std
        self.threshold = threshold
        self.w_photometric_loss = lambda_rgb
        self.batch_rays = batch_rays
        self.use_audio_net = use_audio_net
        self.use_head_pose_net = use_head_pose_net
        self.use_coords2audio = use_coords2audio
        self.use_delta_uv = use_delta_uv
        self.use_canonical_loss = use_canonical_loss
        self.use_head_pose = use_head_pose
        self.use_audio = use_audio
        self.use_loss_bg = use_loss_bg
        self.use_loss_face = use_loss_face
        self.use_loss_facewoaudio = use_loss_facewoaudio
        self.use_loss_lip = use_loss_lip
        self.use_coords_mapping = use_coords_mapping
        self.add_noise_uv = add_noise_uv
        self.add_noise_audio = add_noise_audio
        self.use_time = use_time
        self.use_merge_loss = use_merge_loss
        self.use_post_fusion = use_post_fusion
        self.use_post_fusion_wface = self.cfg['model']['use_post_fusion_wface']

        if self.use_post_fusion:
            self.w_post_fusion = w_post_fusion
        self.use_perceptual_loss = use_perceptual_loss
        if self.use_perceptual_loss:
            self.perceptual_loss_fn = lpips.LPIPS(net='alex',version='0.1',model_path='models/lpips_weights_v0.1/alex.pth').to(self.device)
            self.w_perceptual_loss = w_perceptual_loss
            self.use_perceptual_loss_mask = self.cfg['training']['use_perceptual_loss_mask']

        self.use_syncloss = use_syncloss
        self.fusion_lip_only = fusion_lip_only
        
        if self.use_syncloss:
            from src.face_simple.models.syncnet import SyncNet_color as SyncNet
            self.syncnet = SyncNet().to(self.device)
            for p in self.syncnet.parameters():
                p.requires_grad = False
            syncnet_checkpoint_path = 'models/lipsync_expert.pth'
            self.load_checkpoint_syncnet(syncnet_checkpoint_path, self.syncnet)
            self.syncnet.eval()
            
            self.w_syncloss = w_syncloss
            self.use_fusion_face = use_fusion_face
            self.use_c_lip = use_c_lip
            self.use_low_resolution = self.cfg['training']['use_low_resolution'] 

        if self.cfg['model']['post_fusion_warping'] == 'backward':
            self.K = dict()
            self.inv_K = dict()
            self.backproject_depth = dict()
            self.project_3d = dict()
            
            # img_h, img_w = tgt_depth.shape
            img_h = self.cfg['model']['canonical_depth_height']
            img_w = self.cfg['model']['canonical_depth_width']

            focal = cfg['data']['face_img_focal']
            K = np.array([[focal, 0, img_w/2, 0],
                                [0, focal, img_h/2, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=np.float32)
            inv_K = np.linalg.pinv(K)

            self.K[0] = torch.from_numpy(K).to(device).unsqueeze(0)
            self.inv_K[0] = torch.from_numpy(inv_K).to(device).unsqueeze(0)

            img_batch_size = 1
            self.backproject_depth[0] = BackprojectDepth(img_batch_size, cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width'], device=device)
            self.project_3d[0] = Project3D(img_batch_size, cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width']).to(device)

        if self.cfg['training']['fix_post_net'] == True:
            if multi_gpu:
                for p in self.model.module.post_fusion_unet.parameters():
                    p.requires_grad = False
                self.model.module.post_fusion_unet.eval()
            else:
                for p in self.model.post_fusion_unet.parameters():
                    p.requires_grad = False
                self.model.post_fusion_unet.eval()

    def load_checkpoint_syncnet(self, path, model):
        print("Load checkpoint from: {}".format(path))
        checkpoint = torch.load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)
        return model

    def train_step(self, data, data_zero=None, it=None, seed=None):
        ''' Perform a training step

        Args:
            data (tensor): rays data [N, 11] ( rays_o(3), rays_d(3), rgb(3), min_depth(1), max_depth(1))
            it (int): training iteration

        '''

        self.model.train()

        if self.cfg['training']['stage'] == 'stage1':
            # stage1
            loss, loss_all = self.train_stage1(data, it=it, seed=seed)

        return loss.item(), loss_all

    def predict_lip_image(self, i, coords, audio, pose, data, rgb_zero, lms, seed):
        size = coords[i:i+self.batch_rays, :].shape[0]
        coords_batch = coords[i:i+self.batch_rays, :] # 6000, 2
        
        if self.multi_gpu:
            if self.use_audio:
                if self.use_audio_net:
                    audio_batch = self.model.module.audio_merge_forward(audio).unsqueeze(1).tile(1, self.height * self.width, 1).view(-1, self.audio_dims)[i:i+self.batch_rays, :]
                else:
                    audio_batch = audio.view(-1, self.audio_dims)[i:i+self.batch_rays, :]
        else:
            if self.use_audio:
                if self.use_audio_net:
                    audio_batch = self.model.audio_merge_forward(audio).unsqueeze(1).tile(1, self.height * self.width, 1).view(-1, self.audio_dims)[i:i+self.batch_rays, :]
                else:
                    audio_batch = audio.view(-1, self.audio_dims)[i:i+self.batch_rays, :]

        if self.use_audio:
            feature_length = self.audio_dims + 2
        else:
            feature_length = 2

        if self.use_delta_uv:
            time_pts = data['index'] / data['total_frame']
        elif self.use_time:
            if seed is None:
                time_pts = data['index']
            else:
                time_pts = data['index'] + seed # add noise
        else:
            time_pts = None
        
        if self.cfg['model']['use_text']:
            text_pts = data['text'].reshape(1, -1)
        else:
            text_pts = None

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]

        rx = 0.5 / self.width
        ry = 0.5 / self.height
        eps_shift = ry * torch.rand(1, device=self.device) / 2.0

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coords_batch.clone()
                coord_[:, 0] += vx * rx + eps_shift
                coord_[:, 1] += vy * ry + eps_shift
                # coord_.clamp_(0 + 1e-6, 1 - 1e-6)
                coord_.clamp_(0, 1)

                if self.use_audio:
                    if self.add_noise_audio == True:
                        raw_noise_std = 0.01
                        noise = torch.randn(audio_batch.shape, device=self.device) * raw_noise_std
                        audio_batch += noise
                        
                    uv_audio_rays = torch.cat([coord_[:, None, :], audio_batch[:, None, :]], -1) # [N, 1, 2+64] 
                else:
                    uv_audio_rays = coord_[:, None, :]
                    
                if self.multi_gpu:
                    if rgb_zero is None:
                        pred = self.model.module.rgb_forward(uv_audio_rays.view(-1, feature_length),
                                                                time_pts=time_pts, lms_pts=lms, text_pts=text_pts) # N(6000), 3
                    else:
                        pred = self.model.module.rgb_forward(uv_audio_rays.view(-1, feature_length),
                                                                time_pts=time_pts,
                                                                rgb_pts=rgb_zero[i:i+self.batch_rays], lms_pts=lms, text_pts=text_pts) # N(6000), 3
                else:
                    if rgb_zero is None:
                        pred = self.model.rgb_forward(uv_audio_rays.view(-1, feature_length),
                                                                time_pts=time_pts, lms_pts=lms, text_pts=text_pts) # N(6000), 3
                    else:
                        pred = self.model.rgb_forward(uv_audio_rays.view(-1, feature_length),
                                                                time_pts=time_pts, 
                                                                rgb_pts=rgb_zero[i:i+self.batch_rays], lms_pts=lms, text_pts=text_pts)
                
                preds.append(pred)
                area = torch.abs((coord_[:, 0]-coords_batch[:, 0]) * (coord_[:, 1]-coords_batch[:, 1]))
                areas.append(area + 1e-9)
                
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        rgb_map = ret[:, :3]
    
        return rgb_map

    def prepare_coords(self, coord, b):
        if self.use_coords_mapping == True:
            pix_coords = coord.view(-1, 2) # [height * width, 2] [-1, 1]
            pix_coords = torch.clamp(pix_coords, -1, 1)
            coords = pix_coords / 2.0 + 0.5
        else:
            coords = get_coords(int(self.width), int(self.height), self.device, add_noise_uv=self.add_noise_uv, raw_noise_std=0.5/self.width) # [height * width, 2]
        coords = coords.tile(b, 1) # [N, 2]
        return coords

    def compute_rel_pose(self, canonical_euler, canonical_trans, euler, trans, img_batch_size=1, device=None):
        T_ego_canonical = prepare_transform_matrix(canonical_euler, canonical_trans, img_batch_size, device) # .repeat(img_batch_size, 1, 1)
        T_ego = prepare_transform_matrix(euler, trans, img_batch_size, device)
        T_ego = torch.bmm(T_ego, torch.inverse(T_ego_canonical)) # img_batch_size, 4, 4

        return T_ego

    def compute_rel_pose_inverse(self, canonical_euler, canonical_trans, euler, trans, img_batch_size=1, device=None):
        T_ego_canonical = prepare_transform_matrix(canonical_euler, canonical_trans, img_batch_size, device) # .repeat(img_batch_size, 1, 1)
        T_ego = prepare_transform_matrix(euler, trans, img_batch_size, device)
        T_ego = torch.bmm(T_ego, torch.inverse(T_ego_canonical)) # img_batch_size, 4, 4

        return T_ego.inverse()

    def extract_flow(self, pix_coords):
        _, height, width, _ = pix_coords.shape
        device = pix_coords.device
        new_pix_coords = pix_coords.clone()
        # [-1, 1] -> [0, 1] -> [0, w], [b, h, w, 2]
        new_pix_coords = new_pix_coords / 2.0 + 0.5

        new_pix_coords[:, :, :, 0] *= (new_pix_coords.shape[2]-1) # w
        new_pix_coords[:, :, :, 1] *= (new_pix_coords.shape[1]-1) # h

        xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
        meshgrid = np.transpose(np.stack([xx,yy], axis=-1), [2,0,1]) # [2,h,w]
        # cur_pix_coords = torch.from_numpy(meshgrid).unsqueeze(0).repeat(self.opt.batch_size,1,1,1).float().to(self.device) # [b,2,h,w]
        cur_pix_coords = torch.from_numpy(meshgrid).unsqueeze(0).repeat(new_pix_coords.shape[0],1,1,1).float().to(device) # [b,2,h,w]
        cur_pix_coords = cur_pix_coords.permute(0, 2, 3, 1) # [b,h,w,2]

        flow_pred = new_pix_coords - cur_pix_coords
        return flow_pred
        
    def inverse_warping(self, tgt_depth, rel_pose, src_img):
        img_h, img_w = tgt_depth.shape
        focal = self.cfg['data']['face_img_focal']
        K = np.array([[focal, 0, img_w/2, 0],
                           [0, focal, img_h/2, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        inv_K = np.linalg.pinv(K)

        K = torch.from_numpy(K).to(self.device).unsqueeze(0)
        inv_K = torch.from_numpy(inv_K).to(self.device).unsqueeze(0)
        tgt_depth = tgt_depth.unsqueeze(0)
        cam_points = self.backproject_depth[0](tgt_depth, inv_K)
        
        pix_coords, cam_points_z = self.project_3d[0](cam_points, K, rel_pose, return_z=True)
        src_img = src_img.permute(0, 3, 1, 2)
        predict_img = F.grid_sample(src_img, pix_coords, padding_mode="border")

        return predict_img, cam_points_z

    def add_black_hole(self, input_img, mask):
        output_img = input_img.clone()
        noise = torch.randn(input_img.shape).to(input_img.device)[:, :, :, :1]
        noise[noise < 0.000001] = 0
        noise[noise >= 0.000001] = 1
    
        output_img = output_img * noise
        output_img = output_img * mask + input_img * (1-mask)
        
        return output_img

    def project_new_depth_with_new_pose(self, tgt_depth, rel_pose):
        img_h, img_w = tgt_depth.shape

        K = self.K[0]
        inv_K = self.inv_K[0]

        tgt_depth = tgt_depth.unsqueeze(0)
        
        # img_batch_size = 1
        # backproject_depth = BackprojectDepth(img_batch_size, self.cfg['model']['canonical_depth_height'], self.cfg['model']['canonical_depth_width'], self.device=self.device)
        # project_3d = Project3D(img_batch_size, self.cfg['model']['canonical_depth_height'], self.cfg['model']['canonical_depth_width']).to(self.device)
        cam_points = self.backproject_depth[0](tgt_depth, inv_K)
        pix_coords, cam_points_z = self.project_3d[0](cam_points, K, rel_pose, return_z=True)
        
        # print(cam_points_z.shape) # 1, 1, 500, 500
        # input()
        output_depth = self.forward_warping_controllable_with_pix_coords(cam_points_z.permute(0, 2, 3, 1), pix_coords)
        
        return output_depth
        
    def train_stage1(self, data, eval_model=False, it=None, seed=None):
        '''
            batchrify optimize the network
        '''
        # Initialize loss dictionary and other values
        loss = {}
        loss["loss_rgb"] = 0
        if self.use_perceptual_loss:
            loss["loss_perceptual"] = 0
        if self.use_syncloss:
            loss["loss_sync"] = 0
        if self.cfg['training']['use_canonical_depth_loss_photo'] or self.cfg['training']['use_canonical_depth_loss_photo_v2'] \
            or self.cfg['training']['use_canonical_depth_loss_photo_v3'] or self.cfg['training']['use_canonical_depth_loss_photo_v4'] \
            or self.cfg['training']['use_canonical_depth_loss_photo_v5'] \
            or self.cfg['training']['use_canonical_depth_loss_photo_v6']:
            loss["loss_canonical_depth_photo"] = 0
        if self.cfg['training']['use_canonical_depth_loss_geo'] or self.cfg['training']['use_canonical_depth_loss_geo_v2']:
            loss["loss_canonical_depth_geo"] = 0
        if self.cfg['training']['use_canonical_depth_loss_smooth']:
            loss["loss_canonical_depth_smooth"] = 0
            
        if self.use_audio:
            if self.use_audio_net:
                audio = data['audio']# [b, 16, 29]
            else:
                if self.cfg['model']['use_audio_mel']:
                    audio = data['audio'][:, :, 8].unsqueeze(1) # [B, 1, 80]
                else:
                    audio = data['audio'][:, 8, :].unsqueeze(1) # [B, 1, 29]

                audio = audio.tile(1, self.batch_rays, 1) # [batch_size, 29]
        else:
            audio = None 
            
        if self.use_head_pose:
            pose = data['pose'] # [1, 3]
        else:
            pose = None
        
        if self.cfg['model']['use_lms']:
            lms = data['lms'] # [1, 40]
        else:
            lms = None 
            
        # add batch option
        b = data['rgb'].shape[0] # 1?
        self.height = data['rgb'].shape[1]
        self.width = data['rgb'].shape[2]

        # prepare coords
        coords = self.prepare_coords(data['coord'], b)
        
        rgb = data['rgb'].reshape(-1, 3)
        rgb_zero = data['rgb_zero'].reshape(-1, 3)
        mask = None
        
        times = 0
        for i in range(0, b*self.height*self.width, self.batch_rays):
            self.optimizer.zero_grad()
            loss['loss'] = 0

            if mask is not None:
                mask_batch = mask[i:i+self.batch_rays, :]
            else:
                mask_batch = None

            # predict lip image in canonical space
            rgb_map = self.predict_lip_image(i, coords, audio, pose, data, rgb_zero, lms, seed=seed)
            
            # compute loss within lip area
            if self.cfg['training']['use_lip_photo_loss'] == 'v1':
                self.add_photometric_loss(rgb_map, rgb[i:i+self.batch_rays], loss, mask=mask_batch, weights=self.w_photometric_loss)
            
            if self.use_perceptual_loss and self.cfg['training']['use_lip_perc_loss'] == 'v1':
                self.add_perceptual_loss(rgb_map.reshape((1, self.height, self.width, 3)), rgb[i:i+self.batch_rays].reshape((1, self.height, self.width, 3)), loss, weights=self.w_perceptual_loss)

            # fusion
            rgb_face_recon = None
            if self.use_post_fusion:
                rgb_lip = rgb_map.reshape((1, self.height, self.width, 3))
                
                if self.cfg['model']['post_fusion_warping'] == 'backward':
                    rgb_face_canonical = data['rgb_face_zero']
                    if self.multi_gpu:
                        # paste into canonical face, warp to observed space, and fusion
                        rgb_face_gt = data['rgb_face_ori']
                        if self.fusion_lip_only == True:
                            if self.cfg['model']['use_post_fusion_blackaug'] == True:
                                mask_head_observed = None # data['mask_head_3DMM']
                                rgb_face_recon, _, _ = self.model.module.post_fusion2_onlylip(rgb_lip, rgb_face_canonical, rgb_face_gt, 
                                                                                                data['mask_lip_canonical'], data['lip_lefttop_x'], data['lip_lefttop_y'], data['coord'],
                                                                                                mask_head_observed=mask_head_observed, use_post_fusion_blackaug=True)
                    else:
                        # paste into canonical face, warp to observed space, and fusion
                        rgb_face_gt = data['rgb_face_ori']
                        if self.fusion_lip_only == True:
                            if self.cfg['model']['use_post_fusion_blackaug'] == True:
                                mask_head_observed = None # data['mask_head_3DMM']
                                rgb_face_recon, _, _ = self.model.post_fusion2_onlylip(rgb_lip, rgb_face_canonical, rgb_face_gt, 
                                                                                            data['mask_lip_canonical'], data['lip_lefttop_x'], data['lip_lefttop_y'], 
                                                                                            data['coord'],
                                                                                            mask_head_observed=mask_head_observed, use_post_fusion_blackaug=True)
                else:
                    raise NotImplementedError
                
                # compute loss for the whole image
                if self.use_perceptual_loss and self.cfg['training']['use_face_perc_loss'] == True:
                    # perceptual_loss_mask = torch.ones_like(data['mask_face_3DMM']).permute(0, 3, 1, 2)
                    perceptual_loss_mask = torch.ones_like(rgb_face_recon).permute(0, 3, 1, 2)
                    self.add_perceptual_loss(rgb_face_recon, rgb_face_gt, loss, mask=perceptual_loss_mask, weights=self.w_perceptual_loss * self.w_post_fusion)
                
                if self.cfg['training']['use_face_photo_loss'] == True:
                    self.add_photometric_loss(rgb_face_recon, rgb_face_gt, loss, weights=self.w_photometric_loss * self.w_post_fusion)
                
                # train the canonical depth using photometric loss
                if self.cfg['training']['use_canonical_depth_loss_photo_v2'] == True:
                    if self.multi_gpu:
                        tgt_depth = self.model.module.canonical_depth_head # [500, 500]
                    else:
                        tgt_depth = self.model.canonical_depth_head

                    if self.cfg['model']['post_fusion_warping'] == 'backward':
                        rel_pose = self.compute_rel_pose_inverse(data['canonical_euler'], data['canonical_trans'], data['euler'], data['trans'], device=self.device)
                        src_img = rgb_face_gt
                        tgt_depth = tgt_depth

                        rgb_face_canonical_pred, _ = self.inverse_warping(tgt_depth, rel_pose, src_img) # different lips
                        rgb_face_canonical_pred = rgb_face_canonical_pred.permute(0, 2, 3, 1) 
                        
                        loss_mask = data['mask_head_3DMM_canonical'] * (1 - data['mask_face_3DMM_canonical'])
                        self.add_loss_canonical_depth_photo(rgb_face_canonical_pred, rgb_face_canonical, loss, mask=loss_mask)

                # train the canonical depth using 3DMM geometric loss
                # if self.cfg['training']['use_canonical_depth_loss_geo_v2']:
                #     gt_depth = data['depth_face'][0].float()
                #     mask = (gt_depth * cam_points_z[0, 0, :, :]) > 0
                #     self.add_loss_canonical_depth_geo(cam_points_z[0, 0, :, :], gt_depth, loss, mask=mask, weights=1.0)
                # elif self.cfg['training']['use_canonical_depth_loss_geo']:
                #     gt_depth = data['canonical_depth_face'][0].float()
                #     mask = (gt_depth > 0) # * data['mask_head_3DMM_canonical'][0]
                #     self.add_loss_canonical_depth_geo(tgt_depth, gt_depth, loss, mask=mask, weights=1.0)

            # add sync loss
            # if self.use_syncloss and it > 50000:
            if self.use_syncloss and it > 100000 and self.cfg['training']['stage'] == 'stage1':
                # light version
                h, w = data['rgb_face_zero'].shape[1:3]
                lip_h, lip_w = self.height, self.width
                if data['lip_lefttop_y']+lip_h > h:
                    lip_h = h - data['lip_lefttop_y']
                if data['lip_lefttop_x']+lip_w > w:
                    lip_w = w - data['lip_lefttop_x']
                
                frame_cnt = data['audio_window'].shape[1] # B, 3, 5, lip_h, lip_w
                rgb_window = []
                resize_func = transforms.Resize([96, 96])
                
                for frame_idx in range(frame_cnt):
                    cur_coords = self.prepare_coords(None, b) # torch.Size([9600, 2])
                    if self.use_audio_net:
                        cur_audio = data['audio_window'][:, frame_idx, :, :]
                    else:
                        if self.cfg['model']['use_audio_mel']:
                            cur_audio = data['audio_window'][:, frame_idx, :, :][:, :, 8].unsqueeze(1) # [B, 1, 80]
                        else:
                            cur_audio = data['audio_window'][:, frame_idx, :, :][:, 8, :].unsqueeze(1) # [B, 1, 29]
                        
                    cur_pose = None # FIXME: for light version
                    cur_data = dict() # FIXME: for light version
                    if data['index'] + frame_idx < data['total_frame']:
                        cur_data['index'] = data['index'] + frame_idx
                    else:
                        cur_data['index'] = data['total_frame'] - 1
                    cur_data['total_frame'] = data['total_frame']
                    
                    if self.cfg['model']['use_text']:
                        cur_data['text'] = data['text']
                    
                    rgb_lip_warped = self.predict_lip_image(i, cur_coords, cur_audio, cur_pose, cur_data, rgb_zero, lms=None, seed=seed).reshape((1, self.height, self.width, 3)) # B, H, W, 3
                    
                    if self.use_fusion_face == True:
                        # use fusion face, selected
                        rgb_face_gt = data['rgb_face_ori']
                        rgb_face_canonical = data['rgb_face_zero']
                        
                        if self.multi_gpu:
                            if self.fusion_lip_only == True:
                                # use canonical space to fuse
                                rgb_merged, _, _ = self.model.module.post_fusion2_onlylip(rgb_lip_warped, rgb_face_canonical, rgb_face_gt, data['mask_lip_canonical'], data['lip_lefttop_x'], data['lip_lefttop_y'], data['coord_window'][:, frame_idx, :, :, :], use_canonical_space=False)
                        else:
                            if self.fusion_lip_only == True:
                                # use canonical space to fuse
                                rgb_merged, _, _ = self.model.post_fusion2_onlylip(rgb_lip_warped, rgb_face_canonical, rgb_face_gt, data['mask_lip_canonical'], data['lip_lefttop_x'], data['lip_lefttop_y'], data['coord_window'][:, frame_idx, :, :, :], use_canonical_space=False)
                    
                    # crop and resize
                    # FIXME: batch_size must be 1
                    face_x, face_y, face_x2, face_y2, _ = data['canonical_face_bbox'][0]
                    rgb_merged = rgb_merged[:, int(face_y):int(face_y2), int(face_x):int(face_x2), :].clone()
                    rgb_merged = resize_func(rgb_merged.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

                    rgb_merged = rgb_merged.unsqueeze(0)
                    rgb_window.append(rgb_merged)
                
                rgb_window = torch.cat(rgb_window, 0) # T, B, H, W, C -> B, C, T, H, W
                rgb_window = rgb_window.permute(1, 4, 0, 2, 3) # T, B, H, W, C -> B, C, T, H, W
                
                if self.cfg['training']['use_sync_contrastive_loss']:
                    loss_sync = self.get_sync_contrastive_loss(data['mel'], rgb_window, data['rgb_window_neg']) * self.w_syncloss
                
                loss["loss_sync"] += loss_sync
                loss["loss"] += loss_sync

            loss["loss"].backward()
            check_weights(self.model.state_dict())
            self.optimizer.step()
            times += 1
        
        loss["loss_rgb"] /= times
        if self.cfg['training']['use_temp_consist_lip'] == True:
            loss["loss_temp_consist_lip"] /= times
        if self.cfg['training']['use_temp_consist'] == True:
            loss["loss_temp_consist"] /= times
        if self.use_perceptual_loss:
            loss["loss_perceptual"] /= times
        if self.use_syncloss:
            loss["loss_sync"] /= times

        return loss['loss_rgb'], loss

    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        loss = nn.BCELoss()(d.unsqueeze(1), y)
        return loss

    def get_sync_contrastive_loss(self, mel, g_rgb_pos, g_rgb_neg, syncnet_T=5):
        '''
            Args
                g_rgb_pos: [B, C, T, H, W], rgb2bgr
                g_rgb_neg: [B, C, T, H, W], rgb2bgr
        '''
        g = g_rgb_pos[:, [2, 1, 0], :, :, :] # rgb2bgr
        g = g[:, :, :, g.size(3)//2:] # 1, 3, 5, 48, 96
        g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1) # B, 3 * T, H//2, W
        a, v = self.syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().to(self.device)

        sync_loss_pos = self.cosine_loss(a, v, y)
        g = g_rgb_neg[:, [2, 1, 0], :, :, :] # rgb2bgr
        g = g[:, :, :, g.size(3)//2:] # 1, 3, 5, 48, 96
        g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1) # B, 3 * T, H//2, W
        a, v = self.syncnet(mel, g)
        y = torch.zeros(g.size(0), 1).float().to(self.device)
        
        sync_loss_neg = self.cosine_loss(a, v, y)

        sync_loss = sync_loss_pos + sync_loss_neg
        return sync_loss

    def add_photometric_loss(self, prediction, target, loss, coarse=False, mask=None, weights=1.0):
        '''
        Args
            prediction: [B, H, W, 3] or [-1, 3]
            target: [B, H, W, 3] or [-1, 3]
        '''
        if mask is not None:
            loss_rgb = (prediction - target) ** 2 * mask
            loss_rgb = torch.sum(loss_rgb) / (torch.sum(mask) + 1e-6)
        else:
            loss_rgb = torch.mean((prediction - target) ** 2)
        
        loss_rgb = loss_rgb * weights
        loss['loss'] += loss_rgb
        loss['loss_rgb'] += loss_rgb.cpu().detach()
    
    def add_loss_canonical_depth_photo(self, prediction, target, loss, mask=None, weights=1.0):
        '''
        Args
            prediction: [B, H, W, 3] or [-1, 3]
            target: [B, H, W, 3] or [-1, 3]
        '''
        if mask is not None:
            loss_rgb = (prediction - target) ** 2 * mask
            loss_rgb = torch.sum(loss_rgb) / (torch.sum(mask) + 1e-6)
        else:
            loss_rgb = torch.mean((prediction - target) ** 2)
        
        loss_rgb = loss_rgb * weights
        loss['loss'] += loss_rgb
        loss['loss_canonical_depth_photo'] += loss_rgb.cpu().detach()
        if self.cfg['training']['use_face_photo_loss'] == False and 'v' not in self.cfg['training']['use_lip_photo_loss']:
            loss['loss_rgb'] += loss_rgb.cpu().detach()
    
    def add_loss_canonical_depth_geo(self, prediction, target, loss, mask=None, weights=1.0):
        '''
        Args
            prediction: [B, H, W, 3] or [-1, 3]
            target: [B, H, W, 3] or [-1, 3]
        '''
        if mask is not None:
            loss_rgb = (prediction - target) ** 2 * mask.int()
            loss_rgb = torch.sum(loss_rgb) / (torch.sum(mask) + 1e-6)
        else:
            loss_rgb = torch.mean((prediction - target) ** 2)
        
        loss_rgb = loss_rgb * weights
        loss['loss'] += loss_rgb
        loss['loss_canonical_depth_geo'] += loss_rgb.cpu().detach()
    
    def add_perceptual_loss(self, prediction, target, loss, mask=None, weights=1.0):   
        '''
        Args
            prediction: [B, H, W, 3], normalized to [-1, 1]
            target: [B, H, W, 3], normalized to [-1, 1]
            mask: [B, H, W, 1]
        '''
        if mask is not None:
            recon_x = mask * (prediction.permute(0, 3, 1, 2))
            x = mask * (target.permute(0, 3, 1, 2))
        else:
            recon_x = prediction.permute(0, 3, 1, 2)
            x = target.permute(0, 3, 1, 2)

        recon_x = (recon_x - 0.5) * 2
        x = (x - 0.5) * 2

        loss_perceptual = self.perceptual_loss_fn(recon_x, x).mean() * weights
        loss['loss'] += loss_perceptual
        loss['loss_perceptual'] += loss_perceptual.cpu().detach()

    def visualize(self, visualize, logger, it):
        self.model.eval()

        with torch.no_grad():
            image = visualize["rgb"].squeeze()
            image_zero = visualize["rgb_zero"].squeeze()
            rgb_zero = visualize['rgb_zero'].reshape(-1, 3)
            H, W = visualize["height"].data, visualize['width'].data
            
            if self.use_audio:
                if self.use_audio_net:
                    audio = visualize['audio'] # [b, 16, 29]
                else:
                    if self.cfg['model']['use_audio_mel']:
                        audio = visualize['audio'][0, :, 8].unsqueeze(0) 
                    else:
                        audio = visualize['audio'][0, 8, :].unsqueeze(0)

                    audio = audio.tile(self.batch_rays, 1) # [batch_size, 29]
            else:
                audio = None
            if self.use_head_pose:
                pose = visualize['pose'] # [1, 3]
            else:
                pose = None
            if self.cfg['model']['use_lms']:
                lms = visualize['lms']
            else:
                lms = None

            if self.use_coords_mapping == True:
                pix_coords = visualize['coord'].view(-1, 2) # [height * width, 2] [-1, 1]
                pix_coords = torch.clamp(pix_coords, -1, 1)
                coords = pix_coords / 2.0 + 0.5
            else:
                coords = get_coords(int(W), int(H), self.device)
            
            rgb_maps = []
            depth_maps = []
            delta_uv_maps = []

            for i in range(0, self.height*self.width, self.batch_rays):
                rgb_map = self.predict_lip_image(i, coords, audio, pose, visualize, rgb_zero, lms, seed=0)
                rgb_maps.append(rgb_map)

            rgb_map = torch.cat(rgb_maps, 0)
            rgb_map = torch.reshape(rgb_map, (H, W, 3))
            mse = torch.mean((rgb_map.cpu() - image.cpu()) ** 2)
            psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).cpu())
            if logger is None:
                return psnr

            rgb_map = to8b(rgb_map.cpu().numpy()).transpose([2, 0, 1])
            logger.add_image('rgb_prediction', rgb_map, it)
            logger.add_image('rgb_gt', image.cpu().numpy().transpose([2, 0, 1]), it)
            logger.add_scalar('val_mini/loss', mse, it)
            logger.add_scalar('val_mini/psnr', psnr, it)

            if self.use_delta_uv:
                delta_uv_map = torch.cat(delta_uv_maps, 0)
                delta_uv_map = torch.reshape(delta_uv_map, (H, W, 2)).cpu().numpy()
                delta_uv_map = fl.flow_to_image(delta_uv_map)
                logger.add_image('flow', delta_uv_map.transpose([2, 0, 1]), it)

        self.model.train()
    
    def evaluate(self, val_loader, focal_length, batch_size, it):

        psnr_list = []
        for inputs in val_loader:
            psnr = self.visualize(inputs, None, it=it)
            psnr_list.append(psnr)

        psnr = torch.stack(psnr_list, 0).mean()

        return {'psnr': psnr}
