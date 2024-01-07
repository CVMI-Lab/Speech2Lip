from numpy.core.numeric import require
from numpy.lib.function_base import quantile
import torch
import torch.nn.functional as F
import numpy as np
from face_tracking.facemodel import Face_3DMM
from face_tracking.util import *
try:
    from face_tracking.render_3dmm import Render_3DMM
except Exception as e:
    print(e)
import os
import sys
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm, trange
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import face_parsing

def standard(x):
    # x = (x - np.mean(x)) / np.std(x)
    # x = (x + 0.5) * 2.0

    x = (x-x.min()) / (x.max() - x.min())
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idname', type=str, default=None)
    parser.add_argument('--id_dir', type=str, default=None)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--parsing_dir', type=str, default=None)
    # parser.add_argument('--img_h', type=int, default=512, help='image height')
    # parser.add_argument('--img_w', type=int, default=512, help='image width')
    parser.add_argument('--dst_mouth_h', type=int, help='lip height')
    parser.add_argument('--dst_mouth_w', type=int, help='lip width')
    parser.add_argument('--focal', type=int, default=None)
    parser.add_argument('--func', type=str, default='compute_3dmm')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--center_point_y_ratio', type=float, default=None)
    
    args = parser.parse_args()
    return args 

def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True

def load_dir(id_dir, input_dir):
    lmss = []
    imgs_paths = []
    input_file_list = sorted(os.listdir(input_dir))
    input_file_list = [input_file for input_file in input_file_list if 'jpg' in input_file]
    # for i in range(start, end):
    for idx, input_file in enumerate(tqdm(input_file_list)):
        # if idx > 51:
        #     break
        input_path = os.path.join(input_dir, input_file)
        # input_path_lms = input_path.replace('images', 'landmarks').replace('jpg', 'lms')
        input_path_lms = os.path.join(id_dir, 'landmarks', input_path.split('/')[-1].replace('jpg', 'lms'))
        if os.path.isfile(input_path_lms):
            lms = np.loadtxt(input_path_lms, dtype=np.float32)
            lmss.append(lms)
            imgs_paths.append(input_path)
            
    lmss_np = np.stack(lmss)
    lmss = torch.as_tensor(lmss_np).cuda()
    return lmss, lmss_np, imgs_paths

def lin_interp(shape, xyd):
    from scipy.interpolate import LinearNDInterpolator
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity
    
def prepare_transform_matrix(euler, trans, batch_size):
    # rot = euler2rot(sel_euler, change_coord_hand=True) # 50, 3, 3
    sel_euler = euler.clone()
    sel_trans = trans.clone()

    sel_euler[:, 2] = -sel_euler[:, 2]
    sel_euler[:, 1] = -sel_euler[:, 1]
    sel_trans[:, 2] = -sel_trans[:, 2]
    sel_trans[:, 1] = -sel_trans[:, 1]

    rot = euler2rot(sel_euler) # 50, 3, 3
    sel_trans = sel_trans.unsqueeze(-1) # 50, 3, 1
    T_ego = torch.cat([rot, sel_trans], -1) # 50, 3, 4 
    zeros = torch.tensor((0, 0, 0, 1), dtype=torch.float).cuda()
    zeros = zeros.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1) # 50, 1, 4 
    T_ego = torch.cat([T_ego, zeros], 1) # 50, 4, 4 
    
    return T_ego

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # Prepare Coordinates shape [b,3,h*w]
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, return_z=False):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points) # bs, 3, 12288
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        if return_z == False:
            return pix_coords
        else:
            return pix_coords, cam_points[:, 2, :].unsqueeze(1).view(self.batch_size, 1, self.height, self.width)

class FaceModel():
    def __init__(self, args):
        self.args = args
        self.id_dim, self.exp_dim, self.tex_dim, self.point_num = 100, 79, 100, 34650
        
        # empirical setting
        if 'obama2' in self.args.id_dir:
            self.canonical_idx = 12
        else:
            self.canonical_idx = 0

        self.model_3dmm = self.generate_model()
        self.lms, self.lms_np, self.img_paths, self.cxy = self.load_data()
        self.batch_size = 50

        input_path = os.path.join(self.id_dir, 'track_params.pt')
        if os.path.exists(input_path):
            self.params_dict = torch.load(input_path)
            self.focal = self.params_dict['focal']
        else:
            self.params_dict = None
            if args.focal:
                self.focal = args.focal
            else:
                self.focal = self.find_focal()

        self.K = np.array([[self.focal, 0, self.img_w/2, 0],
                           [0, self.focal, self.img_h/2, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.inv_K = np.linalg.pinv(self.K)
        self.K = torch.from_numpy(self.K).cuda().unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.inv_K = torch.from_numpy(self.inv_K).cuda().unsqueeze(0).repeat(self.batch_size, 1, 1)

    def generate_model(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_3dmm = Face_3DMM(os.path.join(dir_path, 'face_tracking', '3DMM'),
                            self.id_dim, self.exp_dim, self.tex_dim, self.point_num)
        return model_3dmm

    def load_data(self):
        # self.id_dir = os.path.join('dataset', self.args.idname)
        self.id_dir = self.args.id_dir
        # lms, lms_np, img_paths = load_dir(os.path.join(self.id_dir, 'images'))
        lms, lms_np, img_paths = load_dir(self.args.id_dir, self.args.input)
        self.img_h, self.img_w = cv2.imread(img_paths[0]).shape[:2]
        cxy = torch.tensor((self.img_w/2.0, self.img_h/2.0), dtype=torch.float).cuda()
        return lms, lms_np, img_paths, cxy

    def find_focal(self):
        print('find the best focal...')
        num_frames = self.lms.shape[0]
        sel_ids = np.arange(0, num_frames, 40)
        sel_num = sel_ids.shape[0]
        arg_focal = 1600
        arg_landis = 1e5

        for focal in trange(600, 1500, 100):
            id_para = self.lms.new_zeros((1, self.id_dim), requires_grad=True)
            exp_para = self.lms.new_zeros((sel_num, self.exp_dim), requires_grad=True)
            euler_angle = self.lms.new_zeros((sel_num, 3), requires_grad=True)
            trans = self.lms.new_zeros((sel_num, 3), requires_grad=True)
            trans.data[:, 2] -= 7
            focal_length = self.lms.new_zeros(1, requires_grad=False)
            focal_length.data += focal
            set_requires_grad([id_para, exp_para, euler_angle, trans])

            optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
            optimizer_frame = torch.optim.Adam(
                [euler_angle, trans], lr=.1)

            for iter in range(2000):
                id_para_batch = id_para.expand(sel_num, -1)
                geometry = self.model_3dmm.get_3dlandmarks(
                    id_para_batch, exp_para, euler_angle, trans, focal_length, self.cxy)
                proj_geo = forward_transform(
                    geometry, euler_angle, trans, focal_length, self.cxy)
                loss_lan = cal_lan_loss(
                    proj_geo[:, :, :2], self.lms[sel_ids].detach())
                loss = loss_lan
                optimizer_frame.zero_grad()
                loss.backward()
                optimizer_frame.step()
                if iter % 100 == 0 and False:
                    print(focal, 'pose', iter, loss.item())

            for iter in range(2500):
                id_para_batch = id_para.expand(sel_num, -1)
                geometry = self.model_3dmm.get_3dlandmarks(
                    id_para_batch, exp_para, euler_angle, trans, focal_length, self.cxy)
                proj_geo = forward_transform(
                    geometry, euler_angle, trans, focal_length, self.cxy)
                loss_lan = cal_lan_loss(
                    proj_geo[:, :, :2], self.lms[sel_ids].detach())
                loss_regid = torch.mean(id_para*id_para)
                loss_regexp = torch.mean(exp_para*exp_para)
                loss = loss_lan + loss_regid*0.5 + loss_regexp*0.4
                optimizer_idexp.zero_grad()
                optimizer_frame.zero_grad()
                loss.backward()
                optimizer_idexp.step()
                optimizer_frame.step()
                if iter % 100 == 0 and False:
                    print(focal, 'poseidexp', iter, loss_lan.item(),
                        loss_regid.item(), loss_regexp.item())
                if iter % 1500 == 0 and iter >= 1500:
                    for param_group in optimizer_idexp.param_groups:
                        param_group['lr'] *= 0.2
                    for param_group in optimizer_frame.param_groups:
                        param_group['lr'] *= 0.2
            # print(focal, loss_lan.item(), torch.mean(trans[:, 2]).item())

            if loss_lan.item() < arg_landis:
                arg_landis = loss_lan.item()
                arg_focal = focal

        print('best focal: ', arg_focal)
        return arg_focal

    def save_image(self, output_dir, input_file_list, input_image, ori_image=None, is_bgr=True):
        batch_size = input_image.shape[0]
        for i in range(batch_size):
            output_path = os.path.join(output_dir, input_file_list[i])
            img = input_image[i]
            if ori_image is not None:
                img = torch.cat([img, ori_image[i]], -1)

            img = img.permute(1, 2, 0).cpu().numpy()
            if is_bgr == True:
                cv2.imwrite(output_path, img)
            else:
                cv2.imwrite(output_path, img[:, :, ::-1])
    
    def save_coords(self, output_dir, input_file_list, input_coords):
        batch_size = input_coords.shape[0]
        for i in range(batch_size):
            output_path = os.path.join(output_dir, input_file_list[i].replace('jpg', 'npy'))
            img = input_coords[i]
            img = img.cpu().numpy()
            np.save(output_path, img)

    def compute_3dmm(self):
        '''
        Compute 3DMM parameters
        '''
        print('compute 3dmm paramters...')
        num_frames = self.lms.shape[0]
        id_para = self.lms.new_zeros((1, self.id_dim), requires_grad=True)
        exp_para = self.lms.new_zeros((num_frames, self.exp_dim), requires_grad=True)
        tex_para = self.lms.new_zeros((1, self.tex_dim), requires_grad=True)
        euler_angle = self.lms.new_zeros((num_frames, 3), requires_grad=True)
        trans = self.lms.new_zeros((num_frames, 3), requires_grad=True)
        light_para = self.lms.new_zeros((num_frames, 27), requires_grad=True)
        trans.data[:, 2] -= 7
        focal_length = self.lms.new_zeros(1, requires_grad=True)
        focal_length.data += self.focal

        set_requires_grad([id_para, exp_para, tex_para,
                        euler_angle, trans, light_para])

        optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
        optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=1)

        print('\tregress using landmark loss...')
        for iter in trange(1500):
            id_para_batch = id_para.expand(num_frames, -1)
            geometry = self.model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para, euler_angle, trans, focal_length, self.cxy)
            proj_geo = forward_transform(
                geometry, euler_angle, trans, focal_length, self.cxy)
            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], self.lms.detach())
            loss = loss_lan
            optimizer_frame.zero_grad()
            loss.backward()
            optimizer_frame.step()
            if iter == 1000:
                for param_group in optimizer_frame.param_groups:
                    param_group['lr'] = 0.1
            if iter % 100 == 0 and False:
                print('pose', iter, loss.item())

        for param_group in optimizer_frame.param_groups:
            param_group['lr'] = 0.1

        print('\tregress using landmark loss + regression loss...')
        for iter in trange(2000):
            id_para_batch = id_para.expand(num_frames, -1)
            geometry = self.model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para, euler_angle, trans, focal_length, self.cxy)
            proj_geo = forward_transform(
                geometry, euler_angle, trans, focal_length, self.cxy)
            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], self.lms.detach())
            loss_regid = torch.mean(id_para*id_para)
            loss_regexp = torch.mean(exp_para*exp_para)
            loss = loss_lan + loss_regid*0.5 + loss_regexp*0.4
            optimizer_idexp.zero_grad()
            optimizer_frame.zero_grad()
            loss.backward()
            optimizer_idexp.step()
            optimizer_frame.step()
            if iter % 100 == 0 and False:
                print('poseidexp', iter, loss_lan.item(),
                    loss_regid.item(), loss_regexp.item())
            if iter % 1000 == 0 and iter >= 1000:
                for param_group in optimizer_idexp.param_groups:
                    param_group['lr'] *= 0.2
                for param_group in optimizer_frame.param_groups:
                    param_group['lr'] *= 0.2
        # print(loss_lan.item(), torch.mean(trans[:, 2]).item())

        batch_size = self.batch_size

        device_default = torch.device('cuda:0')
        device_render = torch.device('cuda:0')
        renderer = Render_3DMM(self.focal, self.img_h, self.img_w, batch_size, device_render)

        sel_ids = np.arange(0, num_frames, int(num_frames/batch_size))[:batch_size]
        imgs = []
        for sel_id in sel_ids:
            imgs.append(cv2.imread(self.img_paths[sel_id])[:, :, ::-1])
        imgs = np.stack(imgs)
        sel_imgs = torch.as_tensor(imgs).cuda()
        sel_lms = self.lms[sel_ids]
        sel_light = light_para.new_zeros((batch_size, 27), requires_grad=True)
        set_requires_grad([sel_light])
        optimizer_tl = torch.optim.Adam([tex_para, sel_light], lr=.1)
        optimizer_id_frame = torch.optim.Adam(
            [euler_angle, trans, exp_para, id_para], lr=.01)

        print('\tregress using landmark loss + regression loss + photometric loss...')
        for iter in trange(71):
            sel_exp_para, sel_euler, sel_trans = exp_para[sel_ids], euler_angle[sel_ids], trans[sel_ids]
            sel_id_para = id_para.expand(batch_size, -1)
            geometry = self.model_3dmm.get_3dlandmarks(
                sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, self.cxy)
            proj_geo = forward_transform(
                geometry, sel_euler, sel_trans, focal_length, self.cxy)
            loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
            loss_regid = torch.mean(id_para*id_para)
            loss_regexp = torch.mean(sel_exp_para*sel_exp_para)

            sel_tex_para = tex_para.expand(batch_size, -1)
            sel_texture = self.model_3dmm.forward_tex(sel_tex_para)
            geometry = self.model_3dmm.forward_geo(sel_id_para, sel_exp_para)
            rott_geo = forward_rott(geometry, sel_euler, sel_trans)
            render_imgs = renderer(rott_geo.to(device_render),
                                sel_texture.to(device_render),
                                sel_light.to(device_render))
            render_imgs = render_imgs.to(device_default)
            mask = (render_imgs[:, :, :, 3]).detach() > 0.0

            render_proj = sel_imgs.clone()
            render_proj[mask] = render_imgs[mask][..., :3].byte()
            loss_col = cal_col_loss(render_imgs[:, :, :, :3], sel_imgs.float(), mask)
            loss = loss_col + loss_lan*3 + loss_regid*2.0 + loss_regexp*1.0
            if iter > 50:
                loss = loss_col + loss_lan*0.05 + loss_regid*1.0 + loss_regexp*0.8
            optimizer_tl.zero_grad()
            optimizer_id_frame.zero_grad()
            loss.backward()
            optimizer_tl.step()
            optimizer_id_frame.step()
            if iter % 50 == 0 and iter >= 5:
                for param_group in optimizer_id_frame.param_groups:
                    param_group['lr'] *= 0.2
                for param_group in optimizer_tl.param_groups:
                    param_group['lr'] *= 0.2
                    
        light_mean = torch.mean(sel_light, 0).unsqueeze(0).repeat(num_frames, 1)
        light_para.data = light_mean

        exp_para = exp_para.detach()
        euler_angle = euler_angle.detach()
        trans = trans.detach()
        light_para = light_para.detach()

        print('\tregress using landmark loss + regression loss + photometric loss + lap loss...')
        for i in trange(int((num_frames-1)/batch_size+1)):
            if (i+1)*batch_size > num_frames:
                start_n = num_frames-batch_size
                sel_ids = np.arange(num_frames-batch_size, num_frames)
            else:
                start_n = i*batch_size
                sel_ids = np.arange(i*batch_size, i*batch_size+batch_size)
            imgs = []
            for sel_id in sel_ids:
                imgs.append(cv2.imread(self.img_paths[sel_id])[:, :, ::-1])
            imgs = np.stack(imgs)
            sel_imgs = torch.as_tensor(imgs).cuda()
            sel_lms = self.lms[sel_ids]

            sel_exp_para = exp_para.new_zeros(
                (batch_size, self.exp_dim), requires_grad=True)
            sel_exp_para.data = exp_para[sel_ids].clone()
            sel_euler = euler_angle.new_zeros(
                (batch_size, 3), requires_grad=True)
            sel_euler.data = euler_angle[sel_ids].clone()
            sel_trans = trans.new_zeros((batch_size, 3), requires_grad=True)
            sel_trans.data = trans[sel_ids].clone()
            sel_light = light_para.new_zeros(
                (batch_size, 27), requires_grad=True)
            sel_light.data = light_para[sel_ids].clone()

            set_requires_grad([sel_exp_para, sel_euler, sel_trans, sel_light])

            optimizer_cur_batch = torch.optim.Adam(
                [sel_exp_para, sel_euler, sel_trans, sel_light], lr=0.005)

            sel_id_para = id_para.expand(batch_size, -1).detach()
            sel_tex_para = tex_para.expand(batch_size, -1).detach()

            pre_num = 5
            if i > 0:
                pre_ids = np.arange(
                    start_n-pre_num, start_n)

            for iter in range(50):
                geometry = self.model_3dmm.get_3dlandmarks(
                    sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, self.cxy)
                proj_geo = forward_transform(
                    geometry, sel_euler, sel_trans, focal_length, self.cxy)
                loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
                loss_regexp = torch.mean(sel_exp_para*sel_exp_para)

                sel_geometry = self.model_3dmm.forward_geo(sel_id_para, sel_exp_para)
                sel_texture = self.model_3dmm.forward_tex(sel_tex_para)
                geometry = self.model_3dmm.forward_geo(sel_id_para, sel_exp_para)
                rott_geo = forward_rott(geometry, sel_euler, sel_trans)
                render_imgs = renderer(rott_geo.to(device_render),
                                    sel_texture.to(device_render),
                                    sel_light.to(device_render))
                render_imgs = render_imgs.to(device_default)
                mask = (render_imgs[:, :, :, 3]).detach() > 0.0
                loss_col = cal_col_loss(
                    render_imgs[:, :, :, :3], sel_imgs.float(), mask)

                if i > 0:
                    geometry_lap = self.model_3dmm.forward_geo_sub(id_para.expand(
                        batch_size+pre_num, -1).detach(), torch.cat((exp_para[pre_ids].detach(), sel_exp_para)), self.model_3dmm.rigid_ids)
                    rott_geo_lap = forward_rott(geometry_lap,  torch.cat(
                        (euler_angle[pre_ids].detach(), sel_euler)), torch.cat((trans[pre_ids].detach(), sel_trans)))

                    loss_lap = cal_lap_loss([rott_geo_lap.reshape(rott_geo_lap.shape[0], -1).permute(1, 0)],
                                            [1.0])
                else:
                    geometry_lap = self.model_3dmm.forward_geo_sub(
                        id_para.expand(batch_size, -1).detach(), sel_exp_para, self.model_3dmm.rigid_ids)
                    rott_geo_lap = forward_rott(geometry_lap,  sel_euler, sel_trans)
                    loss_lap = cal_lap_loss([rott_geo_lap.reshape(rott_geo_lap.shape[0], -1).permute(1, 0)],
                                            [1.0])

                loss = loss_col*0.5 + loss_lan*8 + loss_lap*100000 + loss_regexp*1.0
                if iter > 30:
                    loss = loss_col*0.5 + loss_lan*1.5 + loss_lap*100000 + loss_regexp*1.0
                optimizer_cur_batch.zero_grad()
                loss.backward()
                optimizer_cur_batch.step()
            
            exp_para[sel_ids] = sel_exp_para.clone()
            euler_angle[sel_ids] = sel_euler.clone()
            trans[sel_ids] = sel_trans.clone()
            light_para[sel_ids] = sel_light.clone()

        output_path = os.path.join(self.id_dir, 'track_params.pt')
        print('params saved to ', output_path)
        torch.save({'id': id_para.detach().cpu(), 'exp': exp_para.detach().cpu(),
                    'euler': euler_angle.detach().cpu(), 'trans': trans.detach().cpu(),
                    'focal': focal_length.detach().cpu()}, output_path)

    def compute_uv_mapping(self, debug=False):
        '''
        Compute pixel correspondence for training
        '''
        num_frames = self.lms.shape[0]
        batch_size = self.batch_size
        if debug == True:
            output_dir_vis = os.path.join(self.id_dir, 'warp_images_compute_uv_mapping')
            Path(output_dir_vis).mkdir(parents=True, exist_ok=True)
        output_dir = os.path.join(self.id_dir, 'coords')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # output_mask_dir = os.path.join(self.id_dir, 'masks_face')
        # Path(output_mask_dir).mkdir(parents=True, exist_ok=True)

        id_para = self.params_dict['id'].cuda()
        exp_para = self.params_dict['exp'].cuda()
        euler_angle = self.params_dict['euler'].cuda()
        trans = self.params_dict['trans'].cuda()
        focal_length = self.params_dict['focal'].cuda()

        sel_id_para = id_para.cuda().expand(batch_size, -1)
        self.backproject_depth = BackprojectDepth(self.batch_size, self.img_h, self.img_w).cuda()
        self.project_3d = Project3D(self.batch_size, self.img_h, self.img_w).cuda()
        
        canonical_idx = self.canonical_idx
        canonical_frame = cv2.imread(self.img_paths[canonical_idx])
        canonical_frame = torch.from_numpy(canonical_frame).unsqueeze(0).cuda().float()
        canonical_frame = canonical_frame.permute(0, 3, 1, 2).repeat(self.batch_size, 1, 1, 1)
        canonical_exp_para, canonical_euler, canonical_trans = exp_para[canonical_idx:canonical_idx+1].cuda(), euler_angle[canonical_idx:canonical_idx+1].cuda(), trans[canonical_idx:canonical_idx+1].cuda()
        
        T_ego_canonical = prepare_transform_matrix(canonical_euler, canonical_trans, 1).repeat(self.batch_size, 1, 1)
        for i in trange(int((num_frames-1)/batch_size+1)):
            if (i+1)*batch_size > num_frames:
                start_n = num_frames-batch_size
                sel_ids = np.arange(num_frames-batch_size, num_frames)
            else:
                start_n = i*batch_size
                sel_ids = np.arange(i*batch_size, i*batch_size+batch_size)
            
            imgs = []
            input_file_list = []
            for sel_id in sel_ids:
                imgs.append(cv2.imread(self.img_paths[sel_id]))
                input_file_list.append(self.img_paths[sel_id].split('/')[-1])
            imgs = np.stack(imgs)
            imgs = torch.from_numpy(imgs).cuda().float().permute(0, 3, 1, 2)

            sel_exp_para, sel_euler, sel_trans = exp_para[sel_ids].cuda(), euler_angle[sel_ids].cuda(), trans[sel_ids].cuda()
            T_ego = prepare_transform_matrix(sel_euler, sel_trans, self.batch_size)
            T_ego = torch.bmm(T_ego_canonical, torch.inverse(T_ego))

            depth_list = []
            geometry = self.model_3dmm.forward_geo(sel_id_para, sel_exp_para)
            rott_geo = forward_rott(geometry, sel_euler, sel_trans)
            proj_geo = proj_pts(rott_geo, focal_length, self.cxy) # u,v,z
            proj_geo[:,:,2] = -proj_geo[:,:,2]
            for j in range(self.batch_size):
                depth = lin_interp((self.img_h, self.img_w), proj_geo[j].cpu().numpy())
                depth_list.append(depth)

                # if i < 1:
                #     # empirical setting
                #     mask = depth > 0
                #     output_path = os.path.join(output_mask_dir, input_file_list[j])
                #     cv2.imwrite(output_path, mask * 255)
                
            depth_list = np.stack(depth_list)
            depth_list = torch.from_numpy(depth_list).cuda()
            cam_points = self.backproject_depth(depth_list, self.inv_K).float()
            
            pix_coords = self.project_3d(cam_points, self.K, T_ego)
            pix_coords = torch.clamp(pix_coords, -1, 1)
            
            self.save_coords(output_dir, input_file_list, pix_coords)
            if debug == True:
                warped_image = F.grid_sample(canonical_frame, pix_coords)
                self.save_image(output_dir, input_file_list, warped_image, imgs)
        
    def compute_canonical_mask(self, parsing_path='canonical_face_parsing.jpg', model_path='face_parsing/79999_iter.pth',
                                depth_path='depth_face_canonical.npy', face_mask_path='canonical_face_mask.jpg', head_mask_path='canonical_head_mask.jpg'):
        '''
        Compute head mask, face mask, depth for the canonical space
        '''
        input_parsing_dir = self.args.parsing_dir

        id_para = self.params_dict['id'].cuda()
        exp_para = self.params_dict['exp'].cuda()
        euler_angle = self.params_dict['euler'].cuda()
        trans = self.params_dict['trans'].cuda()
        focal_length = self.params_dict['focal'].cuda()

        sel_id_para = id_para.cuda().expand(1, -1)
        canonical_idx = self.canonical_idx

        # parse face
        face_parsing(os.path.join(self.id_dir, parsing_path), self.img_paths[canonical_idx], model_path)
        
        sel_ids = [canonical_idx]
        imgs = []
        input_file_list = []
        for sel_id in sel_ids:
            imgs.append(cv2.imread(self.img_paths[sel_id]))
            input_file_list.append(self.img_paths[sel_id].split('/')[-1])
        imgs = np.stack(imgs)
        imgs = torch.from_numpy(imgs).cuda().float().permute(0, 3, 1, 2)

        sel_exp_para, sel_euler, sel_trans = exp_para[sel_ids].cuda(), euler_angle[sel_ids].cuda(), trans[sel_ids].cuda()
        geometry = self.model_3dmm.forward_geo(sel_id_para, sel_exp_para)
        rott_geo = forward_rott(geometry, sel_euler, sel_trans)
        proj_geo = proj_pts(rott_geo, focal_length, self.cxy) # u,v,z
        proj_geo[:,:,2] = -proj_geo[:,:,2]
        
        depth = lin_interp((self.img_h, self.img_w), proj_geo[0].cpu().numpy())
        output_path = os.path.join(self.id_dir, depth_path)
        np.save(output_path, depth)
        
        face_mask = depth > 0
        output_path = os.path.join(self.id_dir, face_mask_path)
        cv2.imwrite(output_path, face_mask * 255)

        input_path = os.path.join(self.id_dir, parsing_path)
        parsing_img = cv2.imread(input_path)
        head_mask = (parsing_img[:, :, 0] >= 200) & (parsing_img[:, :, 1] <= 50) & (parsing_img[:, :, 2] <= 50)
        output_path = os.path.join(self.id_dir, head_mask_path)
        cv2.imwrite(output_path, head_mask * 255)

    def warp_image(self):
        '''
        Warp facial images to a canonical space to enable further cropping of the lip region.
        '''
        num_frames = self.lms.shape[0]
        batch_size = self.batch_size
        output_dir = os.path.join(self.id_dir, 'warp_images')
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        id_para = self.params_dict['id'].cuda()
        exp_para = self.params_dict['exp'].cuda()
        euler_angle = self.params_dict['euler'].cuda()
        trans = self.params_dict['trans'].cuda()
        focal_length = self.params_dict['focal'].cuda()

        sel_id_para = id_para.cuda().expand(batch_size, -1)
        self.backproject_depth = BackprojectDepth(self.batch_size, self.img_h, self.img_w).cuda()
        self.project_3d = Project3D(self.batch_size, self.img_h, self.img_w).cuda()
        
        canonical_idx = self.canonical_idx
        canonical_exp_para, canonical_euler, canonical_trans = exp_para[canonical_idx:canonical_idx+1].cuda(), euler_angle[canonical_idx:canonical_idx+1].cuda(), trans[canonical_idx:canonical_idx+1].cuda()
        geometry = self.model_3dmm.forward_geo(sel_id_para[:1], canonical_exp_para)
        rott_geo = forward_rott(geometry, canonical_euler, canonical_trans)
        proj_geo = proj_pts(rott_geo, focal_length, self.cxy) # u,v,z
        proj_geo[:,:,2] = -proj_geo[:,:,2]
        canonical_depth = lin_interp((self.img_h, self.img_w), proj_geo[0].cpu().numpy())
        
        canonical_depth = torch.from_numpy(canonical_depth).cuda()
        canonical_depth = canonical_depth.unsqueeze(0).repeat(self.batch_size, 1, 1)
        cam_points = self.backproject_depth(canonical_depth, self.inv_K).float()
        T_ego_canonical = prepare_transform_matrix(canonical_euler, canonical_trans, 1).repeat(self.batch_size, 1, 1)
        for i in trange(int((num_frames-1)/batch_size+1)):
            if (i+1)*batch_size > num_frames:
                start_n = num_frames-batch_size
                sel_ids = np.arange(num_frames-batch_size, num_frames)
            else:
                start_n = i*batch_size
                sel_ids = np.arange(i*batch_size, i*batch_size+batch_size)

            imgs = []
            input_file_list = []
            for sel_id in sel_ids:
                imgs.append(cv2.imread(self.img_paths[sel_id]))
                input_file_list.append(self.img_paths[sel_id].split('/')[-1])
            imgs = np.stack(imgs)
            imgs = torch.from_numpy(imgs).cuda().float().permute(0, 3, 1, 2)

            sel_exp_para, sel_euler, sel_trans = exp_para[sel_ids].cuda(), euler_angle[sel_ids].cuda(), trans[sel_ids].cuda()
            T_ego = prepare_transform_matrix(sel_euler, sel_trans, self.batch_size)
            T_ego = torch.bmm(T_ego, torch.inverse(T_ego_canonical))

            pix_coords = self.project_3d(cam_points, self.K, T_ego)
            warped_image = F.grid_sample(imgs, pix_coords)

            face_mask = canonical_depth > 0
            face_mask = face_mask.unsqueeze(1)
            self.save_image(output_dir, input_file_list, warped_image * face_mask, imgs)

    def generate_mask(self, matte, im_size):
        trimap = (matte >= 0.9).astype('float32')
        not_bg = (matte > 0).astype('float32')

        d_size = im_size // 256 * random.randint(10, 20)
        e_size = im_size // 256 * random.randint(10, 20)

        trimap[np.where((grey_dilation(not_bg, size=(d_size, d_size))
            - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5     
        return trimap
    
    def crop_lip(self, lip_image_path='canonical_lip_mask.jpg'):
        ''' 
        Crop lip image in the canonical space.
        '''
        
        output_dir = os.path.join(self.id_dir, 'images')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        lms = self.lms_np[self.canonical_idx]
        
        mouth_points = [(point[0], point[1]) for point in lms[48:, :2] ]
        x, y, w, h = cv2.boundingRect(np.array(mouth_points)) # x, y, w, h 
        
        # Please adjust the center_point_y_ratio if the lip region is not well included in the cropped images.
        center_point_x = x + w / 2.0
        center_point_y = y + h / 2.0
        if 'adnerf' in self.args.idname:
            center_point_y = y + h / 2.0
        elif 'macron' in self.args.idname:
            center_point_y = (y + h / 2.0) * self.args.center_point_y_ratio 
        else:
            center_point_y = (y + h / 2.0) * 1.02

        w = args.dst_mouth_w 
        h = args.dst_mouth_h 
        x = center_point_x - w / 2.0
        y = center_point_y - h / 2.0

        mask_img = None
        for img_path in tqdm(self.img_paths):
            output_path = img_path.replace('ori_images_face', 'images')
            face_img = cv2.imread(img_path.replace('ori_images_face', 'warp_images'))
            
            if mask_img is None:
                # mask_face_path = img_path.replace('ori_images_face', 'masks_face')
                # mask_face = np.array(cv2.imread(mask_face_path) / 255.0).astype(np.int8)

                mask_img = np.zeros_like(face_img)[:, :self.img_w, 0]
                mask_img[int(y): int(y)+h, int(x): int(x)+w] = 255
                # cv2.imwrite(os.path.join(self.id_dir, 'canonical_lip_mask.jpg'), mask_img * mask_face[:, :, 0])
                cv2.imwrite(os.path.join(self.id_dir, lip_image_path), mask_img)
                
            mouth_img = face_img[int(y): int(y)+h, int(x): int(x)+w, :]
            cv2.imwrite(output_path, mouth_img)

if __name__ == '__main__':
    args = parse_args()
    model = FaceModel(args)

    if args.func == 'compute_3dmm':
        model.compute_3dmm()
    elif args.func == 'warp_image':
        model.warp_image()
    elif args.func == 'compute_uv_mapping':
        model.compute_uv_mapping()
    elif args.func == 'compute_canonical_mask':
        model.compute_canonical_mask()
    elif args.func == 'crop_lip':
        model.crop_lip()