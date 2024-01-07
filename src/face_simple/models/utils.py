import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def euler2rot(euler_angle, change_coord_hand=False):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    if change_coord_hand == True:
        psi = -psi
        phi = -phi
        
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

def prepare_transform_matrix(euler, trans, batch_size, device):
    sel_euler = euler.clone()
    sel_trans = trans.clone()

    sel_euler[:, 2] = -sel_euler[:, 2]
    sel_euler[:, 1] = -sel_euler[:, 1]
    sel_trans[:, 2] = -sel_trans[:, 2]
    sel_trans[:, 1] = -sel_trans[:, 1]

    rot = euler2rot(sel_euler) # 50, 3, 3
    sel_trans = sel_trans.unsqueeze(-1) # 50, 3, 1
    T_ego = torch.cat([rot, sel_trans], -1) # 50, 3, 4 
    zeros = torch.tensor((0, 0, 0, 1), dtype=torch.float).to(device)
    zeros = zeros.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1) # 50, 1, 4 
    T_ego = torch.cat([T_ego, zeros], 1) # 50, 4, 4 
    
    return T_ego #.inverse()

def compute_rel_pose_from_obs2can(canonical_euler, canonical_trans, euler, trans, img_batch_size=1, device=None):
    T_ego_canonical = prepare_transform_matrix(canonical_euler, canonical_trans, img_batch_size, device) 
    T_ego = prepare_transform_matrix(euler, trans, img_batch_size, device)
    T_ego = torch.bmm(T_ego_canonical, torch.inverse(T_ego))
    return T_ego

def compute_rel_pose_from_can2obs(canonical_euler, canonical_trans, euler, trans, img_batch_size=1, device=None):
    T_ego_canonical = prepare_transform_matrix(canonical_euler, canonical_trans, img_batch_size, device) # .repeat(img_batch_size, 1, 1)
    T_ego = prepare_transform_matrix(euler, trans, img_batch_size, device)
    T_ego = torch.bmm(T_ego, torch.inverse(T_ego_canonical)) 
    return T_ego

def compute_rel_pose(canonical_euler, canonical_trans, euler, trans, img_batch_size=1, device=None):
    T_ego_canonical = prepare_transform_matrix(canonical_euler, canonical_trans, img_batch_size, device) # .repeat(img_batch_size, 1, 1)
    T_ego = prepare_transform_matrix(euler, trans, img_batch_size, device)
    T_ego = torch.bmm(T_ego, torch.inverse(T_ego_canonical)) # img_batch_size, 4, 4

    return T_ego

def compute_rel_pose_inverse(canonical_euler, canonical_trans, euler, trans, img_batch_size=1, device=None):
    T_ego_canonical = prepare_transform_matrix(canonical_euler, canonical_trans, img_batch_size, device) # .repeat(img_batch_size, 1, 1)
    T_ego = prepare_transform_matrix(euler, trans, img_batch_size, device)
    T_ego = torch.bmm(T_ego, torch.inverse(T_ego_canonical)) # img_batch_size, 4, 4

    return T_ego.inverse()
    
def extract_flow(pix_coords):
    _, height, width, _ = pix_coords.shape
    device = pix_coords.device
    new_pix_coords = pix_coords.clone()
    # [-1, 1] -> [0, 1] -> [0, w], [b, h, w, 2]
    new_pix_coords = new_pix_coords / 2.0 + 0.5

    new_pix_coords[:, :, :, 0] *= (new_pix_coords.shape[2]-1) # w
    new_pix_coords[:, :, :, 1] *= (new_pix_coords.shape[1]-1) # h

    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    meshgrid = np.transpose(np.stack([xx,yy], axis=-1), [2,0,1]) # [2,h,w]
    cur_pix_coords = torch.from_numpy(meshgrid).unsqueeze(0).repeat(new_pix_coords.shape[0],1,1,1).float().to(device) # [b,2,h,w]
    cur_pix_coords = cur_pix_coords.permute(0, 2, 3, 1) # [b,h,w,2]

    flow_pred = new_pix_coords - cur_pix_coords
    return flow_pred

def resize_flow_torch(flow, des_height, des_width):
    '''
    Args
    flow: [b, h, w, 2]

    '''
    src_height, src_width  = flow.shape[1:3]
    ratio_height    = float(des_height) / float(src_height)
    ratio_width     = float(des_width) / float(src_width)
    
    flow = F.interpolate(flow.permute(0, 3, 1, 2), size=(des_height, des_width)).permute(0, 2, 3, 1)

    flow[:, :, :, 0] = flow[:, :, :, 0] * ratio_width
    flow[:, :, : ,1] = flow[:, :, :, 1] * ratio_height
    
    return flow

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width, device=None):
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
                                 requires_grad=False).to(device)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1).to(device)
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

def sample_rel_pose_from_can2obs(edit, euler_index, trans_index, pose_index, canonical_euler, canonical_trans, img_batch_size=1, device=None):
    T_ego_canonical = prepare_transform_matrix(canonical_euler, canonical_trans, img_batch_size, device) # .repeat(img_batch_size, 1, 1)
    euler = canonical_euler.clone()
    trans = canonical_trans.clone()

    if edit == 'euler':
        euler[0, euler_index] = pose_index
    elif edit == 'trans':
        trans[0, trans_index] = pose_index
    else:
        assert NotImplementedError
    T_ego = prepare_transform_matrix(euler, trans, img_batch_size, device)
    T_ego = torch.bmm(T_ego, torch.inverse(T_ego_canonical)) 
    return T_ego

def sample_rel_pose_from_can2obs_inverse(edit, euler_index, trans_index, pose_index, canonical_euler, canonical_trans, img_batch_size=1, device=None):
    T_ego_canonical = prepare_transform_matrix(canonical_euler, canonical_trans, img_batch_size, device) # .repeat(img_batch_size, 1, 1)
    
    euler = canonical_euler.clone()
    trans = canonical_trans.clone()

    if edit == 'euler':
        euler[0, euler_index] = pose_index
    elif edit == 'trans':
        trans[0, trans_index] = pose_index
    else:
        assert NotImplementedError
    T_ego = prepare_transform_matrix(euler, trans, img_batch_size, device)
    T_ego = torch.bmm(T_ego, torch.inverse(T_ego_canonical)) 
    return T_ego.inverse()

def inverse_warping(cfg, tgt_depth, rel_pose, src_img, face_mask, device):
    # print(tgt_depth.shape, rel_pose.shape, src_img.shape) # torch.Size([500, 500]) torch.Size([1, 4, 4]) torch.Size([1, 500, 500, 3])
    img_h, img_w = tgt_depth.shape
    focal = cfg['data']['face_img_focal']
    K = np.array([[focal, 0, img_w/2, 0],
                        [0, focal, img_h/2, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
    inv_K = np.linalg.pinv(K)

    K = torch.from_numpy(K).to(device).unsqueeze(0)
    inv_K = torch.from_numpy(inv_K).to(device).unsqueeze(0)
    tgt_depth = tgt_depth.unsqueeze(0)
    
    img_batch_size = 1
    backproject_depth = BackprojectDepth(img_batch_size, cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width'], device=device)
    project_3d = Project3D(img_batch_size, cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width']).to(device)
    cam_points = backproject_depth(tgt_depth, inv_K)
    pix_coords = project_3d(cam_points, K, rel_pose)

    src_img = src_img.permute(0, 3, 1, 2)
    predict_img = F.grid_sample(src_img, pix_coords, padding_mode="border")

    return predict_img


def forward_warping_controllable(cfg, tgt_depth, rel_pose, src_img, obs_img=None, device=None, half_res=False):
    # re-project
    img_h, img_w = tgt_depth.shape
    focal = cfg['data']['face_img_focal']
    K = np.array([[focal, 0, img_w/2, 0],
                        [0, focal, img_h/2, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
    inv_K = np.linalg.pinv(K)

    K = torch.from_numpy(K).to(device).unsqueeze(0)
    inv_K = torch.from_numpy(inv_K).to(device).unsqueeze(0)
    tgt_depth = tgt_depth.unsqueeze(0)

    img_batch_size = 1
    backproject_depth = BackprojectDepth(img_batch_size, cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width'], device=device)
    project_3d = Project3D(img_batch_size, cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width']).to(device)
    cam_points = backproject_depth(tgt_depth, inv_K)

    if half_res == True:
        # level 1
        scale_factor = 2
        K_half = np.array([[focal/scale_factor, 0, img_w/2/scale_factor, 0],
                            [0, focal/scale_factor, img_h/2/scale_factor, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        inv_K_half = np.linalg.pinv(K_half)
        K_half = torch.from_numpy(K_half).to(device).unsqueeze(0)
        inv_K_half = torch.from_numpy(inv_K_half).to(device).unsqueeze(0)
        tgt_depth_half = F.interpolate(tgt_depth.unsqueeze(1), size=(img_w//scale_factor, img_h//scale_factor)).permute(0, 2, 3, 1)[0]

        backproject_depth_half = BackprojectDepth(img_batch_size, cfg['model']['canonical_depth_height']//scale_factor, cfg['model']['canonical_depth_width']//scale_factor, device=device)
        project_3d_half = Project3D(img_batch_size, cfg['model']['canonical_depth_height']//scale_factor, cfg['model']['canonical_depth_width']//scale_factor).to(device)
        cam_points_half = backproject_depth_half(tgt_depth_half, inv_K_half)
        pix_coords = project_3d_half(cam_points_half, K_half, rel_pose)
    else:
        pix_coords = project_3d(cam_points, K, rel_pose)

    flow = extract_flow(pix_coords)
    fw = forward_warp(interpolation_mode="Nearest")
    src_img = F.interpolate(src_img.permute(0, 3, 1, 2), size=(436, 1024)).permute(0, 2, 3, 1) 
    flow = resize_flow_torch(flow, 436, 1024).contiguous()
    predict_img = fw(src_img.permute(0, 3, 1, 2), flow) 
    predict_img = F.interpolate(predict_img, size=(img_h, img_w)).permute(0, 2, 3, 1) 

    return predict_img

def forward_warping_controllable_depth(cfg, tgt_depth, rel_pose, src_img, obs_img=None, device=None, half_res=False):
    # re-project
    img_h, img_w = tgt_depth.shape
    focal = cfg['data']['face_img_focal']
    K = np.array([[focal, 0, img_w/2, 0],
                        [0, focal, img_h/2, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
    inv_K = np.linalg.pinv(K)

    K = torch.from_numpy(K).to(device).unsqueeze(0)
    inv_K = torch.from_numpy(inv_K).to(device).unsqueeze(0)
    tgt_depth = tgt_depth.unsqueeze(0)

    img_batch_size = 1
    backproject_depth = BackprojectDepth(img_batch_size, cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width'], device=device)
    project_3d = Project3D(img_batch_size, cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width']).to(device)
    cam_points = backproject_depth(tgt_depth, inv_K)

    if half_res == True:
        # level 1
        scale_factor = 2
        K_half = np.array([[focal/scale_factor, 0, img_w/2/scale_factor, 0],
                            [0, focal/scale_factor, img_h/2/scale_factor, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        inv_K_half = np.linalg.pinv(K_half)
        K_half = torch.from_numpy(K_half).to(device).unsqueeze(0)
        inv_K_half = torch.from_numpy(inv_K_half).to(device).unsqueeze(0)
        tgt_depth_half = F.interpolate(tgt_depth.unsqueeze(1), size=(img_w//scale_factor, img_h//scale_factor)).permute(0, 2, 3, 1)[0]

        backproject_depth_half = BackprojectDepth(img_batch_size, cfg['model']['canonical_depth_height']//scale_factor, cfg['model']['canonical_depth_width']//scale_factor, device=device)
        project_3d_half = Project3D(img_batch_size, cfg['model']['canonical_depth_height']//scale_factor, cfg['model']['canonical_depth_width']//scale_factor).to(device)
        cam_points_half = backproject_depth_half(tgt_depth_half, inv_K_half)
        pix_coords, points_z = project_3d_half(cam_points_half, K_half, rel_pose, return_z=True)
    else:
        pix_coords, points_z = project_3d(cam_points, K, rel_pose, return_z=True)
    
    points_z = points_z.permute(0, 2, 3, 1)
    points_z = torch.cat([points_z, points_z, points_z], -1)
    return points_z

def compute_lip_mask_trimap(mask_lip_canonical, lip_h, lip_w, lip_lefttop_x, lip_lefttop_y):
    h, w = mask_lip_canonical.shape[1:3]
    mask_lip_trimap = torch.zeros_like(mask_lip_canonical)[:, :, :, :1]
    # if self.expand_lip_mask == True:
    if True:
        padding_size = lip_w // 5 + 15
        left, right = max(0, lip_lefttop_x-padding_size), min(w, lip_lefttop_x+lip_w+padding_size)
        up, bottom = max(0, lip_lefttop_y-padding_size), min(h, lip_lefttop_y+lip_h+padding_size)
        mask_lip_trimap[:, up:bottom, left:right, :] = 0.5

        left, right = max(0, lip_lefttop_x+padding_size), min(w, lip_lefttop_x+lip_w-padding_size)
        up, bottom = max(0, lip_lefttop_y+padding_size), min(h, lip_lefttop_y+lip_h-padding_size)
        mask_lip_trimap[:, up:bottom, left:right, :] = 1

    padding_size = lip_w // 8
    left, right = max(0, lip_lefttop_x-padding_size), min(w, lip_lefttop_x+lip_w+padding_size)
    up, bottom = max(0, lip_lefttop_y-padding_size), min(h, lip_lefttop_y+lip_h+padding_size)
    mask_lip_trimap[:, up:bottom, left:right, :] = 0.5

    left, right = max(0, lip_lefttop_x+padding_size), min(w, lip_lefttop_x+lip_w-padding_size)
    up, bottom = max(0, lip_lefttop_y+padding_size), min(h, lip_lefttop_y+lip_h-padding_size)
    mask_lip_trimap[:, up:bottom, left:right, :] = 1
    mask_lip_warped_trimap = mask_lip_trimap.permute(0, 3, 1, 2).float().clone() # HIGHLIGHT: clone is important

    return mask_lip_warped_trimap

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
    
def project_new_depth_with_new_pose(cfg, tgt_depth, rel_pose, device):
    img_h, img_w = tgt_depth.shape
    focal = cfg['data']['face_img_focal']
    K = np.array([[focal, 0, img_w/2, 0],
                        [0, focal, img_h/2, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
    inv_K = np.linalg.pinv(K)

    K = torch.from_numpy(K).to(device).unsqueeze(0)
    inv_K = torch.from_numpy(inv_K).to(device).unsqueeze(0)
    tgt_depth = tgt_depth.unsqueeze(0)
    
    img_batch_size = 1
    backproject_depth = BackprojectDepth(img_batch_size, cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width'], device=device)
    project_3d = Project3D(img_batch_size, cfg['model']['canonical_depth_height'], cfg['model']['canonical_depth_width']).to(device)
    cam_points = backproject_depth(tgt_depth, inv_K)
    pix_coords, cam_points_z = project_3d(cam_points, K, rel_pose, return_z=True)
    
    uv = pix_coords / 2 + 0.5
    uv[..., 0] *= img_w
    uv[..., 1] *= img_h

    input_uvd = torch.cat([uv.int(), cam_points_z.permute(0, 2, 3, 1)], -1).squeeze().reshape(-1, 3)
    output_depth = torch.from_numpy(lin_interp((img_h, img_w), input_uvd.cpu().numpy())).to(device)
    
    return output_depth