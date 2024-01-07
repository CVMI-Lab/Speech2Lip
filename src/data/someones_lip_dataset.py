import torch
import torch.utils.data as data
import os
import cv2
import json
import random
import imageio 
import numpy as np
# from torch.utils.tensorboard.summary import audio
import sys
from tqdm import trange 
# import torch.nn.functional as F
from src.data import audio as myaudio 

class SomeonesLipDataset(data.Dataset):
    def __init__(self, dataset_folder, mode, cfg=None,
                img_ext='.png', 
                use_syncloss=False,
                change_pose=-1):
        ''' Dataset for Someone's lip. Data must be loaded IN ORDER.
        structure:
            -- dataset_folder:
                -- images: lip images
                -- audio/audio.npy
                -- coords: 
                -- ori_images_face: original images
                -- warp_images: warped face images
                -- audio_test (optional)
        '''
        super(SomeonesLipDataset, self).__init__()
        self.dataset_folder = dataset_folder
        self.mode = mode
        self.img_ext = img_ext
        self.use_syncloss = cfg['training']['use_syncloss']
        self.cfg = cfg

        # empirical setting
        if 'obama2' in dataset_folder:
            self.canonical_idx = 12
        else:
            self.canonical_idx = 0
        
        # load audio
        audios_folder = os.path.join(dataset_folder, "audio")
        audio_path = os.path.join(audios_folder, 'audio.npy')

        # load pixel corresponding
        self.coords_folder = os.path.join(dataset_folder, "coords")

        # load mask   
        self.mask_3DMM_folder = os.path.join(dataset_folder, "masks_head")
        self.mask_3DMM_folder_face = os.path.join(dataset_folder, "masks_face")
        
        self.ori_images_face_folder = os.path.join(dataset_folder, "ori_images_face") # lip dataset, ori_images_face stands for original face image
        
        # load canonical face image
        rgb_face_zero, _, _ = self.get_color(os.path.join(self.ori_images_face_folder, "{:05d}.jpg".format(int(self.canonical_idx+1))))
        self.rgb_face_zero = torch.Tensor(rgb_face_zero)
        self.face_h, self.face_w = rgb_face_zero.shape[:2]

        # load parsing map
        # self.face_parsing_dir = os.path.join(dataset_folder, "masks")

        self.warp_images_folder = os.path.join(dataset_folder, "warp_images") # all the face images in canonical space
        self.images_folder = os.path.join(dataset_folder, "images")
        self.input_file_list = self.list_dir(self.images_folder, img_ext=self.img_ext)
        mask_path = os.path.join(dataset_folder, 'canonical_lip_mask.jpg')

        self.rgb_zero, self.dst_mouth_h, self.dst_mouth_w = self.get_color(os.path.join(self.images_folder, "{:05d}.jpg".format(int(self.canonical_idx+1)))) # test for val
        self.rgb_zero = torch.Tensor(self.rgb_zero)

        self.mask_lip_canonical = cv2.imread(mask_path) / 255
        self.mask_lip_canonical = torch.Tensor(self.mask_lip_canonical)

        if self.cfg['model']['use_canonical_depth'] == True:
            # canonical_depth_face_path = os.path.join(dataset_folder, "depth_face_canonical.npy")
            # self.canonical_depth_face = np.load(canonical_depth_face_path)
            
            head_pose_6dof_path = os.path.join(dataset_folder, "track_params.pt")
            if os.path.exists(head_pose_6dof_path):
                params_dict = torch.load(head_pose_6dof_path)
                self.pose_features_euler = params_dict['euler']
                self.pose_features_trans = params_dict['trans']
                self.canonical_euler = self.pose_features_euler[self.canonical_idx]
                self.canonical_trans = self.pose_features_trans[self.canonical_idx]
                
            mask_path = os.path.join(dataset_folder, 'canonical_head_mask.jpg')
            self.mask_head_canonical = cv2.imread(mask_path) / 255
            self.mask_head_canonical = torch.Tensor(self.mask_head_canonical[:,:,:1])

            mask_path = os.path.join(dataset_folder, 'canonical_face_mask.jpg')
            self.mask_face_canonical = cv2.imread(mask_path) / 255
            self.mask_face_canonical = torch.Tensor(self.mask_face_canonical)

        lms_path = os.path.join(dataset_folder, "landmarks", "{:05d}.lms".format(int(self.canonical_idx+1)))
        lms = np.loadtxt(lms_path, dtype=np.float32)
        x, y, w, h = self.compute_mouth_bbox(lms)
        self.lefttop_x = int(x)
        self.lefttop_y = int(y)

        print('load audio: ', audio_path)
        self.aud_features = np.load(audio_path)
        
        if 'may' in dataset_folder:
            # female
            self.fmin = 95
        else:
            # male
            self.fmin = 55

        self.coords_file_list = self.list_dir(self.coords_folder, img_ext='.npy')
        
        if self.use_syncloss == True and self.mode == 'train':
            wavpath = os.path.join(self.dataset_folder, "audio", "audio.wav")
            print('load audio\t', wavpath)
            wav = myaudio.load_wav(wavpath, sr=16000)
            self.orig_mel = myaudio.melspectrogram(wav, self.fmin).T
            facebboxpath = os.path.join(self.dataset_folder, "face_bbox_dict.npy")
            print('load face bbox\t', facebboxpath)
            self.face_bbox_dict = np.load(facebboxpath, allow_pickle=True).item()

        if 'lip_train' in dataset_folder:
            length = int(self.aud_features.shape[0])
        else:
            length = int(self.aud_features.shape[0] * 0.9) # same to AD-NeRF
            
        if mode == "train":
            self.input_file_list = self.input_file_list[:length]
            self.aud_features = self.aud_features[:length]
            self.dataset_len = len(self.input_file_list)
            self.data_zero, _ = self.load_one_frame(index=self.canonical_idx)
            self.data_zero['rgb'] = self.data_zero['rgb'].unsqueeze(0)
            self.data_zero['audio'] = self.data_zero['audio'].unsqueeze(0)
            if os.path.exists(self.coords_folder):
                self.coords_file_list = self.coords_file_list[:length]
            if self.cfg['model']['use_canonical_depth'] == True:
                self.pose_features_euler = self.pose_features_euler[:length]
                self.pose_features_trans = self.pose_features_trans[:length]
        elif mode == "val":
            # according to length of audio feature
            if 'may' in dataset_folder:
                length = -598
            elif 'obama2_face_crop' in dataset_folder:
                length = -650
            elif 'obama_adnerf' in dataset_folder:
                length = -800

            self.input_file_list = self.input_file_list[length:]
            self.aud_features = self.aud_features[length:]
            self.dataset_len = len(self.input_file_list)
            if os.path.exists(self.coords_folder):
                self.coords_file_list = self.coords_file_list[length:]
            if self.cfg['model']['use_canonical_depth'] == True:
                self.pose_features_euler = self.pose_features_euler[length:]
                self.pose_features_trans = self.pose_features_trans[length:]
        elif self.mode == "test":
            audios_test_folder = os.path.join(dataset_folder, "audio_test")
            audio_test_path = os.path.join(audios_test_folder, 'audio.npy')
            print('Use test set audio: ', audio_test_path)
            self.aud_features = np.load(audio_test_path)
            self.dataset_len = self.aud_features.shape[0]
            print(self.dataset_len)
        else:
            self.dataset_len = 0

    def list_dir(self, input_dir, img_ext='.jpg'):
        input_file_list = os.listdir(input_dir)
        input_file_list = [input_file for input_file in input_file_list if img_ext in input_file]
        input_file_list = sorted(input_file_list)
        
        return input_file_list

    def compute_mouth_bbox(self, lms):
        mouth_points = [(point[0], point[1]) for point in lms[48:, :2] ]
        x, y, w, h = cv2.boundingRect(np.array(mouth_points)) # x, y, w, h 
        
        # use fixed bbox
        center_point_x = x + w / 2.0
        center_point_y = y + h / 2.0
        
        if 'adnerf' in self.dataset_folder:
            center_point_y = y + h / 2.0
        elif 'macron' in self.dataset_folder:
            center_point_y = (y + h / 2.0) * self.cfg['data']['center_point_y_ratio']
        else:
            center_point_y = (y + h / 2.0) * 1.02

        w = int(self.dst_mouth_w)
        h = int(self.dst_mouth_h)
        x = int(center_point_x - w / 2.0)
        y = int(center_point_y - h / 2.0)

        return x, y, w, h

    def get_color(self, fname, resize_h=None, resize_w=None):
        ''' Load image

        Returns
            cur_frame
            H
            W
        '''
        try:
            cur_frame = imageio.imread(fname)
            if resize_h is not None:
                cur_frame = cv2.resize(cur_frame, (resize_w, resize_h))

        except Exception as e:
            print(e)
            print(fname)
            print()
            print()
            input()

        H, W = cur_frame.shape[:2]
        cur_frame = (np.array(cur_frame) / 255.).astype(np.float32)
        
        return cur_frame, H, W

    def align_lms(self, lands):
        left = lands[49, :]
        right = lands[55, :]
        mid = (left + right) / 2
        
        # translation. Move to (0, 0)
        lands[:, 0] -= mid[0]
        lands[:, 1] -= mid[1]

        # rotate 
        tan_theta = lands[55, 1] / (lands[55, 0] + 1e-6)
        theta = - np.arctan(tan_theta)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        lands = lands.transpose()
        new_lands = np.matmul(rot_matrix, lands).T 

        # rescale
        scale = np.abs(new_lands[55, 0] - new_lands[49, 0]) + 1e-6
        new_lands /= scale

        return new_lands
        
    def load_one_frame(self, index):
        inputs = {}
        
        # load audio
        inputs['audio'] = torch.Tensor(self.aud_features[index])
        inputs['index'] = index
        inputs['total_frame'] = self.dataset_len

        # load coord
        if os.path.exists(self.coords_folder):
            fname_coord = os.path.join(self.coords_folder, self.coords_file_list[index])
            try:
                coord = np.load(fname_coord)
            except Exception as e:
                try:
                    coord = np.load(fname_coord, allow_pickle=True)
                except Exception as e:
                    print(e)
                    print(fname_coord)
                    input()
            inputs['coord'] = torch.Tensor(coord)

        inputs['rgb_face_zero'] = self.rgb_face_zero

        # load mask_lip_canonical
        if self.mask_lip_canonical is not None:
            inputs['mask_lip_canonical'] = self.mask_lip_canonical 
            inputs['lip_lefttop_x'] = self.lefttop_x
            inputs['lip_lefttop_y'] = self.lefttop_y

        if self.cfg['model']['use_post_fusion'] == True \
            or self.mode == 'val' or self.mode == 'test':
            rgb_face_ori, _, _ = self.get_color(os.path.join(self.ori_images_face_folder, self.input_file_list[index]))
            inputs['rgb_face_ori'] = torch.Tensor(rgb_face_ori)

        if self.cfg['model']['use_post_fusion'] == True \
            or self.cfg['training']['use_fusion_face'] == True \
            or self.cfg['training']['use_coords_mapping'] == True:

            # fname_mask = os.path.join(self.mask_3DMM_folder, self.input_file_list[index])
            # mask = cv2.imread(fname_mask) / 255
            # inputs['mask_head_3DMM'] = torch.Tensor(mask).int()

            # if self.mode == 'val' or self.mode == 'test':
            #     fname_mask = os.path.join(self.mask_3DMM_folder_face, self.input_file_list[index])
            #     mask = cv2.imread(fname_mask) / 255
            #     inputs['mask_face_3DMM'] = torch.Tensor(mask).int()

            if self.cfg['training']['use_coords_mapping'] == True:
                fname_mask = os.path.join(self.mask_3DMM_folder_face, self.input_file_list[index])
                mask = cv2.imread(fname_mask) / 255
                inputs['mask_face_3DMM_face'] = torch.Tensor(mask).int()

        if self.cfg['model']['use_canonical_depth'] == True:
            inputs['mask_head_3DMM_canonical'] = self.mask_head_canonical
            inputs['mask_face_3DMM_canonical'] = self.mask_face_canonical

        if self.mode == "test":
            rgb_zero, _, _ = self.get_color(os.path.join(self.images_folder, "{:05d}.jpg".format(int(self.canonical_idx+1))))
            inputs['rgb_zero'] = torch.Tensor(rgb_zero)
            inputs['rgb_face_zero'] = self.rgb_face_zero 

            if self.cfg['model']['use_canonical_depth'] == True:
                inputs['canonical_euler'] = self.canonical_euler
                inputs['canonical_trans'] = self.canonical_trans

                inputs['euler'] = self.pose_features_euler[index]
                inputs['trans'] = self.pose_features_trans[index]

                # inputs['canonical_depth_face'] = self.canonical_depth_face


            return inputs, index

        fname = os.path.join(self.images_folder, self.input_file_list[index])
        
        # load image
        rgb, H, W = self.get_color(fname)
        inputs['rgb'] = torch.Tensor(rgb)

        inputs['rgb_zero'] = self.rgb_zero 
        inputs['height'] = H
        inputs['width'] = W
        inputs["face_h"] = self.face_h
        inputs["face_w"] = self.face_w
        
        if self.use_syncloss == True and self.mode == 'train':
            mel = self.crop_audio_window(self.orig_mel.copy(), index+2)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            inputs['mel'] = mel # 1, 80, 16

            coord_window = []
            coord_prev = inputs['coord']
            for sub_index in range(5):
                cur_index = index + sub_index 
                try:
                    fname_coord = os.path.join(self.coords_folder, self.coords_file_list[cur_index])
                    coord_cur = np.load(fname_coord)
                    coord_window.append(coord_cur)
                    coord_prev = coord_cur
                except:
                    coord_window.append(coord_prev)

            x = np.asarray(coord_window)
            x = torch.FloatTensor(x)
            inputs["coord_window"] = x # 5, H, W, 2

            audio_window = []
            audio_prev = inputs['audio']
            for sub_index in range(5):
                cur_index = index + sub_index 
                try:
                    audio_cur = self.aud_features[cur_index]
                    audio_window.append(audio_cur)
                    audio_prev = audio_cur
                except:
                    audio_window.append(audio_prev)

            x = np.asarray(audio_window)
            x = torch.FloatTensor(x)
            inputs["audio_window"] = x # 5, 16, 29
            inputs["canonical_face_bbox"] = self.face_bbox_dict["{:05d}.jpg".format(int(self.canonical_idx+1))] # x, y, w, h, conf

            if self.cfg['training']['use_sync_contrastive_loss']:
                rgb_window = []
                if index + 5 + 5 < len(self.input_file_list):
                    start_frame = index + 5 
                else:
                    start_frame = index - 10

                for sub_index in range(5):
                    cur_index = start_frame + sub_index 
                    try:
                        fname = os.path.join(self.ori_images_face_folder, self.input_file_list[cur_index])
                        rgb_cur, _, _ = self.get_color(fname, resize_h=96, resize_w=96)
                        rgb_window.append(rgb_cur)
                        rgb_prev = rgb_cur
                    except:
                        rgb_window.append(rgb_prev)

                x = np.asarray(rgb_window)
                x = np.transpose(x, (3, 0, 1, 2))
                x = torch.FloatTensor(x)
                inputs["rgb_window_neg"] = x # 3, 5, lip_h, lip_w
              
        if self.cfg['model']['use_canonical_depth'] == True and (self.mode == 'train' or self.mode == 'val'):
            inputs['canonical_euler'] = self.canonical_euler
            inputs['canonical_trans'] = self.canonical_trans

            inputs['euler'] = self.pose_features_euler[index]
            inputs['trans'] = self.pose_features_trans[index]

            # inputs['canonical_depth_face'] = self.canonical_depth_face

        if self.mode == "train":
            return inputs, index
        else:
            return inputs, index

    def crop_audio_window(self, spec, start_frame, fps=25, syncnet_mel_step_size=16):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            assert NotImplementedError

        start_idx = int(80. * (start_frame_num / float(fps)))
         
        end_idx = start_idx + syncnet_mel_step_size
        if end_idx > spec.shape[0]: 
            start_idx = spec.shape[0] - 16
            end_idx = spec.shape[0]

        return spec[start_idx : end_idx, :]

    def __getitem__(self, index):
        return self.load_one_frame(index)

    def __len__(self):
        return self.dataset_len

def collate_remove_none(batch):
    ''' Collater that puts each data field, except None, into a tensor with outer dimension
        batch size.

    Args: 
        batch: list of dataset with batch size
    '''

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)

if __name__=='__main__':
    use_multi_gpu = False
    datadir = sys.argv[1]
    train_dataset = SomeonesLipDataset(dataset_folder=datadir,
                                        mode='train')
    if use_multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloder = data.DataLoader(dataset=train_dataset,
                         batch_size=8, # FIXME
                         sampler=train_sampler)
    else:
        train_sampler = None
        train_dataloder = data.DataLoader(train_dataset, batch_size=8,
                                    num_workers=0, drop_last=False, shuffle=True)
    
    train_data_iterator = iter(train_dataloder)
    for i in trange(0, 100):
        try:
            inputs = train_data_iterator.next()
            print(inputs['rgb'].shape)
        except StopIteration:
            train_data_iterator = iter(train_dataloder)
            inputs = train_data_iterator.next()
