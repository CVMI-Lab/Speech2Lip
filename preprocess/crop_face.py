import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

import argparse
import os

class ImageFittingOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--tar_size', type=int, default=256,
                                 help='size for rendering window. We use a square window.')
        self.parser.add_argument('--padding_ratio', type=float, default=0.3,
                                 help='enlarge the face detection bbox by a margin.')
        self.parser.add_argument('--recon_model', type=str, default='bfm09',
                                 help='choose a 3dmm model, default: bfm09')
        self.parser.add_argument('--first_rf_iters', type=int, default=1000,
                                 help='iteration number of rigid fitting for the first frame in video fitting.')
        self.parser.add_argument('--first_nrf_iters', type=int, default=500,
                                 help='iteration number of non-rigid fitting for the first frame in video fitting.')
        self.parser.add_argument('--rest_rf_iters', type=int, default=50,
                                 help='iteration number of rigid fitting for the remaining frames in video fitting.')
        self.parser.add_argument('--rest_nrf_iters', type=int, default=30,
                                 help='iteration number of non-rigid fitting for the remaining frames in video fitting.')
        self.parser.add_argument('--rf_lr', type=float, default=1e-2,
                                 help='learning rate for rigid fitting')
        self.parser.add_argument('--nrf_lr', type=float, default=1e-2,
                                 help='learning rate for non-rigid fitting')
        self.parser.add_argument('--lm_loss_w', type=float, default=100,
                                 help='weight for landmark loss')
        self.parser.add_argument('--rgb_loss_w', type=float, default=1.6,
                                 help='weight for rgb loss')
        self.parser.add_argument('--id_reg_w', type=float, default=1e-3,
                                 help='weight for id coefficient regularizer')
        self.parser.add_argument('--exp_reg_w', type=float, default=0.8e-3,
                                 help='weight for expression coefficient regularizer')
        self.parser.add_argument('--tex_reg_w', type=float, default=1.7e-6,
                                 help='weight for texture coefficient regularizer')
        self.parser.add_argument('--rot_reg_w', type=float, default=1,
                                 help='weight for rotation regularizer')
        self.parser.add_argument('--trans_reg_w', type=float, default=1,
                                 help='weight for translation regularizer')

        self.parser.add_argument('--tex_w', type=float, default=1,
                                 help='weight for texture reflectance loss.')
        self.parser.add_argument('--cache_folder', type=str, default='fitting_cache',
                                 help='path for the cache folder')
        self.parser.add_argument('--nframes_shape', type=int, default=16,
                                 help='number of frames used to estimate shape coefficient in video fitting')
        self.parser.add_argument('--res_folder', type=str, required=True,
                                 help='output path for the image')
        self.parser.add_argument('--res_mask_folder', type=str, default=None)
        self.parser.add_argument('--mouth_dir', type=str, default=None)
        self.parser.add_argument('--face_dir', type=str, default=None)
        self.parser.add_argument('--face_detect_path', type=str, default=None)
        self.parser.add_argument('--face_parsing_dir', type=str, default=None)
        self.parser.add_argument('--dst_mouth_h', type=int, default=0)
        self.parser.add_argument('--dst_mouth_w', type=int, default=0)
        self.parser.add_argument('--dst_face_h', type=int, default=0)
        self.parser.add_argument('--dst_face_w', type=int, default=0)
        self.parser.add_argument('--random_paste', action='store_true')
        self.parser.add_argument('--lms_dir', type=str, default=None)
        self.parser.add_argument('--face_center_point_x', type=int, default=0)
        self.parser.add_argument('--face_center_point_y', type=int, default=0)
        self.parser.add_argument('--img_path', type=str, required=True,
                                 help='path for the image')
        self.parser.add_argument('--gpu', type=int, default=0,
                                 help='gpu device')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt

def crop_face(args):
    print('loading images')
    input_file_list = sorted(os.listdir(args.img_path))
    output_dir = args.res_folder
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print(f"Directory '{output_dir}' already exists.")

    output_dict = dict()
    face_center_point_x = args.face_center_point_x
    face_center_point_y = args.face_center_point_y

    face_w = args.dst_face_w 
    face_h = args.dst_face_h 
    face_x = int(face_center_point_x - face_w / 2.0)
    face_y = int(face_center_point_y - face_h / 2.0)

    for idx, input_file in enumerate(tqdm(input_file_list)):
        output_path = os.path.join(output_dir, input_file).replace('png', 'jpg')
        try:
            input_path = os.path.join(args.img_path, input_file)
            img_arr = cv2.imread(input_path)
            
            face_img = img_arr[face_y: face_y+face_h, face_x: face_x+face_w, :]
            cv2.imwrite(output_path, face_img)
        except Exception as e:
            print(input_path)
            print(e)

if __name__ == '__main__':
    args = ImageFittingOptions()
    args = args.parse()
    args.device = 'cuda:%d' % args.gpu
    crop_face(args)
